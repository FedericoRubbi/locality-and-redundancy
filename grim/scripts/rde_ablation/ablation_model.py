import os
import sys
import argparse
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import yaml

import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator, InitProcessGroupKwargs
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.rde import MLPEncoder
from scripts.metric_logging import run_evaluation_suite
from data.data import get_ablation_dataloaders


class BaseAblationRunner(ABC):
    def __init__(self):
        self.config = None
        self.accelerator = None
        self.run_name = None
        self.checkpoint_dir = None
        
        self.enc1 = None
        self.enc2 = None
        self.optimizer1 = None
        self.optimizer2 = None

    def setup_modules(self, input_dim):
        pass

    @abstractmethod
    def compute_loss(self, z1a, z1b, z2a, z2b):
        pass

    def post_eval_hook(self, eval_train_loader, metrics):
        return metrics

    def train_one_run(self, task='both'):
        train_loader, test_loader, input_dim, _ = get_ablation_dataloaders(
            self.config['training']['dataset'],
            self.config['training']['batch_size'],
            self.config['training']['num_workers']
        )
        
        self.enc1 = MLPEncoder(input_dim=input_dim, embed_dim=self.config['training']['embed_dim'])
        self.enc2 = MLPEncoder(input_dim=input_dim, embed_dim=self.config['training']['embed_dim'])
        
        self.setup_modules(input_dim)

        self.optimizer1 = optim.SGD(self.enc1.parameters(), lr=self.config['training']['lr'])
        self.optimizer2 = optim.SGD(self.enc2.parameters(), lr=self.config['training']['lr'])

        self.enc1, self.enc2, self.optimizer1, self.optimizer2, train_loader, test_loader = \
            self.accelerator.prepare(self.enc1, self.enc2, self.optimizer1, self.optimizer2, train_loader, test_loader)

        self.checkpoint_dir = f"checkpoints/ablations/{self.run_name}"

        if self.accelerator.is_main_process:
            run_id_path = os.path.join(self.checkpoint_dir, "wandb_run_id.txt")
            run_id = None
            if task == 'eval':
                try:
                    with open(run_id_path, 'r') as f:
                        run_id = f.read().strip()
                    print(f"Found wandb run ID, resuming run: {run_id}")
                except FileNotFoundError:
                    print(f"Warning: Could not find wandb run ID at {run_id_path}. Starting a new run.")
            
            wandb.init(
                project=self.config['wandb']['project'], 
                name=self.run_name, 
                config=self.config,
                id=run_id,
                resume="allow"
            )

        epochs = self.config['training']['epochs']

        if task in ['train', 'both']:
            for epoch in range(epochs):
                self.enc1.train(); self.enc2.train()
                total_loss, total_batches, alignment_accum, sign_inv_accum = 0.0, 0, 0.0, 0.0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not self.accelerator.is_local_main_process)

                for batch in progress_bar:
                    self.optimizer1.zero_grad()
                    self.optimizer2.zero_grad()

                    (views, _) = batch
                    v1, v2 = views[0].to(self.accelerator.device), views[1].to(self.accelerator.device)
                    
                    z1a, z1b = self.enc1(v1), self.enc1(v2)
                    z2a, z2b = self.enc2(v1), self.enc2(v2)
                    
                    loss1, loss2 = self.compute_loss(z1a, z1b, z2a, z2b)
                    
                    self.accelerator.backward(loss1)
                    self.optimizer1.step()
                    
                    self.accelerator.backward(loss2)
                    self.optimizer2.step()
                    
                    loss = loss1 + loss2
                    align_tensor_a, align_tensor_b = z1a, z2a

                    total_loss += loss.item()
                    total_batches += 1

                    with torch.no_grad():
                        align = F.cosine_similarity(align_tensor_a, align_tensor_b, dim=-1).mean().item()
                        alignment_accum += align
                        coord_product = align_tensor_a * align_tensor_b
                        sign_invariant_align = torch.abs(coord_product.mean(dim=0)).sum().item()
                        sign_inv_accum += sign_invariant_align

                    if self.accelerator.is_local_main_process:
                        wandb.log({'train/loss': loss.item(), 'train/align_batch': align, 'train/sign_inv_batch': sign_invariant_align, 'epoch': epoch+1})

                if self.accelerator.is_main_process:
                    wandb.log({'train/epoch_loss': total_loss / max(1, total_batches), 'train/epoch_align': alignment_accum / max(1, total_batches), 'train/epoch_sign_inv': sign_inv_accum / max(1, total_batches), 'epoch': epoch+1})

            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(os.path.join(self.checkpoint_dir, "wandb_run_id.txt"), 'w') as f:
                    f.write(wandb.run.id)

            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(self.checkpoint_dir)

        if task == 'train':
            if self.accelerator.is_main_process:
                wandb.finish()
            return None

        if task in ['eval', 'both'] and task == 'eval':
            self.accelerator.load_state(self.checkpoint_dir)

        self.enc1.eval(); self.enc2.eval()
        metrics = run_evaluation_suite(self.enc1, self.enc2, (train_loader, test_loader), self.config, self.accelerator, self._collect_embeddings_joint, run_name=self.run_name)
        
        metrics = self.post_eval_hook(train_loader, metrics)

        if self.accelerator.is_main_process:
            wandb.log(metrics)
            wandb.finish()
            return metrics
        return None

    def _collect_embeddings_joint(self, loader, encoder1, encoder2):
        embs1, embs2, labs = [], [], []
        progress_bar = tqdm(loader, desc="Collecting embeddings", disable=not self.accelerator.is_local_main_process)
        for batch in progress_bar:
            data, labels = batch
            imgs = data[0] if isinstance(data, (list, tuple)) else data
            imgs, labels = imgs.to(self.accelerator.device), labels.to(self.accelerator.device)
            
            with torch.no_grad():
                e1 = self.accelerator.unwrap_model(encoder1)(imgs)
                e2 = self.accelerator.unwrap_model(encoder2)(imgs)
            
            if self.accelerator.is_main_process:
                embs1.append(self.accelerator.gather_for_metrics(e1).cpu())
                embs2.append(self.accelerator.gather_for_metrics(e2).cpu())
                labs.append(self.accelerator.gather_for_metrics(labels).cpu())
            else:
                self.accelerator.gather_for_metrics(e1)
                self.accelerator.gather_for_metrics(e2)
                self.accelerator.gather_for_metrics(labels)
                
        if self.accelerator.is_main_process:
            return torch.cat(embs1, dim=0), torch.cat(embs2, dim=0), torch.cat(labs, dim=0)
        return None, None, None

    def run(self, default_config_path):
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='both', choices=['train', 'eval', 'both'])
        parser.add_argument('--config', type=str, default=default_config_path)
        parser.add_argument('--eval-config', type=str, default='configs/ablations/eval.yaml')
        parser.add_argument('--run-name', type=str, default=None)
        args = parser.parse_args()

        with open(args.config, 'r') as f:
            self.config = yaml.safe_load(f)

        if args.eval_config and os.path.exists(args.eval_config):
            with open(args.eval_config, 'r') as f:
                eval_config = yaml.safe_load(f) or {}
            self.config['evaluation'] = eval_config.get('evaluation', eval_config)
        else:
            self.config['evaluation'] = {}

        self.accelerator = Accelerator(
            mixed_precision=self.config.get('accelerator', {}).get('mixed_precision', 'no'),
            kwargs_handlers=[InitProcessGroupKwargs(backend='gloo')]
        )

        if self.accelerator.is_main_process:
            print("======== CONFIGURATION ========\n", yaml.dump(self.config, default_flow_style=False), "===============================")

        self.run_name = args.run_name or os.path.splitext(os.path.basename(args.config))[0]
        res = self.train_one_run(task=args.task)
        if self.accelerator.is_main_process:
            print('Result:', res)