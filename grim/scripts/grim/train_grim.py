import os
import sys
import copy
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
import yaml
from tqdm import tqdm
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

torch.set_float32_matmul_precision('high')

from data.data import get_ablation_dataloaders, get_probe_train_loader
from src.grim_models import Adapter, ConvAdapter, LightConvAdapter, ResNetAdapter, RecurrentEncoder, BidirectionalDecoder
from src.loss import info_nce_loss
from scripts.grim.eval_grim import run_evaluation, run_frontier_evaluation


class Trainer:
    def __init__(self, config_path: str, eval_config_path: str):
        self.init_config(config_path, eval_config_path)
        self.init_wandb()
        self.init_model()

        print(f"Running on {self.device}")

        self.frozen_teacher: BidirectionalDecoder | RecurrentEncoder | None = None
        self._total_training_steps = 0

    def init_config(self, config_path: str, eval_config_path: str) -> None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(eval_config_path, 'r') as f:
            eval_config = yaml.safe_load(f)

        self.config = config
        self.eval_config = eval_config

        self.run_name = config.get('run_name', 'grim_test')
        self.dataset = config['training'].get('dataset', 'mnist')
        self.num_workers = config['training'].get('num_workers', 4)
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs_per_step']
        self.lr = float(config['training']['lr'])
        self.T = config['training']['T']
        self.temperature = config['training']['temperature']

        self.k_anchor = config['training'].get('k_anchor', 4)
        self.anchor_weight = float(config['training'].get('anchor_weight', 1.0))

        self.delta = config['model']['delta']
        self.embed_dim = config['model']['embed_dim']
        self.adapter_type = config['model'].get('adapter_type', 'mlp')
        self.adapter_hidden_dim = config['model'].get('adapter_hidden_dim', 2048)
        self.patch_size = config['model'].get('patch_size', 8)
        self.resnet_arch = config['model'].get('resnet_arch', 'resnet50')
        self.resnet_pretrained = config['model'].get('resnet_pretrained', False)
        self.resnet_small_input = config['model'].get('resnet_small_input', True)
        self.time_dim = config['model'].get('time_dim', 128)
        self.num_blocks = config['model'].get('num_blocks', 3)
        self.mlp_ratio = config['model'].get('mlp_ratio', 4)
        self.dropout = config['model'].get('dropout', 0.1)

        self.adapter_epochs = config['adapter']['epochs']
        self.adapter_lr = float(config['adapter']['lr'])

        decoder_cfg = config.get('decoder', {})
        self.decoder_epochs = decoder_cfg.get('epochs', 0)
        self.decoder_lr = float(decoder_cfg.get('lr', self.lr))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ckpt_dir = Path(f"checkpoints/grim/{self.run_name}")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def init_wandb(self) -> None:
        wandb.init(project="grim", name=self.run_name, config=self.config)

    def init_model(self) -> None:
        train_loader, test_loader, flat_dim, num_classes = get_ablation_dataloaders(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.train_loader, self.test_loader = train_loader, test_loader
        self.probe_train_loader = get_probe_train_loader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.flat_dim = flat_dim
        self.num_classes = num_classes

        if self.adapter_type == 'conv':
            (vA, _), _ = train_loader.dataset[0]
            self.adapter = ConvAdapter(tuple(vA.shape), embed_dim=self.embed_dim, patch_size=self.patch_size).to(self.device)
        elif self.adapter_type == 'lightconv':
            (vA, _), _ = train_loader.dataset[0]
            self.adapter = LightConvAdapter(tuple(vA.shape), embed_dim=self.embed_dim).to(self.device)
        elif self.adapter_type == 'resnet':
            (vA, _), _ = train_loader.dataset[0]
            self.adapter = ResNetAdapter(
                tuple(vA.shape), embed_dim=self.embed_dim, arch=self.resnet_arch,
                pretrained=self.resnet_pretrained, small_input=self.resnet_small_input).to(self.device)
        else:
            self.adapter = Adapter(flat_dim, embed_dim=self.embed_dim, hidden_dim=self.adapter_hidden_dim).to(self.device)

        self.encoder = RecurrentEncoder(num_features=self.embed_dim, time_dim=self.time_dim, num_blocks=self.num_blocks,
                                        dropout=self.dropout, delta=self.delta, mlp_ratio=self.mlp_ratio).to(self.device)
        self.decoder = BidirectionalDecoder(num_features=self.embed_dim, time_dim=self.time_dim, num_blocks=self.num_blocks,
                                            dropout=self.dropout, delta=self.delta, mlp_ratio=self.mlp_ratio).to(self.device)

        self.opt_adapter = Adam(self.adapter.parameters(), lr=self.adapter_lr)
        self.opt_enc = Adam(self.encoder.parameters(), lr=self.lr)
        self.opt_dec = Adam(self.decoder.parameters(), lr=self.decoder_lr)

    def train_adapter(self) -> None:
        print(f"\n=== Phase 1: Adapter Pre-training ({self.adapter_epochs} epochs) ===")
        self.adapter.train()
        for epoch in range(self.adapter_epochs):
            pbar = tqdm(self.train_loader, desc=f"Adapter Epoch {epoch+1}/{self.adapter_epochs}")
            for (vA, vB), _ in pbar:
                vA, vB = vA.to(self.device), vB.to(self.device)
                h0A = self.adapter(vA)
                h0B = self.adapter(vB)
                loss = info_nce_loss(h0A, h0B, temperature=self.temperature)

                self.opt_adapter.zero_grad()
                loss.backward()
                self.opt_adapter.step()

                wandb.log({"adapter/loss": loss.item()})
                pbar.set_postfix({'L_adapter': loss.item()})

        self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad = False
        torch.save(self.adapter.state_dict(), self.ckpt_dir / "adapter.pt")
        print(f"Adapter frozen and saved to {self.ckpt_dir / 'adapter.pt'}")

    def precompute_trajectory(self, h0A: torch.Tensor, h0B: torch.Tensor,
                              stage_limit: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if stage_limit == 0:
            return [], []
        assert self.frozen_teacher is not None, "teacher must exist for stage_limit >= 1"

        B = h0A.shape[0]
        targets_A: list[torch.Tensor] = []
        targets_B: list[torch.Tensor] = []
        with torch.no_grad():
            t_curr_A = h0A.to(torch.bfloat16)
            t_curr_B = h0B.to(torch.bfloat16)
            for s_idx in range(stage_limit):
                s_emb = torch.full((B, 1), float(s_idx), device=self.device, dtype=torch.bfloat16)
                t_next_A = self.frozen_teacher(t_curr_A, s_emb)
                t_next_B = self.frozen_teacher(t_curr_B, s_emb)

                tgt_A = F.normalize(t_next_A.float(), p=2, dim=-1)
                tgt_B = F.normalize(t_next_B.float(), p=2, dim=-1)
                targets_A.append(tgt_A)
                targets_B.append(tgt_B)
                t_curr_A = tgt_A.to(torch.bfloat16)
                t_curr_B = tgt_B.to(torch.bfloat16)
        return targets_A, targets_B

    def _teacher_input(self, h0: torch.Tensor, targets: list[torch.Tensor], j: int) -> torch.Tensor:
        return h0 if j == 0 else targets[j - 1]

    def _trajectory_velocity_metrics(self, h0A: torch.Tensor, targets_A: list[torch.Tensor]) -> dict:
        states = [h0A] + targets_A
        if len(states) < 2:
            return {}
        mags, angles = [], []
        v_prev = None
        for j in range(len(states) - 1):
            v = states[j + 1] - states[j]
            mags.append(torch.norm(v, p=2, dim=-1).mean().item())
            if v_prev is not None:
                cos = F.cosine_similarity(v_prev, v, dim=-1)
                angles.append(torch.acos(torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)).mean().item())
            v_prev = v
        t_mags = torch.tensor(mags)
        payload = {
            "trajectory_velocity/mean_magnitude": t_mags.mean().item(),
            "trajectory_velocity/max_magnitude": t_mags.max().item(),
            "trajectory_velocity/min_magnitude": t_mags.min().item(),
        }
        if angles:
            t_angles = torch.tensor(angles)
            payload["trajectory_alignment/mean_step_angle"] = t_angles.mean().item()
            payload["trajectory_alignment/max_step_angle"] = t_angles.max().item()
        return payload

    def train_stage(self, stage_limit: int) -> tuple[float, dict]:
        stage_metrics = {f"stage_{stage_limit}_drift/mean_drift": []}
        last_cl_loss = 0.0

        for epoch in range(self.epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for (vA, vB), _ in pbar:
                B = vA.shape[0]
                vA = vA.to(self.device)
                vB = vB.to(self.device)

                with torch.no_grad():
                    h0A = self.adapter(vA)
                    h0B = self.adapter(vB)

                targets_A, targets_B = self.precompute_trajectory(h0A, h0B, stage_limit)

                self.opt_enc.zero_grad()
                front_emb = torch.full((B, 1), float(stage_limit), device=self.device)
                front_in_A = self._teacher_input(h0A, targets_A, stage_limit)
                front_in_B = self._teacher_input(h0B, targets_B, stage_limit)
                next_A = self.encoder(front_in_A, front_emb)
                next_B = self.encoder(front_in_B, front_emb)
                cl_loss = info_nce_loss(next_A, next_B, temperature=self.temperature)
                loss = cl_loss

                mse_vals = []
                if stage_limit > 0:
                    if self.k_anchor <= 0 or self.k_anchor >= stage_limit:
                        sampled = list(range(stage_limit))
                    else:
                        sampled = random.sample(range(stage_limit), self.k_anchor)
                    scale = self.anchor_weight * (stage_limit / len(sampled))
                    for j in sampled:
                        step_emb = torch.full((B, 1), float(j), device=self.device)
                        in_A = self._teacher_input(h0A, targets_A, j)
                        in_B = self._teacher_input(h0B, targets_B, j)
                        pred_A = self.encoder(in_A, step_emb)
                        pred_B = self.encoder(in_B, step_emb)
                        mse = F.mse_loss(pred_A, targets_A[j]) + F.mse_loss(pred_B, targets_B[j])
                        loss = loss + scale * mse
                        mse_vals.append(mse.item())

                loss.backward()
                self.opt_enc.step()

                last_cl_loss = cl_loss.item()
                self._total_training_steps += 1

                log_payload = {
                    "global_step": self._total_training_steps,
                    "losses/enc_loss_tot": loss.item(),
                    "losses/enc_loss_cl": cl_loss.item(),
                }
                if mse_vals:
                    mean_mse = sum(mse_vals) / len(mse_vals)
                    log_payload["losses/anchor_mse"] = mean_mse
                    stage_metrics[f"stage_{stage_limit}_drift/mean_drift"].append(mean_mse)
                log_payload.update(self._trajectory_velocity_metrics(h0A, targets_A))
                wandb.log(log_payload)

                pbar.set_postfix({'L_cl': cl_loss.item(), 'L_tot': loss.item()})

        return last_cl_loss, stage_metrics

    def train_decoder(self) -> None:
        if self.decoder_epochs <= 0:
            print("\n=== Phase 3: Decoder training skipped (decoder.epochs = 0) ===")
            return

        print(f"\n=== Phase 3: Decoder Training ({self.decoder_epochs} epochs) ===")
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.decoder.train()

        for epoch in range(self.decoder_epochs):
            total = 0.0
            pbar = tqdm(self.train_loader, desc=f"Decoder Epoch {epoch+1}/{self.decoder_epochs}")
            for (vA, _), _ in pbar:
                B = vA.shape[0]
                vA = vA.to(self.device)
                with torch.no_grad():
                    h = self.adapter(vA)
                    states = [h]
                    for t in range(self.T):
                        step_emb = torch.full((B, 1), float(t), device=self.device)
                        h = self.encoder(h, step_emb)
                        states.append(h)

                self.opt_dec.zero_grad()
                loss = torch.zeros((), device=self.device)
                for t in range(self.T):
                    step_emb = torch.full((B, 1), float(t), device=self.device)
                    rec = self.decoder(states[t + 1], step_emb)
                    loss = loss + F.mse_loss(rec, states[t])
                loss.backward()
                self.opt_dec.step()

                total += loss.item()
                wandb.log({"decoder/loss": loss.item()})
                pbar.set_postfix({'L_dec': loss.item()})
            print(f"Decoder epoch {epoch+1} mean loss: {total/len(self.train_loader):.6f}")

        torch.save(self.decoder.state_dict(), self.ckpt_dir / f"decoder_t{self.T-1}.pt")
        print(f"Decoder saved to {self.ckpt_dir / f'decoder_t{self.T-1}.pt'}")

    def train_grim(self) -> None:
        self._total_training_steps = 0
        stage_metrics: dict = {}
        eval_results: dict = {}

        self.train_adapter()

        for t in range(self.T):
            print(f"\n=== Phase 2: Training Stage t={t} ===")
            self.encoder.train()

            last_cl_loss, metrics = self.train_stage(stage_limit=t)
            stage_metrics |= metrics
            print(f"Stage {t} completed. Frontier InfoNCE: {last_cl_loss:.4f}")

            self.save_model(stage_t=t)

            print(f"--- Frontier Evaluation for Stage t={t} ---")
            eval_results[t] = run_frontier_evaluation(
                adapter=self.adapter, encoder=self.encoder, decoder=None, test_loader=self.test_loader,
                train_loader=self.probe_train_loader, T=t+1, device=self.device, linear_probe_config=self.eval_config['linear_probe']
            )

        self.train_decoder()

        if any(len(v) for v in stage_metrics.values()):
            wandb.log({
                "custom_plot": wandb.plot.line_series(
                    xs=range(max(len(stage_metrics[k]) for k in stage_metrics)),
                    ys=list(stage_metrics.values()),
                    keys=list(stage_metrics.keys()),
                    title="Stage metrics",
                    xname="Batch index"
                )
            })

        print("Training Complete. Running final full evaluation...")
        final_results = run_evaluation(
            self.adapter, self.encoder, self.decoder, self.test_loader, self.probe_train_loader, self.T, self.device,
            linear_probe_config=self.eval_config['linear_probe']
        )

        self.log_evaluation_curves(eval_results, final_results)

        wandb.finish()
        print("Evaluation complete. Models and logs saved at:", self.ckpt_dir)

    def log_evaluation_curves(self, frontier_results: dict, final_results: dict) -> None:
        stages = sorted(frontier_results.keys())
        frontier_depths = [frontier_results[s]["frontier_depth"] for s in stages]

        wandb.log({
            "evaluation/frontier_accuracy_vs_depth": wandb.plot.line_series(
                xs=[frontier_depths],
                ys=[[frontier_results[s]["frontier_accuracy"] for s in stages]],
                keys=["frontier_accuracy"],
                title="frontier_accuracy_vs_depth", xname="depth"
            )
        })

        final_metrics = [
            ("evaluation/accuracy_vs_depth", "accuracy_vs_depth", "depths"),
            ("evaluation/local_mse_vs_depth", "local_mse_vs_depth", "mse_depths"),
            ("evaluation/elastic_accuracy_vs_depth", "elastic_accuracy_vs_depth", "depths"),
        ]
        for panel_key, series_field, depth_field in final_metrics:
            wandb.log({
                panel_key: wandb.plot.line_series(
                    xs=[final_results[depth_field]],
                    ys=[final_results[series_field]],
                    keys=["final"],
                    title=panel_key.split('/')[-1], xname="depth"
                )
            })

    def save_model(self, stage_t: int) -> None:
        torch.save(self.encoder.state_dict(), self.ckpt_dir / f"encoder_t{stage_t}.pt")

        if stage_t < self.T - 1:
            frozen_enc = copy.deepcopy(self.encoder).to(torch.bfloat16).eval()
            for p in frozen_enc.parameters():
                p.requires_grad = False
            self.frozen_teacher = frozen_enc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grim/train.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/grim/eval.yaml")
    args = parser.parse_args()
    trainer = Trainer(args.config, args.eval_config)
    trainer.train_grim()
