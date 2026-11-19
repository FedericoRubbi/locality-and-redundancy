import os
import sys
import yaml
import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm.auto import tqdm
import wandb
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encoder import ShallowVisualCache
from src.embeddings import TargetEmbeddings
from src.diffusion import DiffusionSchedule
from src.denoiser import Denoiser, ddim_step
from data.data import get_cifar100_supervised_dataloaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True, help="Path to experiment folder containing config.yaml and checkpoints")
    parser.add_argument('--tau', type=float, default=None, help="Temperature for Softmax Prediction")
    args = parser.parse_args()

    config_path = os.path.join(args.experiment_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"Config config.yaml not found in {args.experiment_dir}.")
        sys.exit(1)
    
    config = load_config(config_path)
    variant = config['training']['variant']
    tau = args.tau if args.tau is not None else config['training']['temperature']

    import glob
    def get_latest_ckpt(prefix):
        ckpts = glob.glob(os.path.join(args.experiment_dir, f"{prefix}_epoch_*.pt"))
        if not ckpts: return None
        ckpts.sort(key=lambda x: int(os.path.basename(x).split('_epoch_')[1].split('.pt')[0]))
        return ckpts[-1]
    
    encoder_ckpt = get_latest_ckpt("encoder")
    denoiser_ckpt = get_latest_ckpt("denoiser")
    
    if not encoder_ckpt or not denoiser_ckpt:
        print(f"Could not find encoder or denoiser checkpoints in {args.experiment_dir}.")
        sys.exit(1)
        
    embeds_ckpt = None
    if variant == "supcon":
        embeds_ckpt = get_latest_ckpt("embeds")
        if not embeds_ckpt:
            print(f"Could not find embeds checkpoints in {args.experiment_dir}.")
            sys.exit(1)

    kwargs = InitProcessGroupKwargs(backend="gloo")
    accelerator = Accelerator(mixed_precision=config['training']['mixed_precision'], kwargs_handlers=[kwargs])

    if accelerator.is_main_process:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['experiment_name'] + "_eval",
            config=config,
            tags=["evaluation"]
        )
        print("======== CONFIGURATION ========")
        print(yaml.dump(config, default_flow_style=False))
        print("===============================")
        wandb.config.update(config, allow_val_change=True)

    encoder_cfg = config['encoder']
    denoiser_cfg = config['denoiser']

    encoder = ShallowVisualCache(
        img_size=encoder_cfg['image_size'],
        patch_size=encoder_cfg['patch_size'],
        embed_dim=encoder_cfg['embed_dim'],
        depth=encoder_cfg['depth'],
        num_heads=encoder_cfg['num_heads'],
        mlp_ratio=encoder_cfg['mlp_ratio']
    )
    
    if accelerator.is_main_process: print(f"Loading encoder from {encoder_ckpt}...")
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location='cpu', weights_only=True))
    encoder.eval()

    denoiser = Denoiser(
        embed_dim=denoiser_cfg['embed_dim'],
        num_blocks=denoiser_cfg['num_blocks']
    )
    if accelerator.is_main_process: print(f"Loading denoiser from {denoiser_ckpt}...")
    denoiser.load_state_dict(torch.load(denoiser_ckpt, map_location='cpu', weights_only=True))
    denoiser.eval()

    if variant == 'supcon':
        label_embeds = TargetEmbeddings(num_classes=100, embed_dim=denoiser_cfg['embed_dim'])
        if embeds_ckpt:
            if accelerator.is_main_process: print(f"Loading label embeddings from {embeds_ckpt}...")
            label_embeds.load_state_dict(torch.load(embeds_ckpt, map_location='cpu', weights_only=True))
    
    schedule = DiffusionSchedule(num_timesteps=config['diffusion']['num_timesteps'])

    _, test_loader = get_cifar100_supervised_dataloaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    denoiser, test_loader = accelerator.prepare(denoiser, test_loader)
    schedule = schedule.to(accelerator.device)
    encoder = encoder.to(accelerator.device)
    if variant == 'supcon':
        label_embeds = label_embeds.to(accelerator.device)
        y_all_classes = label_embeds(torch.arange(100, device=accelerator.device))

    for p in encoder.parameters(): p.requires_grad = False
    for p in denoiser.parameters(): p.requires_grad = False

    num_steps = schedule.num_timesteps
    
    step_metrics = {
        "drift": torch.zeros(num_steps, device=accelerator.device),
        "alignment_tgt": torch.zeros(num_steps, device=accelerator.device),
        "accuracy": torch.zeros(num_steps, device=accelerator.device),
        "entropy": torch.zeros(num_steps, device=accelerator.device),
        "counts": torch.zeros(num_steps, device=accelerator.device),
        "y_t_l2": torch.zeros(num_steps, device=accelerator.device),
        "y_0_hat_l2": torch.zeros(num_steps, device=accelerator.device),
    }

    if accelerator.is_main_process:
        print("Starting evaluation across the full reverse chain...")
        
    progress_bar = tqdm(test_loader, disable=not accelerator.is_local_main_process)
    
    for inputs, labels in progress_bar:
        batch_size = inputs.size(0)
        c = encoder(inputs)
        
        if variant == 'supcon':
            y_true = label_embeds(labels)

        y_t = torch.randn((batch_size, denoiser_cfg['embed_dim']), device=accelerator.device)
        y_t = F.normalize(y_t, p=2, dim=-1)
        
        for step in reversed(range(num_steps)):
            t_tensor = torch.full((batch_size,), step, device=accelerator.device, dtype=torch.long)
            
            with accelerator.autocast():
                y_0_hat = denoiser(y_t, t_tensor, c)
            
            y_t_l2 = y_t.float().norm(dim=-1)
            y_0_hat_l2 = y_0_hat.float().norm(dim=-1)

            if variant == 'supcon':
                alignment = F.cosine_similarity(y_0_hat, y_true, dim=-1)
                y_hat_norm = F.normalize(y_0_hat, dim=-1)
                y_all_norm = F.normalize(y_all_classes, dim=-1)
                sim_matrix = torch.matmul(y_hat_norm, y_all_norm.T)
                
                preds = torch.argmax(sim_matrix, dim=-1)
                correct = (preds == labels).float()
                
                probs = F.softmax(sim_matrix / tau, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                
                step_metrics["alignment_tgt"][step] += alignment.sum()
                step_metrics["accuracy"][step] += correct.sum()
                step_metrics["entropy"][step] += entropy.sum()

            step_metrics["counts"][step] += batch_size
            step_metrics["y_t_l2"][step] += y_t_l2.sum()
            step_metrics["y_0_hat_l2"][step] += y_0_hat_l2.sum()

            if step > 0:
                alphas_bar_t = schedule.alphas_cumprod[step].view(-1, 1)
                alphas_bar_t_minus_1 = schedule.alphas_cumprod[step-1].view(-1, 1)
                
                y_t_next = ddim_step(y_t, y_0_hat, alphas_bar_t, alphas_bar_t_minus_1)
                
                drift = F.cosine_similarity(y_t, y_t_next, dim=-1)
                step_metrics["drift"][step] += drift.sum()
                
                y_t = y_t_next
            else:
                step_metrics["drift"][step] += torch.zeros(batch_size, device=accelerator.device).sum()
                
    for key in step_metrics.keys():
        step_metrics[key] = accelerator.reduce(step_metrics[key], reduction="sum")
        
    if accelerator.is_main_process:
        print("Evaluation complete! Logging to Weights & Biases...")
        
        counts = step_metrics["counts"].cpu().numpy()
        counts = np.maximum(counts, 1.0)
        
        avg_drift = step_metrics["drift"].cpu().numpy() / counts
        avg_y_t_l2 = step_metrics["y_t_l2"].cpu().numpy() / counts
        avg_y_0_hat_l2 = step_metrics["y_0_hat_l2"].cpu().numpy() / counts
        
        log_data = {
            "eval/step": list(range(num_steps)),
            "eval/trajectory_cosine_drift": avg_drift.tolist(),
            "eval/y_t_l2": avg_y_t_l2.tolist(),
            "eval/y_0_hat_l2": avg_y_0_hat_l2.tolist(),
        }

        if variant == 'supcon':
            avg_alignment = step_metrics["alignment_tgt"].cpu().numpy() / counts
            avg_accuracy = step_metrics["accuracy"].cpu().numpy() / counts
            avg_entropy = step_metrics["entropy"].cpu().numpy() / counts
            log_data["eval/accuracy"] = avg_accuracy.tolist()
            log_data["eval/alignment"] = avg_alignment.tolist()
            log_data["eval/entropy"] = avg_entropy.tolist()
            print(f"Final Step (t=0) Accuracy: {avg_accuracy[0]*100:.2f}%")

        for t in range(num_steps):
            wandb.log({k: v[t] for k, v in log_data.items()})

        wandb.finish()

if __name__ == '__main__':
    main()
