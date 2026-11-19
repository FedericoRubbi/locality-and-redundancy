import os
import sys
import yaml
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm.auto import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encoder import ShallowVisualCache
from src.embeddings import TargetEmbeddings
from src.diffusion import DiffusionSchedule
from src.denoiser import Denoiser, ddim_step
from src.loss import loss_infomax as contrastive_loss_infomax
from src.loss import loss_infonce as contrastive_loss_infonce
from src.loss import loss_supcon as contrastive_loss_supcon
from src.auxiliary import VariancePredictor, get_base_noise_schedule
from data.data import get_cifar100_supervised_dataloaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_dcsm_supcon.yaml')
    parser.add_argument('--encoder_ckpt', type=str, required=True, help="Path to pretrained encoder checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config {args.config} not found.")
        quit()
    else:
        config = load_config(args.config)

    kwargs = InitProcessGroupKwargs(backend="gloo")
    accelerator = Accelerator(
        mixed_precision=config['training']['mixed_precision'],
        kwargs_handlers=[kwargs]
    )

    if accelerator.is_main_process:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['experiment_name'],
            config=config
        )
        print("======== CONFIGURATION ========")
        print(yaml.dump(config, default_flow_style=False))
        print("===============================")
        wandb.config.update(config, allow_val_change=True)
        run_name = wandb.run.name if wandb.run else config['experiment_name']
        run_id = wandb.run.id if wandb.run else "local"
        ckpt_dir = os.path.join("checkpoints", f"dcsm_{run_name}_{run_id}")
        os.makedirs(ckpt_dir, exist_ok=True)
        import shutil
        shutil.copy(args.config, os.path.join(ckpt_dir, "config.yaml"))

    encoder_cfg = config['encoder']
    denoiser_cfg = config['denoiser']
    variant = config['training']['variant']

    encoder = ShallowVisualCache(
        img_size=encoder_cfg['image_size'],
        patch_size=encoder_cfg['patch_size'],
        embed_dim=encoder_cfg['embed_dim'],
        depth=encoder_cfg['depth'],
        num_heads=encoder_cfg['num_heads'],
        mlp_ratio=encoder_cfg['mlp_ratio']
    )
    
    if accelerator.is_main_process: print(f"Loading encoder from {args.encoder_ckpt}...")
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location='cpu', weights_only=True))
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    denoiser = Denoiser(
        embed_dim=denoiser_cfg['embed_dim'],
        num_blocks=denoiser_cfg['num_blocks']
    )

    params_to_optimize = list(denoiser.parameters())
    temperature = config['training']['temperature']
    lambda_info = config['training']['lambda_info']

    if variant == 'supcon':
        label_embeds = TargetEmbeddings(num_classes=100, embed_dim=denoiser_cfg['embed_dim'])
    elif variant == 'infonce':
        variance_predictor = VariancePredictor(embed_dim=denoiser_cfg['embed_dim'])
        params_to_optimize += list(variance_predictor.parameters())
        base_noise_schedule = get_base_noise_schedule(
            embed_dim=denoiser_cfg['embed_dim'],
            rho_max=config['info_nce']['rho_max']
        )

    schedule = DiffusionSchedule(num_timesteps=config['diffusion']['num_timesteps'])

    train_loader, _ = get_cifar100_supervised_dataloaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    optimizer = optim.AdamW(params_to_optimize, lr=float(config['training']['learning_rate']))

    if variant == 'supcon':
        denoiser, label_embeds, optimizer, train_loader = accelerator.prepare(
            denoiser, label_embeds, optimizer, train_loader
        )
    elif variant == 'infonce':
        denoiser, variance_predictor, optimizer, train_loader = accelerator.prepare(
            denoiser, variance_predictor, optimizer, train_loader
        )
    else:
        denoiser, optimizer, train_loader = accelerator.prepare(
            denoiser, optimizer, train_loader
        )

    schedule = schedule.to(accelerator.device)
    encoder = encoder.to(accelerator.device)
    if variant == 'infonce':
        base_noise_schedule = base_noise_schedule.to(accelerator.device)

    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        denoiser.train()
        if variant == 'infonce':
            variance_predictor.train()

        total_loss_accum = 0
        total_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_local_main_process)

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            with torch.no_grad():
                c = encoder(inputs)

            batch_size = inputs.size(0)
            t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=accelerator.device).long()
            
            with accelerator.autocast():
                if variant == 'supcon':
                    y_0 = label_embeds(labels)
                    y_t, noise = schedule.add_noise(y_0, t)
                    y_0_hat = denoiser(y_t, t, c)
                    loss = contrastive_loss_supcon(y_0_hat, labels, temperature=temperature)
                elif variant == 'infonce':
                    y_0 = torch.randn(batch_size, denoiser_cfg['embed_dim'], device=accelerator.device)
                    y_0 = F.normalize(y_0, p=2, dim=-1)
                    y_t, noise = schedule.add_noise(y_0, t)
                    
                    u_t = denoiser(y_t, t, c)
                    alpha_t = variance_predictor(y_t, t, c)
                    
                    sigma_t_sq = base_noise_schedule.unsqueeze(0) + alpha_t**2
                    
                    eps = torch.randn_like(u_t)
                    u_tilde_t = u_t + torch.sqrt(sigma_t_sq) * eps
                    
                    contrastive_term = contrastive_loss_infonce(u_tilde_t, temperature=temperature)
                    loss_info = -torch.log(sigma_t_sq).sum(dim=-1).mean()
                    
                    loss = contrastive_term + lambda_info * loss_info
                else:
                    y_0 = torch.randn(batch_size, denoiser_cfg['embed_dim'], device=accelerator.device)
                    y_0 = F.normalize(y_0, p=2, dim=-1)
                    y_t, noise = schedule.add_noise(y_0, t)
                    
                    y_0_hat = denoiser(y_t, t, c)
                    
                    alphas_bar_t = schedule.alphas_cumprod[t].view(-1, 1)
                    alphas_bar_t_minus_1 = torch.where(
                        t.unsqueeze(-1) > 0,
                        schedule.alphas_cumprod[(t - 1).clamp_min(0)].view(-1, 1),
                        torch.ones_like(alphas_bar_t),
                    )
                    
                    y_t_minus_1 = ddim_step(y_t, y_0_hat, alphas_bar_t, alphas_bar_t_minus_1)
                    loss = contrastive_loss_infomax(y_t, y_t_minus_1, temperature=temperature)

            accelerator.backward(loss)
            optimizer.step()
            
            total_loss_accum += loss.item()
            total_batches += 1
            
            if accelerator.is_local_main_process:
                log_dict = {
                    f"train/batch_loss_{variant}": loss.item(),
                    "epoch": epoch + 1,
                }
                
                postfix_dict = {
                    'loss': f"{loss.item():.4f}",
                }

                wandb.log(log_dict)
                progress_bar.set_postfix(postfix_dict)
        
        epoch_metrics = torch.tensor([
            total_loss_accum,
            float(total_batches),
        ], device=accelerator.device)
        reduced_metrics = accelerator.reduce(epoch_metrics, reduction="sum")

        if accelerator.is_main_process:
            global_batches = max(reduced_metrics[1].item(), 1.0)
            avg_loss = reduced_metrics[0].item() / global_batches
            
            wandb.log({
                f"train/loss_total_{variant}": avg_loss,
                "epoch": epoch + 1
            })
            
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                unwrapped = accelerator.unwrap_model(denoiser)
                torch.save(unwrapped.state_dict(), os.path.join(ckpt_dir, f"denoiser_epoch_{epoch+1}.pt"))
                
                unwrapped_encoder = accelerator.unwrap_model(encoder)
                torch.save(unwrapped_encoder.state_dict(), os.path.join(ckpt_dir, f"encoder_epoch_{epoch+1}.pt"))

                if variant == 'supcon':
                    unwrapped_embeds = accelerator.unwrap_model(label_embeds)
                    torch.save(unwrapped_embeds.state_dict(), os.path.join(ckpt_dir, f"embeds_epoch_{epoch+1}.pt"))
                elif variant == 'infonce':
                    unwrapped_variance_predictor = accelerator.unwrap_model(variance_predictor)
                    torch.save(unwrapped_variance_predictor.state_dict(), os.path.join(ckpt_dir, f"variance_predictor_epoch_{epoch+1}.pt"))

    if accelerator.is_main_process:
        wandb.finish()

if __name__ == '__main__':
    main()
