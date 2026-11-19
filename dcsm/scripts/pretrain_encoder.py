import os
import yaml
import torch
import torch.optim as optim
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm.auto import tqdm
import wandb
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encoder import ShallowVisualCache
from data.data import get_cifar100_pretrain_dataloaders
from src.loss import InfoNCELoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('configs/pretrain_cifar100.yaml')
    
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
        
        run_name = wandb.run.name if wandb.run else config['experiment_name']
        run_id = wandb.run.id if wandb.run else "local"
        ckpt_dir = os.path.join("checkpoints", f"pretrain_{run_name}_{run_id}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
    model = ShallowVisualCache(
        img_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        proj_dim=config['model']['proj_dim']
    )
    
    train_loader = get_cifar100_pretrain_dataloaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay']
    )
    criterion = InfoNCELoss(temperature=config['training']['temperature'])
    
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    
    print(f"[Rank {accelerator.process_index}] Prepared model!")

    
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        for batch_idx, ((view1, view2), _) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            with accelerator.autocast():
                z1 = model(view1)
                z2 = model(view2)
                loss = criterion(z1, z2)
                
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        
        if accelerator.is_main_process:
            wandb.log({"train/loss": avg_loss, "epoch": epoch})
            
            if (epoch + 1) % 10 == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, f"encoder_epoch_{epoch+1}.pt"))
                
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
