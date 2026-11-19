import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm.auto import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encoder import ShallowVisualCache
from data.data import get_cifar100_supervised_dataloaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class LinearProbeModel(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes=100):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder.forward_features(x)
            pooled_features = features.mean(dim=1)
        out = self.classifier(pooled_features)
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pretrain_cifar100.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/encoder_epoch_100.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

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
            name=f"{config['experiment_name']}_probe",
            config={**config, "probe_lr": args.lr, "probe_epochs": args.epochs, "probe_batch_size": args.batch_size}
        )
        run_name = wandb.run.name if wandb.run else f"{config['experiment_name']}_probe"
        run_id = wandb.run.id if wandb.run else "local"
        ckpt_dir = os.path.join("checkpoints", f"probe_{run_name}_{run_id}")
        os.makedirs(ckpt_dir, exist_ok=True)

    encoder = ShallowVisualCache(
        img_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        proj_dim=config['model']['proj_dim']
    )

    if accelerator.is_main_process:
        print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        encoder.load_state_dict(state_dict)
    except FileNotFoundError:
        if accelerator.is_main_process:
            print(f"Checkpoint file {args.checkpoint} not found. Please train encoder first.")
        return

    model = LinearProbeModel(encoder, embed_dim=config['model']['embed_dim'], num_classes=100)
    
    train_loader, test_loader = get_cifar100_supervised_dataloaders(
        batch_size=args.batch_size, num_workers=config['data']['num_workers']
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=not accelerator.is_local_main_process, leave=False)
        for inputs, targets in train_bar:
            optimizer.zero_grad()
            
            with accelerator.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            accelerator.backward(loss)
            optimizer.step()
            
            all_outputs = accelerator.gather(outputs)
            all_targets = accelerator.gather(targets)
            
            total_loss += loss.item() * all_inputs_size_hack(all_targets)
            
            _, predicted = all_outputs.max(1)
            total += all_targets.size(0)
            correct += predicted.eq(all_targets).sum().item()
            
            if accelerator.is_local_main_process:
                train_bar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
        train_acc = 100. * correct / total

        
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Test]", disable=not accelerator.is_local_main_process, leave=False)
        with torch.no_grad():
            for inputs, targets in test_bar:
                with accelerator.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                all_outputs = accelerator.gather(outputs)
                all_targets = accelerator.gather(targets)
                all_loss = accelerator.gather(loss.repeat(targets.size(0)))

                test_loss += all_loss.sum().item()
                _, predicted = all_outputs.max(1)
                test_total += all_targets.size(0)
                test_correct += predicted.eq(all_targets).sum().item()
                
                if accelerator.is_local_main_process:
                    test_bar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*test_correct/test_total:.2f}%"})
                
        test_loss = test_loss / test_total
        test_acc = 100. * test_correct / test_total
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} Summary: Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            wandb.log({
                "train/acc": train_acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
                "epoch": epoch + 1
            })
            
            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'classifier': unwrapped_model.classifier.state_dict(),
                    'epoch': epoch + 1,
                    'test_acc': test_acc
                }, os.path.join(ckpt_dir, f"probe_epoch_{epoch+1}.pt"))

    if accelerator.is_main_process:
        wandb.finish()

def all_inputs_size_hack(targets):
    return targets.size(0)

if __name__ == '__main__':
    main()
