import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.info_vit_v1 import InfoViTV1
from utils.targets import generate_dirichlet_targets
from data.dataloaders import get_cifar100_dataloaders, get_imagenet100_dataloaders
from utils.metrics import KDE_Renyi_Entropy
from utils.helpers import get_incremental_save_dir, setup_logging
from vit_pytorch import ViT

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def setup_ddp():
    dist.init_process_group(backend="gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    is_main_process = (local_rank == 0)
    if is_main_process:
        print("Initializing Distributed InfoViTV1 Training...")
    
    batch_size = 1024
    batch_size_per_gpu = batch_size
    epochs = 50
    num_classes = 100
    L = 8
    c_l = 1.0
    dataset_name = 'ImageNet100'
    use_label_noise = False
    lambda_rec = 1.0
    
    image_size = 224
    patch_size = 16
    dim = 192
    heads = 3
    mlp_dim = 768

    if is_main_process:
        save_dir = get_incremental_save_dir('plots', dataset_name + '_infovit_v1')
        setup_logging(save_dir)
        print(f"Simulation artifacts will be saved to: {save_dir}")
        config = {
            'batch_size': batch_size,
            'epochs': epochs,
            'num_classes': num_classes,
            'L': L,
            'c_l': c_l,
            'dataset_name': dataset_name,
            'use_label_noise': use_label_noise,
            'lambda_rec': lambda_rec,
            'image_size': image_size,
            'patch_size': patch_size,
            'dim': dim,
            'heads': heads,
            'mlp_dim': mlp_dim
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    else:
        save_dir = None
    
    schedule_path = 'configs/schedule.json'
    if not os.path.exists(schedule_path):
        print("Schedule not found. Please run scripts/compute_schedule.py first.")
        return
    with open(schedule_path, 'r') as f:
        schedule_data = json.load(f)
    s_values = schedule_data['s_values']

    if dataset_name == 'CIFAR100':
        trainloader, testloader, train_sampler = get_cifar100_dataloaders(batch_size=batch_size, data_dir='./data', distributed=True)
    elif dataset_name == 'ImageNet100':
        trainloader, testloader, train_sampler = get_imagenet100_dataloaders(
            batch_size=batch_size_per_gpu, 
            data_dir='./data/imagenet100',
            distributed=True
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    vit = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=L,
        heads=heads,
        mlp_dim=mlp_dim
    )

    model = InfoViTV1(vit)
    model = model.to(device)

    class DDPWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module = model
            
        def forward(self, img):
            m = self.module
            batch = img.shape[0]
            x = m.to_patch_embedding(img)
            cls_tokens = m.cls_token.expand(batch, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            seq = x.shape[1]
            x = x + m.pos_embedding[:seq]
            x = m.dropout(x)

            block_outputs = []
            block_logits = []
            block_rec_losses = []
            
            for l, block in enumerate(m.layers):
                x, logits, rec_loss = block(x)
                block_outputs.append(x)
                block_logits.append(logits)
                block_rec_losses.append(rec_loss)
                
            return block_outputs, block_logits, block_rec_losses

    model = DDP(DDPWrapper(model), device_ids=[local_rank], find_unused_parameters=True)

    optimizers = []
    base_model = model.module.module
    for i in range(L):
        if i == 0:
            params = list(base_model.layers[0].parameters()) + \
                     list(base_model.to_patch_embedding.parameters()) + \
                     [base_model.cls_token, base_model.pos_embedding]
        else:
            params = list(base_model.layers[i].parameters())
            
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
        optimizers.append(opt)

    print(f"Initialized {len(optimizers)} isolated block optimizers.")

    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
        'layer_train_loss': [], 'layer_train_acc': [],
        'layer_test_loss': [], 'layer_test_acc': [],
        'layer_train_rec_loss': [], 'layer_test_rec_loss': [],
        'train_rec_loss': [], 'test_rec_loss': [],
        'mag_mean': [], 'ent_mean': [], 'ortho': [], 'layer_rec_loss': [],
        'alphas': [s_values]
    }
    
    renyi_estimator = KDE_Renyi_Entropy(sigma=1.0)

    try:
        for epoch in range(1, epochs + 1):
            train_sampler.set_epoch(epoch) 
            
            model.train()
            train_loss = [0.0] * L
            train_rec_loss = [0.0] * L
            train_correct = [0] * L 
            total_train = 0
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                for opt in optimizers:
                    opt.zero_grad()
                    
                z_list, logits_list, rec_losses = model(inputs)
                
                block_losses = []
                for l in range(L):
                    if use_label_noise:
                        y_tilde = generate_dirichlet_targets(targets, s_values[l], c_l, num_classes)
                        log_probs = torch.nn.functional.log_softmax(logits_list[l], dim=-1)
                        ce_loss = -torch.mean(torch.sum(y_tilde * log_probs, dim=-1))
                    else:
                        ce_loss = torch.nn.functional.cross_entropy(logits_list[l], targets)
                    
                    loss_l = ce_loss + lambda_rec * rec_losses[l]
                    
                    block_losses.append(loss_l)
                    train_loss[l] += loss_l.item()
                    train_rec_loss[l] += rec_losses[l].item()
    
                    preds = logits_list[l].argmax(dim=1)
                    train_correct[l] += preds.eq(targets).sum().item()
    
                total_train += targets.size(0)
    
                total_loss = sum(block_losses)
                total_loss.backward()
                
                for opt in optimizers:
                    opt.step()
                    
                if is_main_process and batch_idx % 50 == 0:
                    current_acc = 100. * train_correct[-1] / total_train if total_train > 0 else 0
                    print(f"Epoch {epoch} | Batch {batch_idx}/{len(trainloader)} | Block {L-1} Acc: {current_acc:.2f}%")
                          
            model.eval()
            test_loss = [0.0] * L
            test_rec_loss_epoch = [0.0] * L
            test_correct = [0] * L
            block_mags = [0.0] * L
            block_ents = [0.0] * L
            total_test = 0
            
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    z_list, logits_list, test_rec_losses = model(inputs)
                    
                    for l in range(L):
                        loss_l = torch.nn.functional.cross_entropy(logits_list[l], targets)
                        test_loss[l] += loss_l.item()
                        test_rec_loss_epoch[l] += test_rec_losses[l].item()
                        preds = logits_list[l].argmax(dim=1)
                        test_correct[l] += preds.eq(targets).sum().item()
                        
                        z_flat = z_list[l].view(z_list[l].size(0), -1)
                        block_mags[l] += torch.norm(z_flat, dim=-1).mean().item()
                        block_ents[l] += renyi_estimator(z_flat[:100]).item()
                        
                    total_test += targets.size(0)
    
            metrics_tensor = torch.tensor([
                *train_loss, *train_correct, total_train,
                *test_loss, *test_correct, total_test,
                *block_mags, *block_ents, *train_rec_loss, *test_rec_loss_epoch
            ], dtype=torch.float32, device=device)
            
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            
            world_size = 2 
            
            synced_train_loss = (metrics_tensor[0:L] / world_size).cpu().tolist()
            synced_train_correct = metrics_tensor[L:2*L].cpu().tolist()
            synced_total_train = metrics_tensor[2*L].item()
            
            synced_test_loss = (metrics_tensor[2*L+1 : 3*L+1] / world_size).cpu().tolist()
            synced_test_correct = metrics_tensor[3*L+1 : 4*L+1].cpu().tolist()
            synced_total_test = metrics_tensor[4*L+1].item()
            
            synced_mags = (metrics_tensor[4*L+2 : 5*L+2] / world_size).cpu().tolist()
            synced_ents = (metrics_tensor[5*L+2 : 6*L+2] / world_size).cpu().tolist()
            synced_train_rec_loss = (metrics_tensor[6*L+2 : 7*L+2] / world_size).cpu().tolist()
            synced_test_rec_loss = (metrics_tensor[7*L+2 : 8*L+2] / world_size).cpu().tolist()
    
            history['layer_train_loss'].append([synced_train_loss[l] / len(trainloader) for l in range(L)])
            history['layer_train_acc'].append([100. * synced_train_correct[l] / synced_total_train for l in range(L)])
            history['layer_test_loss'].append([synced_test_loss[l] / len(testloader) for l in range(L)])
            history['layer_test_acc'].append([100. * synced_test_correct[l] / synced_total_test for l in range(L)])
            history['layer_rec_loss'].append([synced_train_rec_loss[l] / len(trainloader) for l in range(L)])
            
            history['layer_train_rec_loss'].append([synced_train_rec_loss[l] / len(trainloader) for l in range(L)])
            history['layer_test_rec_loss'].append([synced_test_rec_loss[l] / len(testloader) for l in range(L)])
            
            history['train_loss'].append(history['layer_train_loss'][-1][-1])
            history['train_acc'].append(history['layer_train_acc'][-1][-1])
            history['test_loss'].append(history['layer_test_loss'][-1][-1])
            history['test_acc'].append(history['layer_test_acc'][-1][-1])
            history['train_rec_loss'].append(history['layer_train_rec_loss'][-1][-1])
            history['test_rec_loss'].append(history['layer_test_rec_loss'][-1][-1])
    
            history['mag_mean'].append([synced_mags[l] / len(testloader) for l in range(L)])
            history['ent_mean'].append([synced_ents[l] / len(testloader) for l in range(L)])
            history['ortho'].append(base_model.get_orthogonality())

            if is_main_process:
                print(f"--> Epoch {epoch} Final | Train Acc (L-1): {history['train_acc'][-1]:.2f}% | Test Acc (L-1): {history['test_acc'][-1]:.2f}% | Ent (L-1): {history['ent_mean'][-1][-1]:.2f}")
    except KeyboardInterrupt:
        if is_main_process:
            print("\nTraining interrupted by user. Generating plots and saving current state...")
            epochs = len(history['train_acc'])

    if is_main_process:
        history_path = os.path.join(save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print("\nTraining Complete! Generating post-run plots...")
        from utils.plots import plot_global_metrics_evolution, plot_standard_metrics, plot_alpha_progression, plot_all_metrics_evolution, plot_layerwise_test_train_metrics, plot_orthogonality_evolution
        
        plot_layerwise_test_train_metrics(history, epochs, L, dataset_name, save_dir=save_dir)
        plot_global_metrics_evolution(history, epochs, dataset_name, save_dir=save_dir)
        plot_standard_metrics(history, L, dataset_name, epochs, save_dir=save_dir)
        plot_alpha_progression(history, L, dataset_name, save_dir=save_dir)
        plot_all_metrics_evolution(history, epochs, L, dataset_name, save_dir=save_dir)
        plot_orthogonality_evolution(history, epochs, L, dataset_name, save_dir=save_dir)
        print(f"Plots saved in `{save_dir}/` directory.")
        
        checkpoint_path = os.path.join(save_dir, f'model_infovit_v1_{dataset_name.lower()}.pth')
        torch.save(base_model.state_dict(), checkpoint_path)
        print(f"Checkpoint successfully written to: {checkpoint_path}")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()