import os
import json
import torch
import torch.nn as nn
from models.greedy_vit import GreedyViT
from utils.targets import generate_dirichlet_targets
from data.dataloaders import get_cifar100_dataloaders, get_imagenet100_dataloaders
from utils.metrics import KDE_Renyi_Entropy
from utils.helpers import get_incremental_save_dir
from vit_pytorch import ViT

def init_orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing Greedy Block-wise Training...")
    
    batch_size = 256
    epochs = 50
    num_classes = 100
    L = 8
    c_l = 1.0
    dataset_name = 'ImageNet100'
    use_label_noise = False
    
    save_dir = get_incremental_save_dir('plots', dataset_name)
    print(f"Simulation artifacts will be saved to: {save_dir}")
    
    schedule_path = 'configs/schedule.json'
    if not os.path.exists(schedule_path):
        print("Schedule not found. Please run scripts/compute_schedule.py first.")
        return
    with open(schedule_path, 'r') as f:
        schedule_data = json.load(f)
    s_values = schedule_data['s_values']

    if dataset_name == 'CIFAR100':
        trainloader, testloader, _ = get_cifar100_dataloaders(batch_size=batch_size, data_dir='./data')
    elif dataset_name == 'ImageNet100':
        trainloader, testloader, _ = get_imagenet100_dataloaders(batch_size=batch_size, data_dir='./data/imagenet100')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    vit = ViT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=L,
        heads=3,
        mlp_dim=768
    )
    model = GreedyViT(vit)
    model.apply(init_orthogonal)
    model.to(device)

    optimizers = []
    
    for i in range(L):
        if i == 0:
            params = list(model.layers[0].parameters()) + \
                     list(model.to_patch_embedding.parameters()) + \
                     [model.cls_token, model.pos_embedding]
        else:
            params = list(model.layers[i].parameters())
            
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
        optimizers.append(opt)

    print(f"Initialized {len(optimizers)} isolated block optimizers.")

    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
        'layer_train_loss': [], 'layer_train_acc': [],
        'layer_test_loss': [], 'layer_test_acc': [],
        'mag_mean': [], 'ent_mean': [], 'ortho': [], 'layer_rec_loss': [],
        'alphas': [s_values]
    }
    
    renyi_estimator = KDE_Renyi_Entropy(sigma=1.0)
    

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = [0.0] * L
        train_correct = [0] * L
        total_train = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            z_list, logits_list = model(inputs)
            
            for opt in optimizers:
                opt.zero_grad()
                
            for l in range(L):
                if use_label_noise:
                    y_tilde = generate_dirichlet_targets(targets, s_values[l], c_l, num_classes)
                    log_probs = torch.nn.functional.log_softmax(logits_list[l], dim=-1)
                    loss_l = -torch.mean(torch.sum(y_tilde * log_probs, dim=-1))
                    
                    probs = torch.nn.functional.softmax(logits_list[l], dim=-1)
                else:
                    loss_l = torch.nn.functional.cross_entropy(logits_list[l], targets)
                    probs = torch.nn.functional.softmax(logits_list[l], dim=-1)
                
                loss_l.backward(retain_graph=(l < L-1))
                optimizers[l].step()
                
                train_loss[l] += loss_l.item()
                preds = probs.argmax(dim=1)
                train_correct[l] += preds.eq(targets).sum().item()
                
            total_train += targets.size(0)
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(trainloader)} | Block {L-1} Acc: {100.*train_correct[L-1]/total_train:.2f}%")

        model.eval()
        test_loss = [0.0] * L
        test_correct = [0] * L
        block_mags = [0.0] * L
        block_ents = [0.0] * L
        total_test = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                z_list, logits_list = model(inputs)
                
                for l in range(L):
                    loss_l = torch.nn.functional.cross_entropy(logits_list[l], targets)
                    test_loss[l] += loss_l.item()
                    preds = logits_list[l].argmax(dim=1)
                    test_correct[l] += preds.eq(targets).sum().item()
                    
                    z_flat = z_list[l].view(z_list[l].size(0), -1)
                    block_mags[l] += torch.norm(z_flat, dim=-1).mean().item()
                    block_ents[l] += renyi_estimator(z_flat[:100]).item()
                    
                total_test += targets.size(0)

        history['layer_train_loss'].append([train_loss[l] / len(trainloader) for l in range(L)])
        history['layer_train_acc'].append([100. * train_correct[l] / total_train for l in range(L)])
        history['layer_test_loss'].append([test_loss[l] / len(testloader) for l in range(L)])
        history['layer_test_acc'].append([100. * test_correct[l] / total_test for l in range(L)])
        
        history['train_loss'].append(history['layer_train_loss'][-1][-1])
        history['train_acc'].append(history['layer_train_acc'][-1][-1])
        history['test_loss'].append(history['layer_test_loss'][-1][-1])
        history['test_acc'].append(history['layer_test_acc'][-1][-1])

        history['mag_mean'].append([block_mags[l] / len(testloader) for l in range(L)])
        history['ent_mean'].append([block_ents[l] / len(testloader) for l in range(L)])
        history['ortho'].append([0.0] * L)
        history['layer_rec_loss'].append([0.0] * L)
        
        print(f"--> Epoch {epoch} Final | Test Acc (L-1): {history['test_acc'][-1]:.2f}% | Ent (L-1): {history['ent_mean'][-1][-1]:.2f}")
        

    history_path = os.path.join(save_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    print("\nTraining Complete! Generating post-run plots...")
    from utils.plots import plot_global_metrics_evolution, plot_standard_metrics, plot_alpha_progression, plot_all_metrics_evolution, plot_layerwise_test_train_metrics
    
    plot_layerwise_test_train_metrics(history, epochs, L, dataset_name, save_dir=save_dir)
    plot_global_metrics_evolution(history, epochs, dataset_name, save_dir=save_dir)
    plot_standard_metrics(history, L, dataset_name, epochs, save_dir=save_dir)
    plot_alpha_progression(history, L, dataset_name, save_dir=save_dir)
    plot_all_metrics_evolution(history, epochs, L, dataset_name, save_dir=save_dir)
    print(f"Plots saved in `{save_dir}/` directory.")
    
    checkpoint_path = os.path.join(save_dir, f'model_greedy_vit_{dataset_name.lower()}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint successfully written to: {checkpoint_path}")

if __name__ == "__main__":
    main()
