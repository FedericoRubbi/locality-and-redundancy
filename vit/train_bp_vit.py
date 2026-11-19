import os
import json
import torch
import torch.nn as nn
from data.dataloaders import get_cifar100_dataloaders, get_imagenet100_dataloaders
from utils.helpers import get_incremental_save_dir
from vit_pytorch import ViT

def init_orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing Standard Backpropagation Baseline (ViT)...")
    
    batch_size = 256
    epochs = 50
    num_classes = 100
    L = 8
    dataset_name = 'ImageNet100'
    
    save_dir = get_incremental_save_dir('plots', f"{dataset_name}_baseline")
    print(f"Simulation artifacts will be saved to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    if dataset_name == 'CIFAR100':
        trainloader, testloader, _ = get_cifar100_dataloaders(batch_size=batch_size, data_dir='./data')
    elif dataset_name == 'ImageNet100':
        trainloader, testloader, _ = get_imagenet100_dataloaders(batch_size=batch_size, data_dir='./data/imagenet100')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=L,
        heads=3,
        mlp_dim=768
    )
    model.apply(init_orthogonal)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += preds.eq(targets).sum().item()
            total_train += targets.size(0)
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(trainloader)} | Acc: {100.*train_correct/total_train:.2f}%")

        model.eval()
        test_loss = 0.0
        test_correct = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                
                loss = criterion(logits, targets)
                test_loss += loss.item()
                preds = logits.argmax(dim=1)
                test_correct += preds.eq(targets).sum().item()
                total_test += targets.size(0)

        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(100. * train_correct / total_train)
        history['test_loss'].append(test_loss / len(testloader))
        history['test_acc'].append(100. * test_correct / total_test)
        
        print(f"--> Epoch {epoch} Final | Test Acc: {history['test_acc'][-1]:.2f}% | Test Loss: {history['test_loss'][-1]:.4f}")

    print("\nTraining Complete!")
    
    history_path = os.path.join(save_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print("\nGenerating baseline plots...")
    from utils.plots import plot_global_metrics_evolution
    plot_global_metrics_evolution(history, epochs, f"{dataset_name}_baseline", save_dir=save_dir)
    print(f"Plots saved in `{save_dir}/` directory.")
        
    checkpoint_path = os.path.join(save_dir, f'model_vit_baseline_{dataset_name.lower()}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint successfully written to: {checkpoint_path}")

if __name__ == "__main__":
    main()