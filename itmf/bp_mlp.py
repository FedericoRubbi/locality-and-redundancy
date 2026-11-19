import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from dataset import setup_dataset


class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList()
        self.probes = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        curr_dim = input_dim
        for i in range(num_layers):
            layer = nn.Linear(curr_dim, hidden_dim)
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout))
            self.probes.append(nn.Linear(hidden_dim, num_classes))
            curr_dim = hidden_dim

    def forward(self, x):
        h = x
        h_outputs = []
        probe_logits = []
        for i in range(self.num_layers):
            h = F.relu(self.layers[i](h))
            h = self.dropouts[i](h)
            h_outputs.append(h)
            probe_logits.append(self.probes[i](h))
            
        return probe_logits

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    num_layers = model.num_layers
    
    total_samples = 0
    layer_loss = np.zeros(num_layers)
    layer_correct = np.zeros(num_layers)
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)
        total_samples += batch_size
        
        optimizer.zero_grad()
        
        h = inputs
        h_outputs = []
        for i in range(num_layers):
            h = F.relu(model.layers[i](h))
            h = model.dropouts[i](h)
            h_outputs.append(h)
            
        probe_logits = []
        for i in range(num_layers):
            if i < num_layers - 1:
                probe_logits.append(model.probes[i](h_outputs[i].detach()))
            else:
                probe_logits.append(model.probes[i](h_outputs[i]))
                
        probe_losses = []
        for i in range(num_layers):
            loss = criterion(probe_logits[i], targets)
            probe_losses.append(loss)
            
            layer_loss[i] += loss.item() * batch_size
            _, predicted = probe_logits[i].max(1)
            layer_correct[i] += predicted.eq(targets).sum().item()
            
        combined_loss = sum(probe_losses)
        combined_loss.backward()
        optimizer.step()
        
    layer_loss /= total_samples
    layer_acc = 100. * layer_correct / total_samples
    
    return layer_loss, layer_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    num_layers = model.num_layers
    
    total_samples = 0
    layer_loss = np.zeros(num_layers)
    layer_correct = np.zeros(num_layers)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.size(0)
            total_samples += batch_size
            
            probe_logits = model(inputs)
            
            for i in range(num_layers):
                loss = criterion(probe_logits[i], targets)
                layer_loss[i] += loss.item() * batch_size
                _, predicted = probe_logits[i].max(1)
                layer_correct[i] += predicted.eq(targets).sum().item()
                
    layer_loss /= total_samples
    layer_acc = 100. * layer_correct / total_samples
    
    return layer_loss, layer_acc

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "F-MNIST", "CIFAR10", "CIFAR100"])
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden-dim", type=int, default=3072)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, input_dim, num_classes = setup_dataset(args.dataset, args.batch_size, args.data_dir)
    model = BaselineMLP(input_dim, args.hidden_dim, num_classes, args.num_layers, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        _, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"epoch {epoch + 1}: test accuracy per layer " + " ".join(f"{a:.1f}" for a in test_acc))


if __name__ == "__main__":
    main()
