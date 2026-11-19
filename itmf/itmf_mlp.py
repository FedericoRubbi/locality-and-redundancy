import argparse
import gc
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import setup_dataset
from metrics import compute_orthogonality_penalty, compute_renyi_entropy
from plotting import (plot_all_metrics_evolution, plot_alpha_progression, plot_global_metrics_evolution,
                      plot_layerwise_test_train_metrics, plot_standard_metrics)


class LocalAutoencoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, tie_weights=False, activation='relu', dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tie_weights = tie_weights
        
        self.normalization = nn.LayerNorm(input_dim)
        self.encoder = nn.Linear(input_dim, output_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)
        
        if not self.tie_weights:
            self.decoder = nn.Linear(output_dim, input_dim)
        else:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, h_prev):
        h_prev = self.normalization(h_prev)
        h_i = self.activation(self.encoder(h_prev))
        h_i = self.dropout(h_i)
        
        if self.tie_weights:
            h_prev_hat = F.linear(h_i, self.encoder.weight.t(), self.decoder_bias)
        else:
            h_prev_hat = self.decoder(h_i)
            
        h_prev_hat = self.activation(h_prev_hat)
        return h_i, h_prev_hat


class LocallyDrivenNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks, tie_weights=False, activation='relu', dropout=0.0):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        
        self.blocks = nn.ModuleList()
        self.prediction_heads = nn.ModuleList()
        
        curr_dim = input_dim
        for _ in range(num_blocks):
            self.blocks.append(LocalAutoencoderBlock(curr_dim, hidden_dim, tie_weights, activation, dropout))
            self.prediction_heads.append(nn.Linear(hidden_dim, num_classes))
            curr_dim = hidden_dim

    def forward(self, x):
        h = x
        h_inputs = []  
        h_outputs = [] 
        reconstructions = []
        
        for block in self.blocks:
            h = h.detach().requires_grad_(True)
            h_inputs.append(h)
            
            h_next, h_prev_hat = block(h)
            
            h_outputs.append(h_next)
            reconstructions.append(h_prev_hat)
            h = h_next

        return h_inputs, h_outputs, reconstructions


class LocallyDrivenTrainer:
    def __init__(self, model, train_loader, test_loader, device, num_classes, lr=1e-3, gamma_rec=1.0, alpha_max=10.0, schedule_type='cosine', weight_decay=0.0, use_scheduler=False, epochs=100):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        
        self.gamma_rec = gamma_rec
        self.alpha_max = alpha_max
        self.schedule_type = schedule_type
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.scheduler = None
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def get_alpha(self, i):
        if self.model.num_blocks == 1:
            return self.alpha_max
        
        t = i / (self.model.num_blocks - 1)
        
        if self.schedule_type == 'cosine':
            return self.alpha_max * (1 - math.cos((math.pi / 2) * t))
        elif self.schedule_type == 'quadratic':
            return self.alpha_max * (t ** 2)
        elif self.schedule_type == 'exponential':
            return self.alpha_max * (math.exp(t) - 1) / (math.e - 1)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        num_blocks = self.model.num_blocks
        epoch_metrics = {
            'ortho': np.zeros(num_blocks), 'ent_mean': np.zeros(num_blocks),
            'mag_mean': np.zeros(num_blocks), 'layer_loss': np.zeros(num_blocks), 
            'layer_acc': np.zeros(num_blocks), 'layer_rec_loss': np.zeros(num_blocks),
            'alphas': [self.get_alpha(i) for i in range(num_blocks)]
        }
        
        steps = 0
        total_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for step, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            steps += 1
            
            y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
            
            self.optimizer.zero_grad()
            
            h_inputs, h_outputs, reconstructions = self.model(inputs)

            losses = []
            logits_list = []

            for i in range(num_blocks):
                alpha_i = epoch_metrics['alphas'][i]

                prior = alpha_i * y_onehot + 1.0 
                noisy_targets = torch.distributions.Dirichlet(prior).sample()
                
                logits_i = self.model.prediction_heads[i](h_outputs[i])
                log_probs = F.log_softmax(logits_i, dim=-1)
                ce_loss = torch.sum(-noisy_targets * log_probs, dim=-1).mean()
                
                rec_loss = 1.0 - F.cosine_similarity(reconstructions[i], h_inputs[i], dim=-1).mean()
                
                total_block_loss = ce_loss + self.gamma_rec * rec_loss
                
                total_block_loss.backward()
                
                losses.append(ce_loss)
                logits_list.append(logits_i)
                epoch_metrics['layer_rec_loss'][i] += rec_loss.item()
                epoch_metrics['layer_loss'][i] += ce_loss.item()
                
                _, predicted_i = logits_i.max(1)
                epoch_metrics['layer_acc'][i] += predicted_i.eq(targets).sum().item() / targets.size(0) * 100.0

            self.optimizer.step()
            
            if step == total_batches - 1:
                with torch.no_grad():
                    for i, block in enumerate(self.model.blocks):
                        h_i = h_outputs[i]
                        epoch_metrics['ortho'][i] = compute_orthogonality_penalty(block.encoder.weight)
                        epoch_metrics['ent_mean'][i] = compute_renyi_entropy(h_i)
                        epoch_metrics['mag_mean'][i] = torch.norm(h_i, dim=1).mean().item()
            
            total_loss += losses[-1].item()
            _, predicted = logits_list[-1].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'Loss': f'{total_loss/steps:.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        if self.scheduler:
            self.scheduler.step()

        for key in ['layer_loss', 'layer_acc', 'layer_rec_loss']:
            epoch_metrics[key] /= steps

        return total_loss / steps, 100. * correct / total, epoch_metrics

    def evaluate(self):
        self.model.eval()
        total_loss, correct, total, steps = 0, 0, 0, 0
        num_blocks = self.model.num_blocks
        layer_test_loss = np.zeros(num_blocks)
        layer_test_acc = np.zeros(num_blocks)
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                h_inputs, h_outputs, _ = self.model(inputs)
                
                for i in range(num_blocks):
                    logits_i = self.model.prediction_heads[i](h_outputs[i])
                    loss_i = criterion(logits_i, targets)
                    
                    layer_test_loss[i] += loss_i.item()
                    _, predicted_i = logits_i.max(1)
                    layer_test_acc[i] += predicted_i.eq(targets).sum().item() / targets.size(0) * 100.0
                    
                    if i == num_blocks - 1:
                        total_loss += loss_i.item()
                        total += targets.size(0)
                        correct += predicted_i.eq(targets).sum().item()
                steps += 1
                
        return total_loss / steps, 100. * correct / total, layer_test_loss / steps, layer_test_acc / steps


def run_experiment(model, trainer, epochs, save_dir='.'):
    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
        'layer_train_loss': [], 'layer_train_acc': [], 'layer_rec_loss': [],
        'layer_test_loss': [], 'layer_test_acc': [], 'alphas': [],
        'ortho': [], 'ent_mean': [], 'mag_mean': []
    }
    
    try:
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            train_loss, train_acc, epoch_metrics = trainer.train_epoch()
            test_loss, test_acc, layer_test_loss, layer_test_acc = trainer.evaluate()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['alphas'].append(epoch_metrics['alphas'])
            
            history['layer_train_loss'].append(epoch_metrics['layer_loss'].tolist())
            history['layer_train_acc'].append(epoch_metrics['layer_acc'].tolist())
            history['layer_rec_loss'].append(epoch_metrics['layer_rec_loss'].tolist())
            history['layer_test_loss'].append(layer_test_loss.tolist())
            history['layer_test_acc'].append(layer_test_acc.tolist())
            
            for key in ['ortho', 'ent_mean', 'mag_mean']:
                history[key].append(epoch_metrics[key].tolist())
                
            print(f"Test Acc: {test_acc:.2f}% | Final Block Rec Loss: {epoch_metrics['layer_rec_loss'][-1]:.4f}")
    except KeyboardInterrupt:
        print("\n[!] Halting training gracefully.")

    return history


def train(cfg, dataset, epochs, device, data_dir="./data", plots_dir=None):
    train_loader, test_loader, input_dim, num_classes = setup_dataset(dataset, cfg["batch_size"], data_dir)
    model = LocallyDrivenNetwork(input_dim, cfg["hidden_dim"], num_classes, cfg["num_blocks"],
                                 tie_weights=cfg["tie_weights"], activation=cfg["activation"],
                                 dropout=cfg["dropout"]).to(device)
    trainer = LocallyDrivenTrainer(model, train_loader, test_loader, device, num_classes=num_classes, lr=cfg["lr"],
                                   gamma_rec=cfg["gamma_rec"], alpha_max=cfg["alpha_max"],
                                   schedule_type=cfg["schedule_type"], weight_decay=cfg["weight_decay"],
                                   use_scheduler=True, epochs=epochs)
    history = run_experiment(model, trainer, epochs)
    done = len(history["train_loss"])
    if plots_dir and done:
        os.makedirs(plots_dir, exist_ok=True)
        blocks = cfg["num_blocks"]
        plot_global_metrics_evolution(history, done, dataset, save_dir=plots_dir)
        plot_all_metrics_evolution(history, done, blocks, dataset, save_dir=plots_dir)
        plot_layerwise_test_train_metrics(history, done, blocks, dataset, save_dir=plots_dir)
        plot_alpha_progression(history, blocks, dataset, save_dir=plots_dir)
        plot_standard_metrics(history, blocks, dataset, done, save_dir=plots_dir)
    del model, trainer, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def search_space(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "gamma_rec": trial.suggest_float("gamma_rec", 0.0, 10.0),
        "alpha_max": trial.suggest_float("alpha_max", 1.0, 20.0),
        "schedule_type": trial.suggest_categorical("schedule_type", ["cosine", "quadratic", "exponential"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
        "tie_weights": trial.suggest_categorical("tie_weights", [True, False]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "F-MNIST", "CIFAR10", "CIFAR100"])
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden-dim", type=int, default=3072)
    ap.add_argument("--num-blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma-rec", type=float, default=1.0, help="weight of the reconstruction term")
    ap.add_argument("--alpha-max", type=float, default=10.0, help="Dirichlet concentration of the last block")
    ap.add_argument("--schedule", default="exponential", choices=["cosine", "quadratic", "exponential"])
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--activation", default="relu", choices=["relu", "gelu", "silu"])
    ap.add_argument("--tie-weights", action="store_true", help="tie the decoder to the transposed encoder")
    ap.add_argument("--plots-dir", default=None, help="write training plots here")
    ap.add_argument("--search-trials", type=int, default=0, help="run an Optuna search with this many trials")
    ap.add_argument("--storage", default=None, help="Optuna storage, e.g. sqlite:///itmf.db")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed = {"hidden_dim": args.hidden_dim, "num_blocks": args.num_blocks}

    if args.search_trials:
        import optuna

        def objective(trial):
            history = train({**search_space(trial), **fixed}, args.dataset, args.epochs, device, args.data_dir)
            return history["test_acc"][-1]

        study = optuna.create_study(study_name=f"itmf_{args.dataset}", storage=args.storage, load_if_exists=True,
                                    direction="maximize")
        study.optimize(objective, n_trials=args.search_trials)
        print(f"best trial {study.best_trial.number}: {study.best_value:.2f}%  {study.best_trial.params}")
        return

    cfg = {"lr": args.lr, "gamma_rec": args.gamma_rec, "alpha_max": args.alpha_max, "schedule_type": args.schedule,
           "weight_decay": args.weight_decay, "batch_size": args.batch_size, "dropout": args.dropout,
           "activation": args.activation, "tie_weights": args.tie_weights, **fixed}
    history = train(cfg, args.dataset, args.epochs, device, args.data_dir, args.plots_dir)
    print(f"final test accuracy (last block): {history['test_acc'][-1]:.2f}%")


if __name__ == "__main__":
    main()
