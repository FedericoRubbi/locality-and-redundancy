import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def compute_dimensional_entropy(H):
    dim_energy = torch.mean(H ** 2, dim=0)
    total_energy = torch.sum(dim_energy)
    norm_energy = dim_energy / (total_energy + 1e-8)
    norm_energy = torch.clamp(norm_energy, min=1e-10)
    entropy = -torch.sum(norm_energy * torch.log(norm_energy))
    return entropy.item()

def compute_cse_and_auc(H):
    N = H.shape[1]
    dim_energy = torch.mean(H ** 2, dim=0) 
    total_energy = torch.sum(dim_energy)
    norm_energy = dim_energy / (total_energy + 1e-8)
    cse = torch.cumsum(norm_energy, dim=0)
    auc = torch.mean(cse).item()
    return cse, auc

def compute_subspace_metrics(H_A, H_B, k, t=2.0, block_size=2048, max_samples=None):
    h_A_sub = F.normalize(H_A[:, :k], p=2, dim=-1)
    h_B_sub = F.normalize(H_B[:, :k], p=2, dim=-1)
    
    B = h_A_sub.shape[0]
    device = h_A_sub.device
    if max_samples is not None and B > max_samples:
        idx = torch.randperm(B, device=device)[:max_samples]
        h_A_sub = h_A_sub[idx]
        h_B_sub = h_B_sub[idx]
        B = h_A_sub.shape[0]

    alignment = torch.mean(torch.sum((h_A_sub - h_B_sub) ** 2, dim=-1))

    B = h_A_sub.shape[0]
    mask_val = 1e6
    expected_sum_t = torch.tensor(0.0, device=device)
    
    for i in range(0, B, block_size):
        end = min(i + block_size, B)
        pairwise_dot = torch.mm(h_A_sub[i:end], h_A_sub.t())
        pairwise_l2_sq = 2.0 - 2.0 * pairwise_dot

        rows = torch.arange(0, end - i, device=device)
        cols = torch.arange(i, end, device=device)
        pairwise_l2_sq[rows, cols] = mask_val

        expected_sum_t += torch.sum(torch.exp(-t * pairwise_l2_sq))

    mean_exp = expected_sum_t / (B * (B - 1))
    uniformity = torch.log(mean_exp + 1e-8)

    return alignment.item(), uniformity.item()

def compute_effective_rank(H):
    B, N = H.shape
    if B < N:
        print(f"Warning: Batch size B ({B}) must be >= dimension N ({N}) for stable rank. Skipping.")
        return None
        
    H_centered = H - torch.mean(H, dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(H_centered, full_matrices=False)
    
    lambdas = S ** 2
    prob_dist = lambdas / (torch.sum(lambdas) + 1e-8)
    prob_dist = torch.clamp(prob_dist, min=1e-10)
    entropy = -torch.sum(prob_dist * torch.log(prob_dist))
    
    d_eff = torch.exp(entropy).item()
    return d_eff

def compute_representational_drift(H_clean, H_regularized):
    squared_diff = torch.sum((H_clean - H_regularized) ** 2, dim=-1)
    drift = torch.mean(squared_diff)
    return drift.item()

def run_linear_probe(X_train, y_train, X_test, y_test, epochs, lr, batch_size, model=None, return_model=False):
    if isinstance(X_train, np.ndarray): X_train = torch.from_numpy(X_train).float()
    if isinstance(y_train, np.ndarray): y_train = torch.from_numpy(y_train).long()
    if isinstance(X_test, np.ndarray): X_test = torch.from_numpy(X_test).float()
    if isinstance(y_test, np.ndarray): y_test = torch.from_numpy(y_test).long()

    device = X_train.device if isinstance(X_train, torch.Tensor) and X_train.is_cuda else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    num_features = X_train.shape[1]
    num_classes = int(torch.max(y_train).item() + 1)

    if model is None:
        probe = nn.Linear(num_features, num_classes).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

        probe.train()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = loss_fn(probe(xb), yb)
                loss.backward()
                opt.step()
    else:
        probe = model

    probe.eval()
    with torch.no_grad():
        preds = torch.argmax(probe(X_test), dim=1)
        acc = (preds == y_test).float().mean().item()

    if return_model:
        return probe
    return float(acc)