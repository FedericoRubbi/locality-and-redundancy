import torch
import torch.nn.functional as F
import math


def compute_renyi_entropy(h, sigma=1.0):
    with torch.no_grad():
        dists = torch.cdist(h, h, p=2)**2
        K = torch.exp(-dists / (2 * sigma**2))
        h2 = -torch.log(K.mean() + 1e-8)
        return h2.item()


def compute_orthogonality_penalty(W):
    with torch.no_grad():
        W_W_t = torch.mm(W, W.t())
        identity = torch.eye(W_W_t.size(0), device=W.device)
        return F.mse_loss(W_W_t, identity).item()


def compute_beta_m(m, eta=0.1, alpha_sq=1.0, beta_min=0.5):
    leakage_term = eta / (math.log(m + 1) ** alpha_sq)
    beta_m = 1.0 - leakage_term
    return max(beta_min, beta_m)


def compute_capacity_c_m(m, alpha=1.0):
    return alpha / math.sqrt(m * math.log(m + 1))