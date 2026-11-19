import torch
import torch.nn.functional as F


def compute_asymmetric_similarity(z1, z2, gamma):
    dot_product = torch.matmul(z1, z2.T)

    if gamma == 0.0:
        return dot_product
    
    sum_z1 = z1.sum(dim=-1)
    sum_z2 = z2.sum(dim=-1)
    
    asym_bias = sum_z1.unsqueeze(1) + sum_z2.unsqueeze(0)
    
    return dot_product + gamma * asym_bias


def info_nce_loss(z1, z2, temperature=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    n = 2 * B
    diag = torch.eye(n, device=sim.device).bool()
    sim = sim.masked_fill(diag, -1e9)

    targets = torch.arange(n, device=sim.device)
    positives = (targets + B) % n

    loss = F.cross_entropy(sim, positives)
    return loss


def asymmetric_info_nce_loss(z1, z2, temperature, gamma=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = compute_asymmetric_similarity(z, z, gamma) / temperature
    n = 2 * B
    diag = torch.eye(n, device=sim.device).bool()
    sim = sim.masked_fill(diag, -1e9)

    targets = torch.arange(n, device=sim.device)
    positives = (targets + B) % n

    loss = F.cross_entropy(sim, positives)
    return loss


def compensated_asymmetric_info_nce_loss(z1, z2, var_schedule, mode="additive", gamma=0.1, temperature=0.1):
    B = z1.size(0)
    
    raw_sim = compute_asymmetric_similarity(z1, z2, gamma=gamma) / temperature
    
    if mode == "additive":
        variance_footprint = torch.sum((z2 ** 2) * var_schedule.unsqueeze(0), dim=-1)
        correction = variance_footprint.unsqueeze(0) / (2.0 * (temperature ** 2))
        
    elif mode == "multiplicative":
        scaling_factor = torch.exp(var_schedule) - 1.0
        z1_scaled = z1 ** 2
        z2_scaled = (z2 ** 2) * scaling_factor.unsqueeze(0)
        correction = torch.matmul(z1_scaled, z2_scaled.T) / (2.0 * (temperature ** 2))
        
    else:
        raise ValueError(f"Unknown variance compensation mode: {mode}")
        
    corrected_logits = raw_sim - correction
    
    targets = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(corrected_logits, targets)
    return loss


def matryoshka_info_nce_loss(z1, z2, capacities=None, weights=None, temperature=0.1, sign_bias=0.1):
    if capacities == None:
        D = z1.size(1)
        capacities = [D // (2**i) for i in reversed(range(4))]
    if weights == None:
        weights = [1.0] * len(capacities)
        
    total_loss = 0.0
    B = z1.size(0)
    
    for cap, w in zip(capacities, weights):
        z1_cap = z1[:, :cap]
        z2_cap = z2[:, :cap]
        
        z1_cap = F.normalize(z1_cap, p=2, dim=-1)
        z2_cap = F.normalize(z2_cap, p=2, dim=-1)
        
        sim = compute_asymmetric_similarity(z1_cap, z2_cap, gamma=sign_bias) / temperature
        targets = torch.arange(B, device=z1.device)
        cap_loss = F.cross_entropy(sim, targets)
        
        total_loss += w * cap_loss
        
    return total_loss / len(capacities)
