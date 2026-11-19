import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_variance_schedule(dim, schedule_type="polynomial", sigma_min=1e-5, sigma_max=0.9, gamma=2.0, device="cpu"):
    d = torch.arange(1, dim + 1, dtype=torch.float32, device=device)
    
    if schedule_type == "linear":
        var_schedule = sigma_min + (sigma_max - sigma_min) * ((d - 1) / (dim - 1))
    elif schedule_type == "polynomial":
        var_schedule = sigma_min + (sigma_max - sigma_min) * ((d - 1) / (dim - 1)) ** gamma
    elif schedule_type == "geometric":
        sigma_min = max(sigma_min, 1e-6)
        var_schedule = sigma_min * (sigma_max / sigma_min) ** ((d - 1) / (dim - 1))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
        
    return var_schedule


class AsymmetricAdditiveNoise(nn.Module):
    def __init__(self, var_schedule):
        super().__init__()
        self.register_buffer("var_schedule", var_schedule)
        self.register_buffer("std_schedule", torch.sqrt(var_schedule))

    def forward(self, h_anchor):
        std = self.std_schedule.to(h_anchor.device)
        noise = torch.randn_like(h_anchor) * std.unsqueeze(0)
        return h_anchor + noise

class MultiplicativeLogNormalNoise(nn.Module):
    def __init__(self, var_schedule):
        super().__init__()
        self.register_buffer("var_schedule", var_schedule)
        self.register_buffer("std_schedule", torch.sqrt(var_schedule))

    def forward(self, h):
        std = self.std_schedule.to(h.device)
        var = self.var_schedule.to(h.device)
        xi = torch.randn_like(h) * std.unsqueeze(0)
        epsilon = torch.exp(-0.5 * var.unsqueeze(0) + xi)
        return h * epsilon
    
class CumulativeMultiplicativeBetaNoise(nn.Module):
    def __init__(self, dim, mu=0.98, nu=100.0, renormalize=True):
        super().__init__()
        self.dim = dim
        self.renormalize = renormalize
        
        alpha_val = mu * nu
        beta_val = (1.0 - mu) * nu
        
        if alpha_val <= 1.0 or beta_val <= 1.0:
            raise ValueError(
                f"Invalid parameter pair (mu={mu}, nu={nu}). "
                f"Yields alpha={alpha_val:.2f}, beta={beta_val:.2f}. Both must be > 1.0!"
            )
            
        self.register_buffer("alpha", torch.tensor(alpha_val, dtype=torch.float32))
        self.register_buffer("beta", torch.tensor(beta_val, dtype=torch.float32))

    def forward(self, h):
        if not self.training:
            return h
            
        B, D = h.shape
        
        dist = torch.distributions.Beta(self.alpha, self.beta)
        U = dist.sample((B, D))
        
        m = torch.cumprod(U, dim=-1)
        
        h_noisy = h * m
        
        if self.renormalize:
            return F.normalize(h_noisy, p=2, dim=-1)
        return h_noisy


class HeteroscedasticRandomWalkNoise(nn.Module):
    def __init__(self, dim, gamma=1.02, v_max=0.005, renormalize=True):
        super().__init__()
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if v_max <= 0.0:
            raise ValueError(f"v_max must be positive, got {v_max}")

        self.dim = dim
        self.renormalize = renormalize

        if v_max >= (1.0 / dim):
            print(f"[Warning] v_max {v_max} exceeds 1/D. SNR_D will be < 1.0")

        if abs(gamma - 1.0) < 1e-6:
            sigma0_sq = v_max / dim
        else:
            sigma0_sq = v_max * (gamma - 1.0) / math.expm1(dim * math.log(gamma))

        sigma0 = math.sqrt(sigma0_sq)
        d = torch.arange(1, dim + 1, dtype=torch.float32)
        std_schedule = sigma0 * torch.pow(torch.tensor(gamma, dtype=torch.float32), (d - 1.0) / 2.0)

        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        self.register_buffer("v_max", torch.tensor(v_max, dtype=torch.float32))
        self.register_buffer("sigma0", torch.tensor(sigma0, dtype=torch.float32))
        self.register_buffer("std_schedule", std_schedule)

    def forward(self, h):
        if not self.training:
            return h

        std = self.std_schedule.to(h.device).unsqueeze(0)
        increments = torch.randn_like(h) * std
        xi = torch.cumsum(increments, dim=-1)
        h_noisy = h + xi

        if self.renormalize:
            return F.normalize(h_noisy, p=2, dim=-1)
        return h_noisy


class ExplosiveAR1AdditiveNoise(nn.Module):
    def __init__(self, dim, alpha=1.02, v_max=0.005, renormalize=True):
        super().__init__()
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if v_max <= 0.0:
            raise ValueError(f"v_max must be positive, got {v_max}")

        self.dim = dim
        self.renormalize = renormalize

        if v_max >= (1.0 / dim):
            print(f"[Warning] v_max {v_max} exceeds 1/D. SNR_D will be < 1.0")

        if abs(alpha - 1.0) < 1e-6:
            sigma_sq = v_max / dim
        else:
            sigma_sq = v_max * (alpha**2 - 1.0) / math.expm1(2.0 * dim * math.log(alpha))

        sigma = math.sqrt(sigma_sq)

        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("v_max", torch.tensor(v_max, dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

        indices = torch.arange(dim, dtype=torch.float32)
        dist_matrix = indices.unsqueeze(0) - indices.unsqueeze(1)
        decay_matrix = torch.pow(torch.tensor(alpha, dtype=torch.float32), dist_matrix)
        decay_matrix = torch.tril(decay_matrix)
        self.register_buffer("causal_growth_filter", decay_matrix)

    def forward(self, h):
        if not self.training:
            return h

        epsilon = torch.randn_like(h) * self.sigma.to(h.device)
        xi = torch.matmul(epsilon, self.causal_growth_filter.to(h.device).t())
        h_noisy = h + xi

        if self.renormalize:
            return F.normalize(h_noisy, p=2, dim=-1)
        return h_noisy


class DeterministicNestedDropout(nn.Module):
    def __init__(self, dim, schedule_type="uniform", rho=0.5):
        super().__init__()
        self.dim = dim
        self.schedule_type = schedule_type
        self.rho = rho

    def forward(self, h):
        if not self.training:
            return h
        B, D = h.shape
        
        if self.schedule_type == "geometric":
            u = torch.rand(1, device=h.device)
            k = torch.floor(torch.log(u) / torch.log(torch.tensor(1 - self.rho, device=h.device))) + 1
            k = torch.clamp(k, 1, D).long()
        else:
            k = torch.randint(1, D + 1, (1,), device=h.device)

        dims_range = torch.arange(D, device=h.device)
        mask = (dims_range < k).float().unsqueeze(0)
        
        return F.normalize(h * mask, p=2, dim=-1)

class IndependentNestedDropout(nn.Module):
    def __init__(self, dim, p_max=0.5, gamma=1.5):
        super().__init__()
        d = torch.arange(1, dim + 1, dtype=torch.float32)
        p_schedule = p_max * ((d - 1) / (dim - 1)) ** gamma
        self.register_buffer("p_schedule", p_schedule)

    def forward(self, h):
        if not self.training:
            return h
        B, D = h.shape
        probs = (1.0 - self.p_schedule.to(h.device)).unsqueeze(0).expand(B, D)
        mask = torch.bernoulli(probs)
        return F.normalize(h * mask, p=2, dim=-1)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=3072, embed_dim=128, hidden_dim=512, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)
