import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def cosine_schedule(num_timesteps, s=0.008):
    steps = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float32)
    f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionSchedule(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        betas = cosine_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def add_noise(self, y_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y_0)
            
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        y_t = sqrt_alpha_bar * y_0 + sqrt_one_minus_alpha_bar * noise
        return F.normalize(y_t, p=2, dim=-1), noise

def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    v0 = F.normalize(v0, dim=-1)
    v1 = F.normalize(v1, dim=-1)
    
    dot = torch.sum(v0 * v1, dim=-1)
    
    if torch.abs(dot) > DOT_THRESHOLD:
        return nlerp(v0, v1, t)
        
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    v2 = torch.sin((1.0 - t) * omega) / so * v0 + torch.sin(t * omega) / so * v1
    return v2

def nlerp(v0, v1, t):
    v = (1.0 - t) * v0 + t * v1
    return F.normalize(v, dim=-1)
