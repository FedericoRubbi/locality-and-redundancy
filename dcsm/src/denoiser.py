import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLNResBlock(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.ada_ln_proj = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self, x, w):
        gamma, beta = self.ada_ln_proj(w).chunk(2, dim=-1)
        x_norm = self.norm(x)
        x_ada = gamma * x_norm + beta
        
        h = self.linear1(F.gelu(self.linear2(x_ada)))
        return x + h

class Denoiser(nn.Module):
    def __init__(self, embed_dim=384, num_blocks=6):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cond_proj = nn.Linear(embed_dim, embed_dim)
        
        self.input_proj = nn.Linear(2 * embed_dim, embed_dim)
        
        self.blocks = nn.ModuleList([AdaLNResBlock(embed_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, y_t, t, c):
        t_emb = self.t_embed(t.float().unsqueeze(-1))
        w = self.cond_proj(t_emb)
        
        h = torch.cat([y_t, c], dim=-1)
        h = self.input_proj(h)
        
        for block in self.blocks:
            h = block(h, w)
            
        y_0_hat = self.out_proj(h)
        return F.normalize(y_0_hat, p=2, dim=-1)

def ddim_step(y_t, y_0_hat, alphas_bar_t, alphas_bar_t_minus_1):
    eps = (y_t - torch.sqrt(alphas_bar_t) * y_0_hat) / torch.sqrt(1 - alphas_bar_t)
    y_t_minus_1 = torch.sqrt(alphas_bar_t_minus_1) * y_0_hat + torch.sqrt(1 - alphas_bar_t_minus_1) * eps
    return F.normalize(y_t_minus_1, p=2, dim=-1)