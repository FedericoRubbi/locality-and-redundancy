import torch
import torch.nn as nn
import math

class VariancePredictor(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(3 * embed_dim, 2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Softplus()
        )

    def forward(self, y_t, t, c):
        t_emb = self.t_embed(t.float().unsqueeze(-1))
        z = torch.cat([y_t, t_emb, c], dim=-1)
        return self.net(z)

def get_base_noise_schedule(embed_dim=384, rho_max=10.0):
    k = torch.arange(1, embed_dim + 1, dtype=torch.float32)
    rho = torch.exp((k / embed_dim) * math.log(rho_max))
    return 1.0 / rho
