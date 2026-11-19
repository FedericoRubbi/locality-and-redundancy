import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_norm

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, precompute_grid_size=(32, 32)):
        super().__init__()
        self.dim = dim
        self.grid_size = precompute_grid_size
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        
        base_h, base_w = precompute_grid_size
        t_h = torch.arange(base_h).float()
        t_w = torch.arange(base_w).float()
        
        freqs_h = torch.einsum('i,j->ij', t_h, inv_freq)
        freqs_w = torch.einsum('i,j->ij', t_w, inv_freq)
        
        freqs_h = freqs_h.unsqueeze(1).repeat(1, base_w, 1)
        freqs_w = freqs_w.unsqueeze(0).repeat(base_h, 1, 1)
        
        freqs = torch.cat((freqs_h, freqs_w), dim=-1)
        freqs = freqs.unsqueeze(-1).repeat(1, 1, 1, 2).flatten(2)
        
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)

    def forward(self, x, h, w):
        B, seq_len, N, D = x.shape
        sin = self.sin[:h, :w, :].reshape(1, h*w, 1, D)
        cos = self.cos[:h, :w, :].reshape(1, h*w, 1, D)
        
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + x_rotated * sin

class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.rope = VisionRotaryEmbeddingFast(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, h, w, key_padding_mask=None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.rope(q, h, w)
        k = self.rope(k, h, w)
        
        q_rope = q.permute(0, 2, 1, 3)
        k_rope = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            
        out = F.scaled_dot_product_attention(
            q_rope, k_rope, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_dropout(self.proj(out))

class ConvGLU(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        hidden_dim = int((dim * mlp_ratio) * 2 / 3)
        
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        B, N, C = x.shape
        
        x_proj = self.fc1(x)
        x_gate, x_val = x_proj.chunk(2, dim=-1)
        
        x_gate_spatial = x_gate.transpose(1, 2).reshape(B, -1, h, w)
        x_gate_spatial = self.dwconv(x_gate_spatial)
        x_gate_conv = x_gate_spatial.reshape(B, -1, h * w).transpose(1, 2)
        
        x_gated = self.act(x_gate_conv) * x_val
        x_gated = self.drop(x_gated)
        
        out = self.fc2(x_gated)
        out = self.drop(out)
        
        return out

class HybridEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MHSA(dim, num_heads=num_heads, dropout=dropout)
        
        self.norm2 = RMSNorm(dim)
        self.mlp = ConvGLU(dim, mlp_ratio=mlp_ratio, drop=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, h, w, key_padding_mask=None):
        residual = x
        x_attn = self.norm1(x)
        x_attn = self.attn(x_attn, h, w, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x_attn)
        
        residual = x
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp, h, w)
        x = residual + self.dropout2(x_mlp)
        
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return x, h, w

class ShallowVisualCache(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=3, 
                 embed_dim=384, depth=4, num_heads=8, mlp_ratio=2.0, proj_dim=384):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        self.blocks = nn.ModuleList([
            HybridEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = RMSNorm(embed_dim)
        
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x, h, w = self.patch_embed(x)
        B = x.shape[0]
        
        for block in self.blocks:
            x = block(x, h, w)
            
        x = self.norm(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        pooled_features = features.mean(dim=1)
        proj = self.proj_head(pooled_features)
        return F.normalize(proj, dim=-1)
