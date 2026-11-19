import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)


class AdaLN(nn.Module):
    def __init__(self, time_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_dim * 2)
        )
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TimeConditionedMLPBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1, mlp_ratio: int = 4):
        super().__init__()
        self.adaln = AdaLN(time_dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        x_modulated = self.adaln(x, t_emb)
        out = self.mlp(x_modulated)
        return residual + out


def geodesic_step(h: torch.Tensor, v: torch.Tensor, delta: float) -> torch.Tensor:
    D = h.shape[-1]

    dot_vh = torch.sum(v * h, dim=-1, keepdim=True)
    v_tangent = v - dot_vh * h

    v_raw_norm = torch.norm(v_tangent, p=2, dim=-1, keepdim=True) + 1e-8

    v_rms = v_raw_norm / math.sqrt(D)
    step_size = delta * v_rms / (1.0 + v_rms)

    v_direction = v_tangent / v_raw_norm

    return h * torch.cos(step_size) + v_direction * torch.sin(step_size)


class RecurrentEncoder(nn.Module):
    def __init__(self, num_features: int = 784, time_dim: int = 128, num_blocks: int = 3,
                 dropout: float = 0.1, delta: float = 0.5, mlp_ratio: int = 4):
        super().__init__()
        self.delta = delta
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.blocks = nn.ModuleList([
            TimeConditionedMLPBlock(num_features, time_dim, dropout, mlp_ratio)
            for _ in range(num_blocks)
        ])
        self.v_norm = nn.LayerNorm(num_features)

    def forward(self, h: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(step)
        v = h
        for block in self.blocks:
            v = block(v, t_emb)
        v = self.v_norm(v)
        return geodesic_step(h, v, self.delta)


class BidirectionalDecoder(nn.Module):
    def __init__(self, num_features: int = 784, time_dim: int = 128, num_blocks: int = 3,
                 dropout: float = 0.1, delta: float = 0.5, mlp_ratio: int = 4):
        super().__init__()
        self.delta = delta
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.blocks = nn.ModuleList([
            TimeConditionedMLPBlock(num_features, time_dim, dropout, mlp_ratio)
            for _ in range(num_blocks)
        ])
        self.v_norm = nn.LayerNorm(num_features)

    def forward(self, h: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(step)
        v = h
        for block in self.blocks:
            v = block(v, t_emb)
        v = self.v_norm(v)
        return geodesic_step(h, v, self.delta)


class Adapter(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 512, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(v), p=2, dim=-1)


class ConvAdapter(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int], embed_dim: int, patch_size: int = 8):
        super().__init__()
        C, H, W = in_shape
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"Image size {H}x{W} not divisible by patch_size {patch_size}"
        num_patches = (H // patch_size) * (W // patch_size)
        assert embed_dim % num_patches == 0, \
            f"embed_dim {embed_dim} must be a multiple of num_patches {num_patches}"
        channels = embed_dim // num_patches
        self.conv = nn.Conv2d(C, channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.conv(v).flatten(1), p=2, dim=-1)


class LightConvAdapter(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int], embed_dim: int,
                 widths: tuple[int, ...] = (64, 128, 256), pool: int = 2):
        super().__init__()
        C, H, W = in_shape
        chans = [C, *widths]
        layers: list[nn.Module] = []
        for i in range(len(widths)):
            layers += [
                nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(32, chans[i + 1]), num_channels=chans[i + 1]),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.proj = nn.Linear(widths[-1] * pool * pool, embed_dim)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        f = self.conv(v)
        f = self.pool(f).flatten(1)
        return F.normalize(self.proj(f), p=2, dim=-1)


class ResNetAdapter(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int], embed_dim: int,
                 arch: str = "resnet50", pretrained: bool = False,
                 small_input: bool = True, keep_maxpool: bool = True):
        super().__init__()
        C, H, W = in_shape
        weights = "DEFAULT" if pretrained else None
        backbone = getattr(torchvision.models, arch)(weights=weights)
        assert hasattr(backbone, "fc"), f"Unsupported backbone arch: {arch}"
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        if small_input:
            backbone.conv1 = nn.Conv2d(C, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not keep_maxpool:
                backbone.maxpool = nn.Identity()
        elif C != 3:
            backbone.conv1 = nn.Conv2d(C, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = backbone
        self.feat_dim = feat_dim
        self.proj = nn.Identity() if feat_dim == embed_dim else nn.Linear(feat_dim, embed_dim)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        f = self.backbone(v)
        return F.normalize(self.proj(f), p=2, dim=-1)
