import torch
import torch.nn as nn
import torch.nn.functional as F

from src.grim_models import SinusoidalTimeEmbedding


def num_groups(channels: int, target: int = 32) -> int:
    g = min(target, channels)
    while channels % g != 0:
        g -= 1
    return g


class AdaGN(nn.Module):

    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups(channels), channels, affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1 + scale) + shift


class TimeConditionedConvBlock(nn.Module):

    def __init__(self, channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = AdaGN(time_dim, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = AdaGN(time_dim, channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x, t_emb)))
        h = self.conv2(self.drop(F.gelu(self.norm2(h, t_emb))))
        return x + h


class ConvStem(nn.Module):

    def __init__(self, in_shape: tuple[int, int, int], channels: int = 256,
                 strides: tuple[int, ...] = (2, 2)):
        super().__init__()
        C, H, W = in_shape
        widths = [max(channels // (2 ** (len(strides) - 1 - i)), 32) for i in range(len(strides))]
        widths[-1] = channels
        chans = [C, *widths]
        layers: list[nn.Module] = []
        for i, stride in enumerate(strides):
            layers += [
                nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(num_groups(chans[i + 1]), chans[i + 1]),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)
        self.out_shape = (channels, H // int(torch.tensor(strides).prod()), W // int(torch.tensor(strides).prod()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SphericalProjectionHead(nn.Module):

    def __init__(self, channels: int, embed_dim: int, pool: int = 2):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups(channels), channels)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.feat_dim = channels * pool * pool
        self.proj = nn.Linear(self.feat_dim, embed_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f = self.pool(F.gelu(self.norm(x))).flatten(1)
        h = F.normalize(self.proj(f), p=2, dim=-1)
        return (f, h) if return_features else h


class DenseProjectionHead(nn.Module):

    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups(channels), channels)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(F.gelu(self.norm(x))), p=2, dim=1)


class LatentPredictor(nn.Module):

    def __init__(self, channels: int, time_dim: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups(channels), channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SpatialRecurrentEncoder(nn.Module):

    def __init__(self, channels: int, time_dim: int = 128, num_blocks: int = 3,
                 delta_task: float = 0.5, alpha: float = 0.5, dropout: float = 0.0,
                 tangent_projection: bool = True):
        super().__init__()
        assert 0.0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}"
        self.delta_task = delta_task
        self.alpha = alpha
        self.tangent_projection = tangent_projection
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.blocks = nn.ModuleList([
            TimeConditionedConvBlock(channels, time_dim, dropout) for _ in range(num_blocks)
        ])
        self.out_norm = nn.GroupNorm(num_groups(channels), channels, affine=False)

    @staticmethod
    def _tangent(v: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        flat_v, flat_s = v.flatten(1), state.flatten(1)
        coef = ((flat_v * flat_s).sum(-1, keepdim=True)
                / (flat_s * flat_s).sum(-1, keepdim=True).clamp_min(1e-8))
        return v - coef.view(-1, 1, 1, 1) * state

    def forward(self, state: torch.Tensor, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_emb = self.time_embed(step)
        u = state
        for block in self.blocks:
            u = block(u, t_emb)

        if self.tangent_projection:
            u = self._tangent(u, state)
        v1 = self.delta_task * self.out_norm(u)
        if self.tangent_projection:
            v1 = self._tangent(v1, state)

        return state + v1, state + self.alpha * v1, v1


class SRRModel(nn.Module):

    def __init__(self, in_shape: tuple[int, int, int], channels: int = 256,
                 stem_strides: tuple[int, ...] = (2, 2), num_blocks: int = 3,
                 time_dim: int = 128, embed_dim: int = 1024, pool: int = 2,
                 delta_task: float = 0.5, alpha: float = 0.5, dropout: float = 0.0,
                 tangent_projection: bool = True, dense_head: bool = False,
                 dense_embed_dim: int = 256, masked_predictor: bool = False):
        super().__init__()
        self.stem = ConvStem(in_shape, channels=channels, strides=stem_strides)
        self.encoder = SpatialRecurrentEncoder(
            channels, time_dim=time_dim, num_blocks=num_blocks,
            delta_task=delta_task, alpha=alpha, dropout=dropout,
            tangent_projection=tangent_projection)
        self.head = SphericalProjectionHead(channels, embed_dim, pool=pool)

        self.dense_head = DenseProjectionHead(channels, dense_embed_dim) if dense_head else None
        if masked_predictor:
            self.predictor = LatentPredictor(channels)
            self.mask_token = nn.Parameter(torch.zeros(channels))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.predictor = None

    def rollout(self, x: torch.Tensor, T: int, detach_state: bool = False,
                return_features: bool = False, collect_diagnostics: bool = False,
                return_maps: bool = False):
        assert not (collect_diagnostics and return_features), \
            "diagnostics read the embedding directly and cannot be combined with return_features"
        state = self.stem(x)
        outs = [self.head(state, return_features=return_features)]
        maps: list[torch.Tensor] = [state]
        diagnostics: list[dict] = []
        prev_v = None

        for t in range(T):
            if detach_state:
                state = state.detach()
            step = torch.full((x.shape[0], 1), float(t), device=x.device, dtype=state.dtype)
            z, next_state, v = self.encoder(state, step)
            outs.append(self.head(z, return_features=return_features))
            maps.append(z)

            if collect_diagnostics:
                with torch.no_grad():
                    flat_v, flat_s = v.flatten(1).float(), state.flatten(1).float()
                    entry = {
                        "rel_displacement": (self.encoder.alpha * flat_v.norm(dim=-1)
                                             / (flat_s.norm(dim=-1) + 1e-8)).mean().item(),
                        "state_norm": flat_s.norm(dim=-1).mean().item(),
                        "readout_cos_to_stem": F.cosine_similarity(
                            outs[0].float(), outs[-1].float(), dim=-1).mean().item(),
                    }
                    if prev_v is not None:
                        cos = F.cosine_similarity(prev_v, flat_v, dim=-1)
                        entry["step_angle"] = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)).mean().item()
                    diagnostics.append(entry)
                    prev_v = flat_v

            state = next_state

        if return_maps:
            return (outs, maps, diagnostics) if collect_diagnostics else (outs, maps)
        return (outs, diagnostics) if collect_diagnostics else outs

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        return self.rollout(x, T)[-1]
