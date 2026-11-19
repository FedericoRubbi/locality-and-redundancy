from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.loss import info_nce_loss


@dataclass
class StepContext:
    t: int
    T: int
    emb_A: Tensor
    emb_B: Tensor
    map_A: Tensor | None = None
    map_B: Tensor | None = None
    tgt_A: Tensor | None = None
    tgt_B: Tensor | None = None
    tgt_map_A: Tensor | None = None
    tgt_map_B: Tensor | None = None
    model: nn.Module | None = None
    target_model: nn.Module | None = None


def _log_schedule(lo: float, hi: float, T: int) -> list[float]:
    return torch.exp(torch.linspace(float(torch.log(torch.tensor(lo))),
                                    float(torch.log(torch.tensor(hi))), T)).tolist()


def _linear_schedule(lo: float, hi: float, T: int) -> list[float]:
    return torch.linspace(lo, hi, T).tolist()


class GlobalInfoNCE:

    needs_maps = False
    name = "global_infonce"

    def __init__(self, tau_schedule: list[float], use_ema: bool):
        self.tau = tau_schedule
        self.use_ema = use_ema

    def tau_at(self, t: int) -> float:
        return self.tau[t - 1] if t >= 1 else self.tau[0]

    def __call__(self, ctx: StepContext) -> Tensor:
        tau = self.tau_at(ctx.t)
        if self.use_ema:
            return 0.5 * (_nce_query_key(ctx.emb_A, ctx.tgt_B, tau)
                          + _nce_query_key(ctx.emb_B, ctx.tgt_A, tau))
        return info_nce_loss(ctx.emb_A, ctx.emb_B, temperature=tau)


def _nce_query_key(q: Tensor, k: Tensor, tau: float) -> Tensor:
    logits = (q @ k.t()) / tau
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)


class DenseInfoNCE:

    needs_maps = True
    name = "dense_infonce"

    def __init__(self, tau_schedule: list[float], grid: int, T: int,
                 max_locations: int = 32, k_schedule: list[int] | None = None):
        self.tau = tau_schedule
        self.max_locations = max_locations
        if k_schedule is None:
            max_exp = int(torch.log2(torch.tensor(float(grid))).item())
            exps = torch.linspace(0, max_exp, T).round().int().tolist()
            k_schedule = [2 ** e for e in exps]
        self.k_schedule = k_schedule

    def tau_at(self, t: int) -> float:
        return self.tau[t - 1] if t >= 1 else self.tau[0]

    def k_at(self, t: int) -> int:
        return self.k_schedule[t - 1] if t >= 1 else self.k_schedule[0]

    def __call__(self, ctx: StepContext) -> Tensor:
        k = self.k_at(ctx.t)
        tau = self.tau_at(ctx.t)
        head = ctx.model.dense_head

        zA = head(F.avg_pool2d(ctx.map_A, k) if k > 1 else ctx.map_A)
        zB = head(F.avg_pool2d(ctx.map_B, k) if k > 1 else ctx.map_B)

        B, D = zA.shape[0], zA.shape[1]
        zA = zA.flatten(2).permute(2, 0, 1)
        zB = zB.flatten(2).permute(2, 0, 1)
        L = zA.shape[0]

        if L > self.max_locations:
            idx = torch.randperm(L, device=zA.device)[:self.max_locations]
            zA, zB = zA[idx], zB[idx]
            L = self.max_locations

        sim = torch.bmm(zA, zB.transpose(1, 2)) / tau
        labels = torch.arange(B, device=zA.device).repeat(L)
        loss = 0.5 * (F.cross_entropy(sim.reshape(L * B, B), labels)
                      + F.cross_entropy(sim.transpose(1, 2).reshape(L * B, B), labels))
        return loss


class MaskedJEPA:

    needs_maps = True
    name = "masked_jepa"

    def __init__(self, T: int, mask_low: float = 0.15, mask_high: float = 0.75,
                 var_weight: float = 1.0, var_margin: float = 0.5):
        self.mask_schedule = _linear_schedule(mask_low, mask_high, T)
        self.var_weight = var_weight
        self.var_margin = var_margin
        self.last_stats: dict[str, float] = {}

    def mask_at(self, t: int) -> float:
        return self.mask_schedule[t - 1] if t >= 1 else self.mask_schedule[0]

    def _predict(self, ctx: StepContext, ctx_map: Tensor, tgt_map: Tensor,
                 ratio: float) -> tuple[Tensor, Tensor]:
        model = ctx.model
        B, C, H, W = ctx_map.shape
        L = H * W
        n_mask = max(1, int(round(ratio * L)))

        keep = torch.rand(B, L, device=ctx_map.device).argsort(dim=1)
        mask = torch.zeros(B, L, dtype=torch.bool, device=ctx_map.device)
        mask.scatter_(1, keep[:, :n_mask], True)
        mask_map = mask.view(B, 1, H, W)

        token = model.mask_token.view(1, C, 1, 1).to(ctx_map.dtype)
        context = torch.where(mask_map, token.expand_as(ctx_map), ctx_map)
        pred = model.dense_head(model.predictor(context))

        with torch.no_grad():
            tgt_net = ctx.target_model if ctx.target_model is not None else model
            target = tgt_net.dense_head(tgt_map).detach()

        sel = mask_map.expand(-1, pred.shape[1], -1, -1)
        p = pred.masked_select(sel).view(B, pred.shape[1], -1)
        g = target.masked_select(sel).view(B, target.shape[1], -1)
        loss = (1.0 - (p * g).sum(dim=1)).mean()
        return loss, p, g

    def _collapse_cos(self, g: Tensor) -> Tensor:
        B = g.shape[0]
        z = F.normalize(g.mean(dim=-1), p=2, dim=-1)
        sim = z @ z.t()
        off = (sim.sum() - sim.diagonal().sum()) / max(1, B * (B - 1))
        return off

    def __call__(self, ctx: StepContext) -> Tensor:
        ratio = self.mask_at(ctx.t)
        tgt_A = ctx.tgt_map_A if ctx.tgt_map_A is not None else ctx.map_A.detach()
        tgt_B = ctx.tgt_map_B if ctx.tgt_map_B is not None else ctx.map_B.detach()

        loss_A, pA, gA = self._predict(ctx, ctx.map_A, tgt_B, ratio)
        loss_B, pB, gB = self._predict(ctx, ctx.map_B, tgt_A, ratio)
        loss = 0.5 * (loss_A + loss_B)

        with torch.no_grad():
            cos = 0.5 * (self._collapse_cos(gA) + self._collapse_cos(gB))
            self.last_stats = {"jepa/target_cos": cos.item(),
                               "jepa/mask_ratio": ratio}
        if self.var_weight > 0:
            live = 0.5 * (self._collapse_cos(pA.float()) + self._collapse_cos(pB.float()))
            loss = loss + self.var_weight * F.relu(live - self.var_margin)
        return loss


def build_tau_schedule(training_cfg: dict, T: int) -> list[float]:
    sched = training_cfg.get('temperature_schedule')
    if not sched:
        return [float(training_cfg['temperature'])] * T
    return _log_schedule(float(sched['temp_high']), float(sched['temp_low']), T)


def build_objective(training_cfg: dict, T: int, grid: int, use_ema: bool):
    name = training_cfg.get('objective', 'global_infonce')
    tau = build_tau_schedule(training_cfg, T)

    if name == 'global_infonce':
        return GlobalInfoNCE(tau, use_ema)
    if name == 'dense_infonce':
        cfg = training_cfg.get('dense', {}) or {}
        return DenseInfoNCE(tau, grid=grid, T=T,
                            max_locations=cfg.get('max_locations', 32),
                            k_schedule=cfg.get('k_schedule'))
    if name == 'masked_jepa':
        cfg = training_cfg.get('jepa', {}) or {}
        return MaskedJEPA(T, mask_low=float(cfg.get('mask_low', 0.15)),
                          mask_high=float(cfg.get('mask_high', 0.75)),
                          var_weight=float(cfg.get('var_weight', 1.0)),
                          var_margin=float(cfg.get('var_margin', 0.5)))
    raise ValueError(f"unknown objective: {name}")
