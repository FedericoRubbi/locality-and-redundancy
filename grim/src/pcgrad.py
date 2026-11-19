import torch


def flatten_grads(grads: tuple, params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def assign_flat_grad(params: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset:offset + n].view_as(p)
        offset += n
    assert offset == flat.numel(), f"gradient size mismatch: {offset} vs {flat.numel()}"


def gradient_stats(grads: list[torch.Tensor]) -> dict:
    norms = [g.norm().item() for g in grads]
    stats = {f"grad/norm_step_{t + 1}": n for t, n in enumerate(norms)}

    cosines = []
    for t in range(1, len(grads)):
        denom = (grads[t].norm() * grads[t - 1].norm()).clamp_min(1e-12)
        cos = (torch.dot(grads[t], grads[t - 1]) / denom).item()
        cosines.append(cos)
        stats[f"grad/cos_step_{t + 1}_vs_{t}"] = cos
    if cosines:
        stats["grad/mean_consecutive_cos"] = sum(cosines) / len(cosines)

    conflicts = comparisons = 0
    for t in range(len(grads)):
        for k in range(t):
            comparisons += 1
            if torch.dot(grads[t], grads[k]).item() < 0:
                conflicts += 1
    if comparisons:
        stats["grad/conflict_rate"] = conflicts / comparisons
    return stats


def pcgrad_causal(grads: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros_like(grads[0])
    for t, g in enumerate(grads):
        g_proj = g.clone()
        for k in range(t):
            dot = torch.dot(g_proj, grads[k])
            if dot < 0:
                g_proj -= (dot / torch.dot(grads[k], grads[k]).clamp_min(1e-12)) * grads[k]
        total += g_proj
    return total
