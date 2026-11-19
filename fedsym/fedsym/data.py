import math
import os

import numpy as np
import torch
import torch.nn.functional as F

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def _download(name):
    import torchvision.datasets as D
    cls = {"mnist": D.MNIST, "fmnist": D.FashionMNIST, "cifar10": D.CIFAR10}[name]
    root = os.path.join(DATA_ROOT, "raw")
    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = cls(root, train=train, download=True)
        x = torch.as_tensor(np.asarray(ds.data))
        x = x[:, None] if x.ndim == 3 else x.permute(0, 3, 1, 2)
        out[f"x{split}"] = x.contiguous().to(torch.uint8)
        out[f"y{split}"] = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
    return out


def load_dataset(name, device):
    os.makedirs(DATA_ROOT, exist_ok=True)
    cache = os.path.join(DATA_ROOT, f"{name}.pt")
    if not os.path.exists(cache):
        torch.save(_download(name), cache)
    d = torch.load(cache)
    mean, std = STATS[name]
    d = {k: v.to(device) for k, v in d.items()}
    d["mean"] = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    d["std"] = torch.tensor(std, device=device).view(1, -1, 1, 1)
    d["num_classes"] = 10
    d["in_dim"] = d["xtrain"][0].numel()
    d["name"] = name
    return d


def normalize_flat(x_uint8_or_float, d):
    x = x_uint8_or_float
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    return ((x - d["mean"]) / d["std"]).flatten(1)


def partition(y, num_clients, alpha, seed, min_size=64):
    y = y.cpu().numpy()
    rng = np.random.default_rng(seed)
    n = len(y)
    if alpha <= 0:
        perm = rng.permutation(n)
        return [torch.as_tensor(np.sort(p)) for p in np.array_split(perm, num_clients)]
    classes = np.unique(y)
    for _ in range(1000):
        buckets = [[] for _ in range(num_clients)]
        for c in classes:
            idx = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                buckets[k].extend(chunk.tolist())
        if min(len(b) for b in buckets) >= min_size:
            return [torch.as_tensor(np.sort(np.array(b))) for b in buckets]
    raise RuntimeError("could not satisfy min_size; increase alpha or reduce clients")


class GPUAugment:

    def __init__(self, d):
        self.d = d
        name = d["name"]
        self.color = name == "cifar10"
        self.flip = name in ("cifar10", "fmnist")
        self.scale = (0.3, 1.0) if name == "cifar10" else (0.4, 1.0)
        self.jitter = 0.4

    def __call__(self, x, gen=None):
        B = x.shape[0]
        dev = x.device
        r = lambda *s: torch.rand(*s, device=dev, generator=gen)
        x = x.float() / 255.0
        area = self.scale[0] + (self.scale[1] - self.scale[0]) * r(B)
        logr = math.log(3 / 4) + (math.log(4 / 3) - math.log(3 / 4)) * r(B)
        ratio = torch.exp(logr)
        w = torch.sqrt(area * ratio).clamp(max=1.0)
        h = torch.sqrt(area / ratio).clamp(max=1.0)
        cx = (2 * r(B) - 1) * (1 - w)
        cy = (2 * r(B) - 1) * (1 - h)
        flip = torch.where((r(B) < 0.5) & self.flip, -1.0, 1.0)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = w * flip
        theta[:, 0, 2] = cx
        theta[:, 1, 1] = h
        theta[:, 1, 2] = cy
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

        s = self.jitter
        apply = (r(B) < 0.8).float().view(B, 1, 1, 1)
        bright = 1 + s * (2 * r(B) - 1)
        contr = 1 + s * (2 * r(B) - 1)
        xj = x * bright.view(B, 1, 1, 1)
        m = xj.mean(dim=(1, 2, 3), keepdim=True)
        xj = (xj - m) * contr.view(B, 1, 1, 1) + m
        if self.color:
            sat = 1 + s * (2 * r(B) - 1)
            gray = (0.299 * xj[:, :1] + 0.587 * xj[:, 1:2] + 0.114 * xj[:, 2:3])
            xj = gray + (xj - gray) * sat.view(B, 1, 1, 1)
        x = apply * xj + (1 - apply) * x
        if self.color:
            g = (r(B) < 0.2).float().view(B, 1, 1, 1)
            gray = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
            x = g * gray + (1 - g) * x
        x = x.clamp(0, 1)
        return ((x - self.d["mean"]) / self.d["std"]).flatten(1)
