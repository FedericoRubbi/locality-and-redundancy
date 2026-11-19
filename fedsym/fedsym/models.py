import math

import torch
import torch.nn as nn
import torch.nn.functional as F

W_MECHS = ("wasym", "wasym_band", "wfixsym")
RES_MECHS = ("res_id", "res_diag", "res_rand")
DROP_MECHS = ("dropuni", "dropasym")
HIDDEN_MECHS = W_MECHS + RES_MECHS + DROP_MECHS + ("lrasym", "act_asym")
OUT_MECHS = ("diag", "mrl")
LEGACY = {"sym": ([], "none"), "diag": ([], "diag"), "mrl": ([], "mrl"), "wasym_mrl": (["wasym"], "mrl"),
          "dropasym_mrl": (["dropasym"], "mrl")}


def parse_variant(name):
    if name in LEGACY:
        return list(LEGACY[name][0]), LEGACY[name][1]
    hs, osb = [], "none"
    for m in name.split("+"):
        if m in OUT_MECHS:
            osb = m
        elif m in HIDDEN_MECHS:
            hs.append(m)
        else:
            raise ValueError(f"unknown mechanism {m!r} in variant {name!r}")
    for group in (W_MECHS, RES_MECHS, DROP_MECHS):
        assert sum(m in group for m in hs) <= 1, f"at most one of {group} per variant"
    return hs, osb


def _act(z, a):
    return F.relu(z) if a is None else torch.maximum(z, z * a)


def hidden_layer(ly, h):
    z = _act(F.linear(h, ly["W"], ly["b"]), ly.get("a"))
    if "R" in ly:
        z = z + F.linear(h, ly["R"])
    return z


def mlp_forward(layers, x, n_backbone, drops=None, training=False):
    h, rep, L = x, None, len(layers)
    for l, ly in enumerate(layers):
        if l < L - 1:
            h = hidden_layer(ly, h)
            if l == n_backbone - 1:
                rep = h
        else:
            h = F.linear(h, ly["W"], ly["b"])
        if training and drops is not None and drops[l] is not None:
            p = drops[l]
            keep = (torch.rand_like(h) >= p).to(h.dtype)
            h = h * keep / (1 - p)
    return rep, h


def _smooth_field(t, u, g, K=32, f_max=3.0, h_max=30.0):
    f = torch.rand(K, generator=g) * f_max
    h = torch.rand(K, generator=g) * h_max
    phi = torch.rand(K, generator=g) * 2 * math.pi
    arg = 2 * math.pi * (f[:, None, None] * t[None, :, None] + h[:, None, None] * u[None, None, :]) + phi[:, None, None]
    return math.sqrt(2.0 / K) * torch.cos(arg).sum(0)


class SymBreakMLP(nn.Module):
    def __init__(self, in_dim, backbone=(512, 512), head=(512,), out_dim=128, mechs=(), drop_mean=0.1,
                 lr_ratio=4.0, wasym_frac=0.1, wasym_scale=4.0, arch_seed=0, res_lo=0.0, res_hi=2.0,
                 act_max=0.5):
        super().__init__()
        if isinstance(mechs, str):
            mechs = [] if mechs == "none" else mechs.split("+")
        dims = [in_dim, *backbone, *head, out_dim]
        self.dims, self.n_backbone, self.mechs = dims, len(backbone), list(mechs)
        self.L = len(dims) - 1
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        for i in range(self.L):
            lin = nn.Linear(dims[i], dims[i + 1])
            self.weights.append(lin.weight)
            self.biases.append(lin.bias)

        base = 1_000_003 + arch_seed
        gens = {m: torch.Generator().manual_seed(base + off) for m, off in
                [("legacy", 0), ("res", 7919), ("act", 104729), ("band", 1299709)]}
        g = gens["legacy"]
        self.wtype = next((m for m in self.mechs if m in W_MECHS), None)
        self.rtype = next((m for m in self.mechs if m in RES_MECHS), None)
        for l in range(self.L):
            n_out, n_in = dims[l + 1], dims[l]
            hidden = l < self.L - 1
            if "dropuni" in self.mechs:
                self.register_buffer(f"drop{l}", torch.full((n_out,), drop_mean))
            if "dropasym" in self.mechs:
                self.register_buffer(f"drop{l}", torch.linspace(0, 2 * drop_mean, n_out)[torch.randperm(n_out, generator=g)])
            if "lrasym" in self.mechs:
                m = torch.exp(torch.linspace(-math.log(lr_ratio), math.log(lr_ratio), n_out))
                self.register_buffer(f"lrm{l}", m[torch.randperm(n_out, generator=g)])
            n_fix = max(1, int(round(wasym_frac * n_in)))
            if self.wtype == "wasym":
                pos = torch.rand(n_out, n_in, generator=g).argsort(dim=1)[:, :n_fix]
                mask = torch.zeros(n_out, n_in).scatter_(1, pos, 1.0)
                fixed = torch.randn(n_out, n_in, generator=g) * (wasym_scale / math.sqrt(n_fix)) * mask
            elif self.wtype == "wasym_band":
                gb = gens["band"]
                t = (torch.arange(n_out) + 0.5) / n_out
                u = (torch.arange(n_in) + 0.5) / n_in
                centers = torch.floor(t * n_in).long()
                cols = (centers[:, None] + torch.arange(n_fix)[None, :] - n_fix // 2) % n_in
                mask = torch.zeros(n_out, n_in).scatter_(1, cols, 1.0)
                fixed = _smooth_field(t, u, gb, h_max=3.0 / wasym_frac) * (wasym_scale / math.sqrt(n_fix)) * mask
            elif self.wtype == "wfixsym":
                cols = torch.randperm(n_in, generator=g)[:n_fix]
                mask = torch.zeros(n_out, n_in)
                mask[:, cols] = 1.0
                vals = torch.randn(n_fix, generator=g) * (wasym_scale / math.sqrt(n_fix))
                fixed = torch.zeros(n_out, n_in)
                fixed[:, cols] = vals
            if self.wtype is not None:
                self.register_buffer(f"mask{l}", mask)
                self.register_buffer(f"fixed{l}", fixed)
            if self.rtype is not None and hidden and l >= 1 and n_in == n_out:
                if self.rtype == "res_id":
                    R = torch.eye(n_out)
                elif self.rtype == "res_diag":
                    R = torch.diag(torch.linspace(res_lo, res_hi, n_out))
                else:
                    R, _ = torch.linalg.qr(torch.randn(n_out, n_out, generator=gens["res"]))
                self.register_buffer(f"res{l}", R)
            if "act_asym" in self.mechs and hidden:
                self.register_buffer(f"act{l}", torch.linspace(0, act_max, n_out))

    def _buf(self, name):
        return getattr(self, name, None)

    def drops(self):
        return [self._buf(f"drop{l}") for l in range(self.L)]

    def lr_mults(self):
        if "lrasym" not in self.mechs:
            return None
        return [self._buf(f"lrm{l}") for l in range(self.L)]

    def eff_weights(self):
        out = []
        for l in range(self.L):
            W, b = self.weights[l], self.biases[l]
            if self.wtype is not None:
                M = self._buf(f"mask{l}")
                W = W * (1 - M) + self._buf(f"fixed{l}")
            ly = {"W": W, "b": b}
            if self._buf(f"res{l}") is not None:
                ly["R"] = self._buf(f"res{l}")
            if self._buf(f"act{l}") is not None:
                ly["a"] = self._buf(f"act{l}")
            out.append(ly)
        return out

    def forward(self, x):
        return mlp_forward(self.eff_weights(), x, self.n_backbone, self.drops(), self.training)

    def n_trainable(self):
        n = sum(p.numel() for p in self.parameters())
        if self.wtype is not None:
            n -= sum(int(self._buf(f"mask{l}").sum()) for l in range(self.L))
        return n

    def unit_signatures(self):
        sigs = []
        for l in range(self.L - 1):
            s = []
            for name, kind, fn in ((f"drop{l}", "vec", None), (f"lrm{l}", "vec", torch.log), (f"act{l}", "vec", None),
                                   (f"fixed{l}", "rows", None), (f"res{l}", "rows", None)):
                t = self._buf(name)
                if t is not None:
                    t = fn(t) if fn else t
                    if kind == "vec" and float(t.std()) == 0:
                        continue
                    s.append((kind, t))
            sigs.append(s)
        return sigs


VARIANTS = {k: ("+".join(v[0]) or "none", v[1]) for k, v in LEGACY.items()}
VARIANTS.update({m: (m, "none") for m in HIDDEN_MECHS})
