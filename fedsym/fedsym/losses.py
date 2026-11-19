import torch
import torch.nn.functional as F


class InfoNCE:
    def __init__(self, out_dim, tau=0.2, osb="none", diag_range=(0.25, 1.75), mrl_dims=(16, 32, 64, 128),
                 device="cpu"):
        self.tau, self.osb = tau, osb
        self.w = torch.linspace(*diag_range, out_dim, device=device) if osb == "diag" else None
        self.mrl_dims = [m for m in mrl_dims if m <= out_dim] if osb == "mrl" else None
        if osb not in ("none", "diag", "mrl"):
            raise ValueError(osb)

    def _nce(self, z1, z2, w=None):
        n = z1.shape[0]
        u = F.normalize(torch.cat([z1, z2]), dim=1)
        S = ((u * w) if w is not None else u) @ u.T / self.tau
        S = S.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=u.device), float("-inf"))
        t = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(u.device)
        return F.cross_entropy(S, t)

    def __call__(self, z1, z2):
        if self.osb == "mrl":
            return sum(self._nce(z1[:, :m], z2[:, :m]) for m in self.mrl_dims) / len(self.mrl_dims)
        return self._nce(z1, z2, self.w)
