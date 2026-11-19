import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from .models import hidden_layer


def _corr(a, b, eps=1e-6):
    a = (a - a.mean(0)) / (a.std(0) + eps)
    b = (b - b.mean(0)) / (b.std(0) + eps)
    return a.T @ b / a.shape[0]


def _sig_dist(sig, prev_col):
    Ds = []
    for kind, t in sig:
        if kind == "vec":
            D = (t[:, None] - t[None, :]) ** 2
        else:
            tc = t if prev_col is None else t[:, prev_col]
            D = (t * t).sum(1)[:, None] + (tc * tc).sum(1)[None, :] - 2 * t @ tc.T
            D = D.clamp_min(0)
        Ds.append(D / D.mean().clamp_min(1e-12))
    return torch.stack(Ds).mean(0)


@torch.no_grad()
def align_to(layers, ref, x_probe, perm=True, sig=None, restrict_k=None):
    W = [{k: v.clone() for k, v in ly.items()} for ly in layers]
    L = len(W)
    st = {"perm_id_frac": [], "corr_id": [], "corr_matched": [], "loc_ratio": [], "loc_cov8": [], "loc_cov32": [],
          "loc_disp": [], "n_moved": [], "t_match": []}
    h, h_ref = x_probe, x_probe
    prev_col = None
    for l in range(L - 1):
        a = hidden_layer(W[l], h)
        a_ref = hidden_layer(ref[l], h_ref)
        C = _corr(a_ref, a)
        n = C.shape[0]
        D = _sig_dist(sig[l], prev_col) if (sig is not None and len(sig[l]) > 0) else None
        Cn = C.float().cpu().numpy()
        t0 = time.perf_counter()
        if not perm:
            col = np.arange(n)
        elif restrict_k is not None and D is not None:
            nn_idx = D.topk(min(restrict_k, n), dim=1, largest=False).indices.cpu().numpy()
            rows = np.concatenate([np.repeat(np.arange(n), nn_idx.shape[1]), np.arange(n)])
            cols = np.concatenate([nn_idx.ravel(), np.arange(n)])
            G = coo_matrix((Cn[rows, cols] + 2.0, (rows, cols)), shape=(n, n)).tocsr()
            G.sum_duplicates()
            G.data[:] = Cn[G.nonzero()] + 2.0
            _, col = min_weight_full_bipartite_matching(G, maximize=True)
        else:
            _, col = linear_sum_assignment(Cn, maximize=True)
        st["t_match"].append(time.perf_counter() - t0)
        col_t = torch.as_tensor(col, device=C.device)
        active = (a_ref.std(0) > 1e-5).cpu().numpy()
        ident = col == np.arange(n)
        st["perm_id_frac"].append(float(ident[active].mean()) if active.any() else 1.0)
        st["corr_id"].append(float(torch.diagonal(C).mean()))
        st["corr_matched"].append(float(C[torch.arange(n, device=C.device), col_t].mean()))
        moved = (~ident) & active
        st["n_moved"].append(int(moved.sum()))
        if D is not None and moved.any():
            Dn = D.cpu().numpy()
            mv = np.where(moved)[0]
            dm = Dn[mv, col[mv]]
            st["loc_ratio"].append(float(dm.mean() / Dn[mv].mean()))
            rank = (Dn[mv] < dm[:, None]).sum(1)
            st["loc_cov8"].append(float((rank < 8).mean()))
            st["loc_cov32"].append(float((rank < 32).mean()))
            st["loc_disp"].append(float(np.abs(col[mv] - mv).mean() / n))
        ly = W[l]
        for k in ("W", "b", "R", "a"):
            if k in ly:
                ly[k] = ly[k][col_t]
        nx = W[l + 1]
        nx["W"] = nx["W"][:, col_t]
        if "R" in nx:
            nx["R"] = nx["R"][:, col_t]
        h, h_ref = a[:, col_t], a_ref
        prev_col = col_t
    u = F.normalize(F.linear(h, W[-1]["W"], W[-1]["b"]), dim=1)
    u_ref = F.normalize(F.linear(h_ref, ref[-1]["W"], ref[-1]["b"]), dim=1)
    raw = (u - u_ref).pow(2).sum()
    U, _, Vh = torch.linalg.svd(u.T @ u_ref)
    R = U @ Vh
    res = (u @ R - u_ref).pow(2).sum()
    d = R.shape[0]
    st["rot_explained"] = float(1 - res / raw.clamp_min(1e-12))
    st["rot_dist"] = float(torch.linalg.norm(R - torch.eye(d, device=R.device)) / np.sqrt(2 * d))
    st["emb_cos_raw"] = float((u * u_ref).sum(1).mean())
    st["emb_cos_rot"] = float(((u @ R) * u_ref).sum(1).mean())
    return W, R, st


def rotate_output(layers, R):
    W = list(layers)
    last = dict(W[-1])
    last["W"], last["b"] = R.T @ last["W"], R.T @ last["b"]
    W[-1] = last
    return W


def merge(layer_lists, coeffs):
    coeffs = torch.as_tensor(coeffs, dtype=torch.float32)
    coeffs = (coeffs / coeffs.sum()).tolist()
    return [{k: sum(c * wl[l][k] for c, wl in zip(coeffs, layer_lists)) for k in layer_lists[0][l]}
            for l in range(len(layer_lists[0]))]


def aligned_merge(layer_lists, coeffs, x_probe, sig=None, restrict_k=None, rotate=False, refine=1):
    ref = layer_lists[int(np.argmax(coeffs))]
    for _ in range(refine + 1):
        al, stats = [], []
        for w in layer_lists:
            wp, R, st = align_to(w, ref, x_probe, sig=sig, restrict_k=restrict_k)
            al.append(rotate_output(wp, R) if rotate else wp)
            stats.append(st)
        ref = merge(al, coeffs)
    return ref, stats


def detach_weights(layers):
    return [{k: v.detach().clone() for k, v in ly.items()} for ly in layers]
