import torch
import torch.nn.functional as F

from .models import mlp_forward


@torch.no_grad()
def features(weights, X, n_backbone, bs=16384, which="rep"):
    out = []
    for i in range(0, X.shape[0], bs):
        rep, z = mlp_forward(weights, X[i:i + bs], n_backbone)
        out.append(rep if which == "rep" else z)
    return torch.cat(out)


@torch.no_grad()
def knn_acc(ftr, ytr, fte, yte, k=20, T=0.07, C=10, bs=2048):
    ftr, fte = F.normalize(ftr, dim=1), F.normalize(fte, dim=1)
    correct = 0
    for i in range(0, fte.shape[0], bs):
        s = fte[i:i + bs] @ ftr.T
        v, idx = s.topk(k, dim=1)
        votes = torch.zeros(s.shape[0], C, device=s.device)
        votes.scatter_add_(1, ytr[idx], torch.exp(v / T))
        correct += (votes.argmax(1) == yte[i:i + bs]).sum().item()
    return correct / fte.shape[0]


def linear_probe(ftr, ytr, fte, yte, C=10, epochs=60, bs=4096, lr=1e-2, wd=1e-5):
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    clf = torch.nn.Linear(ftr.shape[1], C).to(ftr.device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = ftr.shape[0]
    with torch.enable_grad():
        for _ in range(epochs):
            perm = torch.randperm(n, device=ftr.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                loss = F.cross_entropy(clf(ftr[idx]), ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
    with torch.no_grad():
        return (clf(fte).argmax(1) == yte).float().mean().item()


@torch.no_grad()
def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    hsic = (A.T @ B).pow(2).sum()
    return float(hsic / ((A.T @ A).pow(2).sum().sqrt() * (B.T @ B).pow(2).sum().sqrt() + 1e-12))


@torch.no_grad()
def eval_loss(weights, views, loss_fn, n_backbone):
    tot = 0.0
    for v1, v2 in views:
        _, z1 = mlp_forward(weights, v1, n_backbone)
        _, z2 = mlp_forward(weights, v2, n_backbone)
        tot += loss_fn(z1, z2).item()
    return tot / len(views)
