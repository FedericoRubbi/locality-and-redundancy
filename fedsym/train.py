import argparse
import copy
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 64

from fedsym.align import align_to, aligned_merge, detach_weights, merge, rotate_output
from fedsym.data import GPUAugment, load_dataset, normalize_flat, partition
from fedsym.losses import InfoNCE
from fedsym.metrics import eval_loss, features, knn_acc, linear_cka, linear_probe
from fedsym.models import W_MECHS, SymBreakMLP, mlp_forward, parse_variant

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist", "cifar10"])
    p.add_argument("--mode", default="fedavg", choices=["fedavg", "central", "gossip"],
                   help="gossip: no server; every round clients are split into groups of --group_size and each group "
                        "averages (or aligns and averages) its members' models")
    p.add_argument("--variant", default="sym", help="legacy name or '+'-combination, see fedsym/models.py")
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1, help="Dirichlet label skew; <=0 means IID")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--client_init", default="shared", choices=["shared", "independent"],
                   help="independent: clients start from different random inits in round 1 (one-shot merging)")
    p.add_argument("--agg", default="naive", choices=["naive", "aligned", "aligned_local"],
                   help="aligned: Hungarian activation matching to the largest client (+1 refinement), average, "
                        "project back onto the shared prior; aligned_local: same, restricted to restrict_k prior-NN")
    p.add_argument("--restrict_k", type=int, default=8)
    p.add_argument("--group_size", type=int, default=2, help="gossip: clients per merge group (K = FedAvg)")
    p.add_argument("--topology", default="random", choices=["random", "ring"],
                   help="gossip: random disjoint groups every round, or consecutive blocks on a fixed ring whose "
                        "offset shifts by one client every round")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--backbone", type=int, nargs="+", default=[512, 512])
    p.add_argument("--head", type=int, nargs="+", default=[512])
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--drop_mean", type=float, default=0.1)
    p.add_argument("--lr_ratio", type=float, default=4.0)
    p.add_argument("--wasym_frac", type=float, default=0.1)
    p.add_argument("--wasym_scale", type=float, default=4.0)
    p.add_argument("--res_lo", type=float, default=0.0)
    p.add_argument("--res_hi", type=float, default=2.0)
    p.add_argument("--act_max", type=float, default=0.5)
    p.add_argument("--arch_seed", type=int, default=0)
    p.add_argument("--private_prior", type=int, default=0,
                   help="CONTROL: client k uses its own asymmetric prior (arch_seed+k) instead of the shared one; "
                        "the global model keeps client 0's prior")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=0, help="0 = auto (~25 evals per run)")
    p.add_argument("--eval_level", default="full", choices=["full", "light"],
                   help="light: global loss/kNN/linear probe + merge penalty every eval, full diagnostics only at the end")
    p.add_argument("--probe_size", type=int, default=4096, help="shared batch used by the alignment oracle")
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--group", default="debug")
    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--wandb_project", default="fedsym")
    p.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = p.parse_args(argv)
    parse_variant(a.variant)
    return a


def run_name(a):
    alpha = "iid" if a.alpha <= 0 else f"a{a.alpha:g}"
    mechs, _ = parse_variant(a.variant)
    base = f"{a.dataset}-{a.mode}-{a.variant}"
    if any(m in W_MECHS for m in mechs):
        base += f"@s{a.wasym_scale:g}f{a.wasym_frac:g}"
    if "res_diag" in mechs and (a.res_lo, a.res_hi) != (0.0, 2.0):
        base += f"@g{a.res_lo:g}-{a.res_hi:g}"
    if "act_asym" in mechs and a.act_max != 0.5:
        base += f"@a{a.act_max:g}"
    if a.private_prior:
        base += "-private"
    if a.mode in ("fedavg", "gossip"):
        base += f"-{alpha}-K{a.clients}-E{a.local_epochs}-R{a.rounds}-{a.client_init}"
        if a.mode == "gossip":
            base += f"-g{a.group_size}" + ("ring" if a.topology == "ring" else "")
        if a.agg != "naive":
            base += f"-{a.agg}" + (f"{a.restrict_k}" if a.agg == "aligned_local" else "")
    else:
        base += f"-ep{a.rounds * a.local_epochs}"
    return base + f"-s{a.seed}"


def build_model(a, d, init_seed, mechs=None):
    torch.manual_seed(init_seed)
    mechs = parse_variant(a.variant)[0] if mechs is None else mechs
    return SymBreakMLP(d["in_dim"], a.backbone, a.head, a.out_dim, mechs, a.drop_mean, a.lr_ratio, a.wasym_frac,
                       a.wasym_scale, a.arch_seed, a.res_lo, a.res_hi, a.act_max).cuda()


def make_loss_step(model, aug, loss_fn, compile_):
    def f(xb):
        _, z = model(torch.cat([aug(xb), aug(xb)]))
        return loss_fn(*z.chunk(2))
    return torch.compile(f, dynamic=False) if compile_ else f


def local_train(model, loss_step, X, epochs, a, opt=None):
    model.train()
    opt = opt or torch.optim.Adam(model.parameters(), lr=a.lr)
    lrm = model.lr_mults()
    tot, steps, n = torch.zeros((), device=X.device), 0, X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=X.device)
        bs = min(a.batch_size, n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            loss = loss_step(X[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if lrm is not None:
                old = [(W.detach().clone(), b.detach().clone()) for W, b in zip(model.weights, model.biases)]
            opt.step()
            if lrm is not None:
                with torch.no_grad():
                    for (W, b), (W0, b0), m in zip(zip(model.weights, model.biases), old, lrm):
                        W.copy_(W0 + (W - W0) * m[:, None])
                        b.copy_(b0 + (b - b0) * m)
            tot += loss.detach()
            steps += 1
    return (tot / max(steps, 1)).item(), opt


class Evaluator:
    def __init__(self, d, a, loss_fn, aug, n_backbone):
        self.d, self.a, self.loss_fn, self.nb = d, a, loss_fn, n_backbone
        self.Xtr, self.Xte = normalize_flat(d["xtrain"], d), normalize_flat(d["xtest"], d)
        self.ytr, self.yte = d["ytrain"], d["ytest"]
        g = torch.Generator(device="cuda").manual_seed(12345)
        idx = torch.randperm(self.Xte.shape[0], device="cuda", generator=g)[:4096]
        xs = d["xtest"][idx]
        self.views = [(aug(xs[i:i + 1024], gen=g), aug(xs[i:i + 1024], gen=g)) for i in range(0, 4096, 1024)]
        pidx = torch.randperm(self.Xtr.shape[0], device="cuda", generator=g)[:a.probe_size]
        self.probe = self.Xtr[pidx]

    def knn(self, w):
        return knn_acc(features(w, self.Xtr, self.nb), self.ytr, features(w, self.Xte, self.nb), self.yte)

    def lin(self, w):
        return linear_probe(features(w, self.Xtr, self.nb), self.ytr, features(w, self.Xte, self.nb), self.yte)

    def loss(self, w):
        return eval_loss(w, self.views, self.loss_fn, self.nb)

    def rep(self, w, X=None):
        return mlp_forward(w, self.probe if X is None else X, self.nb)[0]


def flat_params(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def wmean(xs, ws):
    return float(np.average(xs, weights=ws))


def _mean_stats(stats, m, prefix, keys):
    for k in keys:
        vals = [s[k] for s in stats if len(s.get(k, [])) > 0]
        if not vals:
            continue
        per_layer = [float(np.mean([v[l] for v in vals if l < len(v)])) for l in range(max(len(v) for v in vals))]
        m[f"{prefix}/{k}"] = float(np.mean(per_layer))
        for l, v in enumerate(per_layer):
            m[f"{prefix}_layer/{k}_L{l}"] = v


def evaluate_round(ev, a, gw, cws, ref, coeffs, final, sig, light=False):
    m = {"global/knn": ev.knn(gw), "global/loss": ev.loss(gw)}
    if light or final:
        m["global/linprobe"] = ev.lin(gw)
    if cws is None:
        return m
    cl_loss = [ev.loss(w) for w in cws]
    m["clients/loss"] = wmean(cl_loss, coeffs)
    m["merge/penalty_loss"] = m["global/loss"] - m["clients/loss"]
    if light and not final:
        return m
    cl_knn = [ev.knn(w) for w in cws]
    m["clients/knn"] = wmean(cl_knn, coeffs)
    m["merge/penalty_knn"] = m["clients/knn"] - m["global/knn"]
    naive = merge(cws, coeffs)
    m["merge/naive_loss"] = ev.loss(naive)

    permd, rotd, stats = [], [], []
    for w in cws:
        wp, R, st = align_to(w, ref, ev.probe, sig=sig)
        permd.append(wp)
        rotd.append(rotate_output(wp, R))
        stats.append(st)
    mp, mf = merge(permd, coeffs), merge(rotd, coeffs)
    m["merge/aligned_perm_loss"], m["merge/aligned_full_loss"] = ev.loss(mp), ev.loss(mf)
    m["merge/aligned_perm_knn"], m["merge/aligned_full_knn"] = ev.knn(mp), ev.knn(mf)
    if ref is cws[0]:
        m0, stats0 = mf, stats
    else:
        al0 = [align_to(w, cws[0], ev.probe, sig=sig) for w in cws]
        m0, stats0 = merge([rotate_output(wp, R) for wp, R, _ in al0], coeffs), [s for _, _, s in al0]
    m["merge/aligned_c0_loss"], m["merge/aligned_c0_knn"] = ev.loss(m0), ev.knn(m0)
    agg_full, _ = aligned_merge(cws, coeffs, ev.probe, sig=sig)
    m["merge/agg_full_loss"] = ev.loss(agg_full)
    for k in (8, 32):
        agg_k, st_k = aligned_merge(cws, coeffs, ev.probe, sig=sig, restrict_k=k)
        m[f"merge/agg_k{k}_loss"] = ev.loss(agg_k)
        m[f"time/match_k{k}_ms"] = 1e3 * float(np.mean([np.mean(s["t_match"]) for s in st_k]))
    _, st_full = aligned_merge(cws, coeffs, ev.probe, sig=sig, refine=0)
    m["time/match_full_ms"] = 1e3 * float(np.mean([np.mean(s["t_match"]) for s in st_full]))

    gl = m["merge/naive_loss"]
    m["sym/gap_loss"] = gl - m["merge/aligned_full_loss"]
    m["sym/gap_c0_loss"] = gl - m["merge/aligned_c0_loss"]
    m["sym/gap_best_loss"] = gl - min(m["merge/aligned_full_loss"], m["merge/aligned_c0_loss"])
    m["sym/gap_perm_loss"] = gl - m["merge/aligned_perm_loss"]
    m["sym/gap_rot_loss"] = m["merge/aligned_perm_loss"] - m["merge/aligned_full_loss"]
    m["sym/gap_knn"] = m["merge/aligned_full_knn"] - m["global/knn"]
    m["sym/gap_agg_full_loss"] = gl - m["merge/agg_full_loss"]
    for k in (8, 32):
        m[f"sym/gap_agg_k{k}_loss"] = gl - m[f"merge/agg_k{k}_loss"]
    if ref is cws[0]:
        stats = stats[1:]
    stats0 = stats0[1:]
    _mean_stats(stats, m, "drift", ["perm_id_frac", "corr_id", "corr_matched"])
    _mean_stats(stats0, m, "loc", ["perm_id_frac", "loc_ratio", "loc_cov8", "loc_cov32", "loc_disp", "n_moved"])
    for k in ("rot_explained", "rot_dist", "emb_cos_raw", "emb_cos_rot"):
        m[f"drift/{k}"] = float(np.mean([s[k] for s in stats]))
    ref_rep = ev.rep(ref)
    others = [w for w in cws if w is not ref]
    m["drift/rep_cka"] = float(np.mean([linear_cka(ev.rep(w), ref_rep) for w in others]))
    m["drift/rep_cos_raw"] = float(np.mean([F.cosine_similarity(ev.rep(w), ref_rep, dim=1).mean().item()
                                            for w in others]))
    if final:
        m["merge/aligned_full_linprobe"] = ev.lin(mf)
        m["merge/agg_full_linprobe"] = ev.lin(agg_full)
        m["clients/linprobe_c0"] = ev.lin(cws[0])
        alphas = np.linspace(0, 1, 11)
        w1_al = rotate_output(*align_to(cws[1], cws[0], ev.probe)[:2])
        for tag, w1 in (("naive", cws[1]), ("aligned", w1_al)):
            ls = np.array([ev.loss(merge([cws[0], w1], [1 - t, t])) for t in alphas])
            m[f"barrier/{tag}_loss"] = float(np.max(ls - ((1 - alphas) * ls[0] + alphas * ls[-1])))
            m[f"barrier/{tag}_mid_knn"] = ev.knn(merge([cws[0], w1], [0.5, 0.5]))
    return m


def gossip_groups(K, g, r, topology, rng):
    order = rng.permutation(K) if topology == "random" else np.roll(np.arange(K), -r)
    return [sorted(int(k) for k in order[i:i + g]) for i in range(0, K, g)]


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def load_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(state[n])


def gossip_loop(a, d, ev, gmodel, worker, loss_step, Xc, sizes, sig, run, name, eval_every):
    K = len(Xc)
    rng = np.random.default_rng(10_007 + a.seed)
    if a.client_init == "independent":
        states = [snapshot(build_model(a, d, init_seed=100_000 * (a.seed + 1) + k)) for k in range(K)]
    else:
        states = [snapshot(gmodel) for _ in range(K)]
    history = []
    for r in range(a.rounds):
        t0 = time.time()
        snaps, cws, losses = [], [], []
        for k in range(K):
            load_state(worker, states[k])
            lk, _ = local_train(worker, loss_step, Xc[k], a.local_epochs, a)
            snaps.append(snapshot(worker))
            cws.append(detach_weights(worker.eff_weights()))
            losses.append(lk)

        groups = gossip_groups(K, a.group_size, r, a.topology, rng)
        merged_ws = []
        for G in groups:
            gs = [sizes[k] for k in G]
            if len(G) == 1:
                new = snaps[G[0]]
            elif a.agg == "naive":
                c = [s / sum(gs) for s in gs]
                new = {n: sum(ci * snaps[k][n] for ci, k in zip(c, G)) for n in snaps[G[0]]}
            else:
                mw, _ = aligned_merge([cws[k] for k in G], gs, ev.probe, sig=sig,
                                      restrict_k=a.restrict_k if a.agg == "aligned_local" else None)
                load_state(worker, snaps[G[int(np.argmax(gs))]])
                with torch.no_grad():
                    for l, ly in enumerate(mw):
                        worker.weights[l].copy_(ly["W"])
                        worker.biases[l].copy_(ly["b"])
                new = snapshot(worker)
            load_state(worker, new)
            merged_ws.append(detach_weights(worker.eff_weights()))
            for k in G:
                states[k] = new
        t_train = time.time() - t0

        final = r == a.rounds - 1
        if r == 0 or final or (r + 1) % eval_every == 0:
            m = evaluate_round(ev, a, merge(cws, sizes), cws, cws[0], sizes, final, sig,
                               light=a.eval_level == "light")
            cl_loss = [ev.loss(w) for w in cws]
            pen, post_loss, post_knn, post_w = [], [], [], []
            for G, mw in zip(groups, merged_ws):
                gs = [sizes[k] for k in G]
                lm = ev.loss(mw)
                if len(G) > 1:
                    pen.append(lm - wmean([cl_loss[k] for k in G], gs))
                post_loss.append(lm)
                post_knn.append(ev.knn(mw))
                post_w.append(sum(gs))
            m["gossip/penalty_loss"] = float(np.mean(pen)) if pen else 0.0
            m["gossip/post_loss"] = wmean(post_loss, post_w)
            m["gossip/post_knn"] = wmean(post_knn, post_w)
            flats = [torch.cat([v.flatten() for v in s.values()]) for s in states]
            mean_flat = torch.stack(flats).mean(0)
            m["gossip/consensus_dist"] = float(np.mean([((f - mean_flat).norm() / mean_flat.norm()).item()
                                                        for f in flats]))
            if final:
                m["gossip/post_linprobe"] = wmean([ev.lin(mw) for mw in merged_ws], post_w)
            m.update({"round": r + 1, "epochs": (r + 1) * a.local_epochs, "train/loss": wmean(losses, sizes),
                      "time/train_s": t_train, "time/round_s": time.time() - t0})
            history.append(m)
            if run:
                run.log(m, step=r + 1)
            msg = " ".join(f"{k.split('/', 1)[-1]}={v:.4f}" for k, v in m.items()
                           if k in ("gossip/post_knn", "gossip/penalty_loss", "gossip/consensus_dist",
                                    "global/knn", "global/linprobe", "gossip/post_linprobe"))
            print(f"[{name}] r{r + 1} {msg} ({time.time() - t0:.1f}s)", flush=True)
    return history


def finish(a, run, history, t_start, out_path, info, name):
    summary = {f"final/{k}": v for k, v in history[-1].items()}
    summary["final/best_knn"] = max(h["global/knn"] for h in history)
    if a.mode == "gossip":
        summary["final/best_post_knn"] = max(h["gossip/post_knn"] for h in history)
    summary["time/total_s"] = time.time() - t_start
    if run:
        run.summary.update(summary)
        run.finish()
    with open(out_path, "w") as f:
        json.dump({"config": vars(a), "info": info, "history": history, "summary": summary}, f)
    print(f"[{name}] done in {summary['time/total_s']:.0f}s -> {out_path}", flush=True)


def main(argv=None):
    a = get_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    name = run_name(a)
    out_path = os.path.join(a.out, a.group, name + ".json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    d = load_dataset(a.dataset, "cuda")
    aug = GPUAugment(d)
    mechs, osb = parse_variant(a.variant)
    loss_fn = InfoNCE(a.out_dim, a.tau, osb, device="cuda")
    central = a.mode == "central"
    parts = [torch.arange(d["ytrain"].numel())] if central else partition(d["ytrain"], a.clients, a.alpha, a.seed)
    K = len(parts)
    sizes = [len(p) for p in parts]
    Xc = [d["xtrain"][p.cuda()] for p in parts]

    gmodel = build_model(a, d, init_seed=a.seed)
    ev = Evaluator(d, a, loss_fn, aug, gmodel.n_backbone)
    eval_every = a.eval_every or max(1, a.rounds // 25)
    sig = gmodel.unit_signatures()
    sig_null = all(len(s) == 0 for s in sig) or all(m == "res_id" for m in mechs)
    if sig_null:
        sig = build_model(a, d, init_seed=0, mechs=["wasym"]).unit_signatures()

    label_hist = [torch.bincount(d["ytrain"][p.cuda()], minlength=10).tolist() for p in parts]
    info = {"n_trainable": gmodel.n_trainable(), "client_sizes": sizes, "label_hist": label_hist,
            "hsb": "+".join(mechs) or "none", "osb": osb, "name": name, "sig_null": sig_null}
    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project=a.wandb_project, group=a.group, name=name, config={**vars(a), **info},
                         dir=ROOT, reinit=True, tags=[a.dataset, a.mode, a.variant, a.client_init, a.agg])
    print(f"[{name}] trainable params {info['n_trainable']}, client sizes {sizes}", flush=True)

    w0 = detach_weights(gmodel.eff_weights())
    info["init_knn"], info["init_linprobe"], info["init_loss"] = ev.knn(w0), ev.lin(w0), ev.loss(w0)
    print(f"[{name}] init knn={info['init_knn']:.4f} linprobe={info['init_linprobe']:.4f}", flush=True)
    if run:
        run.summary.update({f"init/{k[5:]}": info[k] for k in ("init_knn", "init_linprobe", "init_loss")})

    worker = gmodel if central else build_model(a, d, init_seed=a.seed)
    loss_step = make_loss_step(worker, aug, loss_fn, a.compile)
    priors = None
    if a.private_prior and not central:
        priors = []
        for k in range(K):
            pa = copy.copy(a)
            pa.arch_seed = a.arch_seed + k
            priors.append({n: b.clone() for n, b in build_model(pa, d, init_seed=0).named_buffers()})
    if a.mode == "gossip":
        assert not a.private_prior, "--private_prior is not supported in gossip mode"
        t_start = time.time()
        history = gossip_loop(a, d, ev, gmodel, worker, loss_step, Xc, sizes, sig, run, name, eval_every)
        return finish(a, run, history, t_start, out_path, info, name)
    history, central_opt = [], None
    t_start = time.time()
    for r in range(a.rounds):
        t0 = time.time()
        start_w = detach_weights(gmodel.eff_weights())
        start_flat = flat_params(gmodel)
        snaps, cws, losses = [], [], []
        for k in range(K):
            if not central:
                src = gmodel
                if a.client_init == "independent" and r == 0:
                    src = build_model(a, d, init_seed=100_000 * (a.seed + 1) + k)
                with torch.no_grad():
                    for p, q in zip(worker.parameters(), src.parameters()):
                        p.copy_(q)
            if priors is not None:
                with torch.no_grad():
                    for n, b in worker.named_buffers():
                        b.copy_(priors[k][n])
            lk, opt = local_train(worker, loss_step, Xc[k], a.local_epochs, a, central_opt if central else None)
            if central:
                central_opt = opt
            else:
                snaps.append({n: p.detach().clone() for n, p in worker.named_parameters()})
                cws.append(detach_weights(worker.eff_weights()))
            losses.append(lk)

        ref = None
        if central:
            cws = None
        else:
            ref = cws[0] if (a.client_init == "independent" and r == 0) else start_w
            coeffs = torch.tensor(sizes, dtype=torch.float32) / sum(sizes)
            with torch.no_grad():
                if a.agg == "naive":
                    for name_p, p in gmodel.named_parameters():
                        p.copy_(sum(c * sn[name_p] for c, sn in zip(coeffs.tolist(), snaps)))
                else:
                    merged, _ = aligned_merge(cws, sizes, ev.probe, sig=sig,
                                              restrict_k=a.restrict_k if a.agg == "aligned_local" else None)
                    for l, ly in enumerate(merged):
                        gmodel.weights[l].copy_(ly["W"])
                        gmodel.biases[l].copy_(ly["b"])
        t_train = time.time() - t0

        final = r == a.rounds - 1
        if r == 0 or final or (r + 1) % eval_every == 0:
            gw = detach_weights(gmodel.eff_weights())
            m = evaluate_round(ev, a, gw, cws, ref, sizes, final, sig, light=a.eval_level == "light")
            if not central:
                deltas = [torch.cat([v.flatten() for v in sn.values()]) - start_flat for sn in snaps]
                cos = [F.cosine_similarity(x, y, dim=0).item() for x, y in itertools.combinations(deltas, 2)]
                m["drift/update_cos"] = float(np.mean(cos))
                m["drift/update_norm"] = float(np.mean([x.norm().item() for x in deltas]))
            m.update({"round": r + 1, "epochs": (r + 1) * a.local_epochs, "train/loss": wmean(losses, sizes),
                      "time/train_s": t_train, "time/round_s": time.time() - t0})
            history.append(m)
            if run:
                run.log(m, step=r + 1)
            msg = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in m.items()
                           if k in ("global/knn", "global/loss", "global/linprobe", "merge/penalty_loss",
                                    "sym/gap_loss", "loc/perm_id_frac", "loc/loc_ratio", "loc/loc_cov8"))
            print(f"[{name}] r{r + 1} {msg} ({time.time() - t0:.1f}s)", flush=True)

    finish(a, run, history, t_start, out_path, info, name)


if __name__ == "__main__":
    main()
