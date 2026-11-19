import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train import get_args, run_name

DATASETS = ["mnist", "fmnist", "cifar10"]
VARIANTS = ["sym", "dropuni", "dropasym", "lrasym", "wasym", "wasym_weak", "diag", "mrl", "wasym_mrl"]


def v(name):
    toks = name.split("+")
    weak = any(t.endswith("_weak") for t in toks)
    out = {"variant": "+".join(t[:-5] if t.endswith("_weak") else t for t in toks)}
    if weak:
        out["wasym_scale"] = 1
    return out
SEEDS = [0, 1, 2]


def g(**kw):
    return [f"--{k}={v}" for k, v in kw.items()]


def exp_strength():
    for ds, s in itertools.product(["fmnist", "cifar10"], [0, 1]):
        settings = [g(mode="central", rounds=20),
                    g(alpha=0, clients=4, local_epochs=20, rounds=1, client_init="independent")]
        for st in settings:
            yield st + g(dataset=ds, seed=s, variant="sym")
            for sc, fr in itertools.product([0.5, 1, 2, 4], [0.1, 0.25, 0.5]):
                yield st + g(dataset=ds, seed=s, variant="wasym", wasym_scale=sc, wasym_frac=fr)
            for p in [0.1, 0.2, 0.3]:
                yield st + g(dataset=ds, seed=s, variant="dropasym", drop_mean=p)
                yield st + g(dataset=ds, seed=s, variant="dropuni", drop_mean=p)
            for r in [4, 16, 64]:
                yield st + g(dataset=ds, seed=s, variant="lrasym", lr_ratio=r)


def exp_central():
    for ds, vn, s in itertools.product(DATASETS, VARIANTS, SEEDS):
        yield g(dataset=ds, mode="central", rounds=50, seed=s, **v(vn))


def exp_drift():
    for ds, alpha, (E, R), vn, s in itertools.product(DATASETS, [0, 0.1], [(1, 50), (5, 10), (25, 2), (50, 1)],
                                                     VARIANTS, SEEDS):
        yield g(dataset=ds, alpha=alpha, clients=10, local_epochs=E, rounds=R, seed=s, **v(vn))


def exp_oneshot():
    for ds, alpha, vn, s in itertools.product(DATASETS, [0, 0.1], VARIANTS, SEEDS):
        yield g(dataset=ds, alpha=alpha, clients=4, local_epochs=20, rounds=1, client_init="independent",
                seed=s, **v(vn))


def exp_control():
    for ds, alpha, s in itertools.product(DATASETS, [0, 0.1], SEEDS):
        base = g(dataset=ds, alpha=alpha, seed=s, variant="wasym", private_prior=1)
        yield base + g(clients=4, local_epochs=20, rounds=1, client_init="independent")
        for E, R in [(1, 50), (50, 1)]:
            yield base + g(clients=10, local_epochs=E, rounds=R)


SCREEN_VARIANTS = ["sym", "wasym", "wasym_weak", "wasym_band", "wasym_band_weak", "res_id", "res_diag", "res_rand",
                   "act_asym", "dropasym", "lrasym", "wasym+res_rand", "wasym+res_diag", "wasym+act_asym",
                   "wasym_band+res_diag", "wasym_band+act_asym", "res_rand+act_asym", "wasym_weak+res_rand",
                   "wasym+res_diag+act_asym"]


def exp_screen(variants=None):
    for ds, vn, s in itertools.product(["fmnist", "cifar10"], variants or SCREEN_VARIANTS, SEEDS):
        yield g(dataset=ds, mode="central", rounds=50, seed=s, **v(vn))
        for alpha in (0, 0.1):
            yield g(dataset=ds, alpha=alpha, clients=4, local_epochs=20, rounds=1, client_init="independent",
                    seed=s, **v(vn))
        yield g(dataset=ds, alpha=0.1, clients=10, local_epochs=20, rounds=2, seed=s, **v(vn))


COMM_VARIANTS = ["sym", "wasym", "wasym_weak"]


def exp_comm(variants=None):
    for ds, alpha, E, vn, s in itertools.product(["fmnist", "cifar10"], [0, 0.1], [1, 2, 5, 10, 20, 50],
                                                 variants or COMM_VARIANTS, SEEDS):
        yield g(dataset=ds, alpha=alpha, clients=10, local_epochs=E, rounds=10, eval_level="light", seed=s, **v(vn))


COMMAGG = [("sym", "aligned"), ("wasym", "aligned_local"), ("wasym_band", "naive"), ("wasym_band", "aligned_local")]


def exp_commagg(variants=None):
    for ds, alpha, E, (vn, agg), s in itertools.product(["fmnist", "cifar10"], [0, 0.1], [20, 50], COMMAGG, SEEDS):
        if variants and vn not in variants:
            continue
        yield g(dataset=ds, alpha=alpha, clients=10, local_epochs=E, rounds=10, eval_level="light", agg=agg,
                seed=s, **v(vn))


GOSSIP = [("sym", "naive"), ("sym", "aligned"), ("wasym", "naive"), ("wasym+act_asym", "naive"), ("res_id", "naive")]


def exp_gossip(variants=None):
    for ds, init, E, (vn, agg), s in itertools.product(["fmnist", "cifar10"], ["independent", "shared"], [5, 20],
                                                       GOSSIP, SEEDS):
        if variants and vn not in variants:
            continue
        yield g(dataset=ds, alpha=0.1, clients=10, mode="gossip", group_size=2, local_epochs=E, rounds=10,
                client_init=init, agg=agg, eval_level="light", seed=s, **v(vn))


def exp_gossip_size(variants=None):
    for ds, gs, vn, s in itertools.product(["fmnist", "cifar10"], [5, 10], variants or ["sym", "wasym"], SEEDS):
        yield g(dataset=ds, alpha=0.1, clients=10, mode="gossip", group_size=gs, local_epochs=20, rounds=10,
                client_init="independent", eval_level="light", seed=s, **v(vn))


EXPERIMENTS = {k[4:]: v for k, v in globals().items() if k.startswith("exp_")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=list(EXPERIMENTS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gpu", default="1", help="comma-separated GPU ids; --workers per GPU")
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--wandb", type=int, default=1)
    p.add_argument("--group", default=None)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    group = a.group or a.experiment
    jobs = []
    gen = EXPERIMENTS[a.experiment]
    for args in (gen(a.variants) if a.variants else gen()):
        args = args + [f"--group={group}", f"--wandb={a.wandb}"]
        ta = get_args(args)
        name = run_name(ta)
        extra = [x for x in args if x.split("=")[0] in ("--wasym_scale", "--wasym_frac", "--drop_mean", "--lr_ratio")]
        if extra and a.experiment == "strength":
            tag = "_".join(x.lstrip("-").replace("=", "") for x in extra)
            args.append(f"--out={os.path.join(ROOT, 'results', group, tag)}")
            name = f"{tag}/{name}"
        out = os.path.join(ROOT, "results", group, name + ".json")
        if not os.path.exists(out) and not os.path.exists(out + ".lock"):
            jobs.append((name, args, ta.rounds * ta.local_epochs * (ta.clients if ta.mode == "fedavg" else 10)))
    jobs.sort(key=lambda j: -j[2])
    print(f"{a.experiment}: {len(jobs)} runs to do", flush=True)
    if a.dry:
        for n, args, _ in jobs[:10]:
            print(n, " ".join(args))
        return
    log_dir = os.path.join(ROOT, "logs", group)
    gpus = a.gpu.split(",")
    done = [0]
    t0 = time.time()

    def work(job, gpu):
        name, args, _ = job
        out = os.path.join(ROOT, "results", group, name + ".json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            done[0] += 1
            return
        try:
            os.close(os.open(out + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            done[0] += 1
            return
        log = os.path.join(log_dir, name.replace("/", "__") + ".log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "WANDB_SILENT": "true"}
        try:
            for attempt in range(3):
                with open(log, "w") as f:
                    rc = subprocess.call([sys.executable, os.path.join(ROOT, "train.py"), *args], stdout=f,
                                         stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                if rc != -11:
                    break
                time.sleep(20 + 20 * attempt)
        finally:
            os.remove(out + ".lock")
        done[0] += 1
        el = time.time() - t0
        print(f"[{done[0]}/{len(jobs)}] rc={rc} {name} ({el / 60:.1f} min, eta "
              f"{el / done[0] * (len(jobs) - done[0]) / 60:.1f} min)", flush=True)

    pools = [ThreadPoolExecutor(a.workers) for _ in gpus]
    futs = [pools[i % len(gpus)].submit(work, job, gpus[i % len(gpus)]) for i, job in enumerate(jobs)]
    for f in futs:
        f.result()


if __name__ == "__main__":
    main()
