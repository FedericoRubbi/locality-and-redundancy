import argparse
import hashlib
import json
import math
import platform
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import bp_mlp as bp
import itmf_mlp as itmf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper" / "data" / "raw" / "itmf_rerun"
OPTUNA_DIR = HERE / "optuna_studies"

DATASETS = ["MNIST", "F-MNIST", "CIFAR10", "CIFAR100"]
SEEDS = [0, 1, 2]
EPOCHS = 25
NUM_BLOCKS = 8
HIDDEN = 3072
VAL_SIZE = 5000
SPLIT_SEED = 12345
RECORDED = {
    "CIFAR10": ("srdan_v6_cifar10_optuna.db", 152),
    "CIFAR100": ("srdan_v6_cifar100_optuna.db", 94),
}


def load_split(name, data_dir, train):
    if name == "MNIST":
        ds = torchvision.datasets.MNIST(data_dir, train=train, download=True)
    elif name == "F-MNIST":
        ds = torchvision.datasets.FashionMNIST(data_dir, train=train, download=True)
    elif name == "CIFAR10":
        ds = torchvision.datasets.CIFAR10(data_dir, train=train, download=True)
    else:
        ds = torchvision.datasets.CIFAR100(data_dir, train=train, download=True)
    if name in ("MNIST", "F-MNIST"):
        x = ds.data.float().div(255.0)
    else:
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div(255.0)
    x = x.sub(0.5).div(0.5).reshape(len(x), -1)
    y = torch.as_tensor(ds.targets, dtype=torch.long)
    return x, y


class GPULoader:

    def __init__(self, x, y, batch_size, shuffle, drop_last, seed=0):
        self.x, self.y, self.bs, self.shuffle, self.drop_last = x, y, batch_size, shuffle, drop_last
        self.gen = torch.Generator(device=x.device)
        self.gen.manual_seed(seed)

    def __len__(self):
        n = len(self.x)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        n = len(self.x)
        idx = (torch.randperm(n, device=self.x.device, generator=self.gen) if self.shuffle
               else torch.arange(n, device=self.x.device))
        for b in range(len(self)):
            j = idx[b * self.bs:(b + 1) * self.bs]
            yield self.x[j], self.y[j]


def get_data(name, data_dir, device, phase):
    x, y = load_split(name, data_dir, train=True)
    if phase == "search":
        perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(SPLIT_SEED))
        tr, ev = perm[:-VAL_SIZE], perm[-VAL_SIZE:]
        x_tr, y_tr, x_ev, y_ev = x[tr], y[tr], x[ev], y[ev]
    else:
        x_tr, y_tr = x, y
        x_ev, y_ev = load_split(name, data_dir, train=False)
    num_classes = 100 if name == "CIFAR100" else 10
    return (x_tr.to(device), y_tr.to(device), x_ev.to(device), y_ev.to(device), x.shape[1], num_classes)


def sample_config(method, rng):
    def loguniform(lo, hi):
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))

    if method == "bp":
        return {"lr": loguniform(1e-4, 1e-2), "weight_decay": loguniform(1e-6, 1e-3),
                "batch_size": rng.choice([128, 256, 512]), "dropout": rng.uniform(0.0, 0.5)}
    return {
        "lr": loguniform(1e-4, 1e-2), "gamma_rec": rng.uniform(0.0, 10.0), "alpha_max": rng.uniform(1.0, 20.0),
        "schedule_type": rng.choice(["cosine", "quadratic", "exponential"]),
        "weight_decay": loguniform(1e-6, 1e-3), "batch_size": rng.choice([128, 256, 512, 1024]),
        "dropout": rng.uniform(0.0, 0.5), "activation": rng.choice(["relu", "gelu", "silu"]),
        "tie_weights": rng.choice([True, False]),
    }


def recorded_config(dataset):
    db, number = RECORDED[dataset]
    con = sqlite3.connect(OPTUNA_DIR / db)
    tid, value = con.execute("""select t.trial_id, v.value from trials t join trial_values v on t.trial_id = v.trial_id
                                where t.number = ?""", (number,)).fetchone()
    cfg = {}
    for name, val, dist in con.execute("select param_name, param_value, distribution_json from trial_params "
                                       "where trial_id = ?", (tid,)):
        dist = json.loads(dist)
        cfg[name] = dist["attributes"]["choices"][int(val)] if dist["name"] == "CategoricalDistribution" else val
    cfg.pop("hidden_dim", None)
    return cfg, {"db": db, "trial": number, "recorded_test_acc": value}


def search_jobs(trials):
    jobs = []
    for method in ("bp", "itmf"):
        for dataset in DATASETS:
            if method == "itmf" and dataset in RECORDED:
                continue
            for t in range(trials):
                jobs.append({"phase": "search", "method": method, "dataset": dataset, "trial": t,
                             "file": OUT / "search" / f"{method}_{dataset}_t{t:02d}.json"})
    return jobs


def final_jobs():
    return [{"phase": "final", "method": m, "dataset": d, "seed": s,
             "file": OUT / "final" / f"{m}_{d}_s{s}.json"}
            for m in ("itmf", "bp") for d in DATASETS for s in SEEDS]


def config_for(job, trials):
    method, dataset = job["method"], job["dataset"]
    if job["phase"] == "search":
        rng = random.Random(f"{method}/{dataset}/{job['trial']}")
        return sample_config(method, rng), {"source": "random search", "trial": job["trial"]}
    if method == "itmf" and dataset in RECORDED:
        cfg, info = recorded_config(dataset)
        return cfg, {"source": "Optuna study", **info}
    runs = [json.loads(f.read_text()) for f in sorted((OUT / "search").glob(f"{method}_{dataset}_t*.json"))]
    if len(runs) < trials:
        raise RuntimeError(f"search for {method}/{dataset} incomplete: {len(runs)}/{trials}")
    best = max(runs, key=lambda r: r["final_acc"])
    return best["config"], {"source": "random search, selected on validation accuracy", "trial": best["trial"],
                            "val_acc": best["final_acc"], "n_trials": len(runs)}


def run(job, data, device, trials):
    cfg, provenance = config_for(job, trials)
    seed = job.get("seed", 1000 + job.get("trial", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    x_tr, y_tr, x_ev, y_ev, input_dim, num_classes = data
    bs = int(cfg["batch_size"])
    train_loader = GPULoader(x_tr, y_tr, bs, shuffle=True, drop_last=True, seed=seed)
    eval_loader = GPULoader(x_ev, y_ev, bs, shuffle=False, drop_last=False)
    history = {"acc": [], "layer_acc": []}
    t0 = time.time()

    if job["method"] == "itmf":
        model = itmf.LocallyDrivenNetwork(input_dim, HIDDEN, num_classes, NUM_BLOCKS, tie_weights=cfg["tie_weights"],
                                           activation=cfg["activation"], dropout=cfg["dropout"]).to(device)
        trainer = itmf.LocallyDrivenTrainer(model, train_loader, eval_loader, device, num_classes=num_classes,
                                             lr=cfg["lr"], gamma_rec=cfg["gamma_rec"], alpha_max=cfg["alpha_max"],
                                             schedule_type=cfg["schedule_type"], weight_decay=cfg["weight_decay"],
                                             use_scheduler=True, epochs=EPOCHS)
        for _ in range(EPOCHS):
            trainer.train_epoch()
            _, acc, _, layer_acc = trainer.evaluate()
            history["acc"].append(float(acc))
            history["layer_acc"].append([float(a) for a in layer_acc])
    else:
        model = bp.BaselineMLP(input_dim, HIDDEN, num_classes, NUM_BLOCKS, cfg["dropout"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()
        for _ in range(EPOCHS):
            bp.train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            _, layer_acc = bp.evaluate(model, eval_loader, criterion, device)
            history["acc"].append(float(layer_acc[-1]))
            history["layer_acc"].append([float(a) for a in layer_acc])

    return {
        "method": job["method"], "dataset": job["dataset"], "phase": job["phase"],
        "eval_split": "validation (5k held out from train)" if job["phase"] == "search" else "test",
        "trial": job.get("trial"), "seed": seed, "config": cfg, "provenance": provenance,
        "epochs": EPOCHS, "num_blocks": NUM_BLOCKS, "hidden": HIDDEN,
        "final_acc": history["acc"][-1], "history": history, "train_seconds": round(time.time() - t0, 1),
    }


def environment():
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
    return {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
            "itmf_sha": sha(HERE / "itmf_mlp.py"), "bp_sha": sha(HERE / "bp_mlp.py"), "this_script_sha": sha(__file__)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["search", "final"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--only", default=None, help="restrict to method/dataset, e.g. itmf/CIFAR10 (smoke tests)")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only; output not saved)")
    args = ap.parse_args()

    global EPOCHS
    smoke = args.epochs is not None
    if smoke:
        EPOCHS = args.epochs
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    env = environment()

    jobs = search_jobs(args.trials) if args.phase == "search" else final_jobs()
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    if args.only:
        m, d = args.only.split("/")
        jobs = [j for j in jobs if j["method"] == m and j["dataset"] == d][:1]
    cache = {}
    for job in jobs:
        if job["file"].exists() and not smoke:
            continue
        key = (job["dataset"], job["phase"])
        if key not in cache:
            cache.clear()
            cache[key] = get_data(job["dataset"], args.data_dir, device, job["phase"])
        result = run(job, cache[key], device, args.trials)
        result["environment"] = env
        tag = f"{job['method']}/{job['dataset']}/{job['phase']} " + \
              (f"t{job['trial']}" if job["phase"] == "search" else f"s{job['seed']}")
        print(f"{tag}: acc {result['final_acc']:.2f} in {result['train_seconds']} s", flush=True)
        if smoke:
            continue
        job["file"].parent.mkdir(parents=True, exist_ok=True)
        tmp = job["file"].with_suffix(".tmp")
        tmp.write_text(json.dumps(result))
        tmp.replace(job["file"])


if __name__ == "__main__":
    main()
