import json

import matplotlib.pyplot as plt

from style import FULL, HALF, MUTED, load, save, select, series_style, write_table

SRC = "figures_grim.py"
ABLATION = [
    ("plain", "Plain InfoNCE"),
    ("additive_rde", "Additive RDE (variance-compensated)"),
    ("multiplicative_rde", "Multiplicative log-normal RDE"),
    ("cumulative_multiplicative_rde", "Cumulative multiplicative RDE"),
    ("explosive_ar1", "Explosive AR(1) noise"),
    ("heteroscedastic_random_walk", "Heteroscedastic random walk"),
    ("nested_dropout", "Nested dropout"),
    ("independent_nested_dropout", "Independent nested dropout"),
    ("matryoshka", "Matryoshka"),
]
SRR = [
    ("v0", "Loss at every step", "every step"),
    ("v1", "\\quad + EMA target", "+ EMA target"),
    ("v2", "\\quad + causal PCGrad", "+ PCGrad"),
    ("v3", "\\quad + EMA + PCGrad", "+ EMA + PCGrad"),
    ("temp_anneal", "\\quad + temperature schedule", "+ temp. schedule"),
    ("random", "Loss at a random step", "random step"),
    ("e2_dense", "Dense local InfoNCE", "dense local"),
    ("d1", "One block per step", "1 block / step"),
    ("lastonly", "Loss at the last step only", "last step only"),
    ("t1_shallow3", "No recurrence, 3 blocks", "no recurrence, 3 blocks"),
    ("t1_deep24", "No recurrence, 24 blocks", "no recurrence, 24 blocks"),
]
SLOT = {"v0": 0, "lastonly": 1, "random": 2, "d1": 3, "e2_dense": 4, "v1": 5, "v2": 6}
LABEL = {run: fig for run, _, fig in SRR}


def f2(x, digits=1, scale=1.0):
    return "---" if x in ("", None) else f"${float(x) * scale:.{digits}f}$"


def ablation_table():
    rows = {r["method"]: r for r in load("grim_ablation.csv")}
    body = []
    for m, label in ABLATION:
        r = rows[m]
        body.append([label, f2(r["lp_acc_A/1. (1/1) 128d"], 1, 100), f2(r["lp_acc_A/3. (1/4) 32d"], 1, 100),
                     f2(r["lp_acc_A/5. (1/16) 8d"], 1, 100), f2(r["alignment/1. (1/1) 128d"], 2),
                     f2(r["alignment/5. (1/16) 8d"], 2), f2(r["effective_rank/mean"], 1)])
    header = [["", r"\multicolumn{3}{c}{Linear probe (\%), first $k$ dims}",
               r"\multicolumn{2}{c}{Cross-model distance}", ""],
              [r"\cmidrule(lr){2-4}\cmidrule(lr){5-6} Regime", "$k=128$", "$k=32$", "$k=8$", "$k=128$", "$k=8$",
               "Eff.\\ rank"]]
    write_table("grim_ablation", "@{}lcccccc@{}", header, body, SRC, best={1: "max", 2: "max", 3: "max"})


def panel():
    fig, ax = plt.subplots(figsize=(HALF, 2.7))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.42)
    return fig, ax


def save_panel(fig, name):
    with plt.rc_context({"savefig.bbox": "standard"}):
        save(fig, name)


def adapters_figure():
    rows = load("grim_adapter_depth.csv")
    fig, ax = panel()
    for slot, (name, label) in enumerate([("lightconv", "light conv adapter"), ("resnet50", "ResNet-50 adapter")]):
        rs = sorted(select(rows, adapter=name), key=lambda r: int(r["t"]))
        t = [int(r["t"]) for r in rs]
        st = series_style(slot)
        ax.plot(t, [float(r["probe_acc"]) for r in rs], label=f"{label}, probe at $t$", **st)
        ax.plot(t, [float(r["elastic_acc"]) for r in rs], label=f"{label}, elastic", linestyle="--",
                markerfacecolor="white", **st)
    ax.set_xlabel("recurrent step $t$")
    ax.set_ylabel("Tiny-ImageNet probe (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=6)
    save_panel(fig, "grim_adapters")


def srr_table():
    runs = {r["run"]: r for r in load("grim_srr_runs.csv")}
    depth = load("grim_srr_depth.csv")
    body = []
    for run, label, _ in SRR:
        lin = {int(r["t"]): float(r["accuracy"]) for r in select(depth, run=run, readout="f", probe="linear")}
        el = {int(r["t"]): float(r["accuracy"]) for r in select(depth, run=run, readout="f", probe="elastic")}
        last = max(lin)
        r = runs[run]
        body.append([label, f2(r["params_m"], 1), f2(lin[0]), f2(lin[last]), f2(max(lin.values())),
                     f2(r["final/best_accuracy_h"], 1, 100), f2(el.get(4)) if last > 1 else "---"])
        if run in ("temp_anneal", "d1"):
            body.append("midrule")
    header = [["", "", r"\multicolumn{4}{c}{Linear probe (\%)}", ""],
              [r"\cmidrule(lr){3-6} Variant", "Params (M)", "Stem", "Deepest", "Best", r"Best ($h$)",
               r"Elastic $t{=}4$"]]
    write_table("grim_srr", "@{}lcccccc@{}", header, body, SRC, best={2: "max", 3: "max", 4: "max", 5: "max", 6: "max"})


def srr_depth_figure():
    depth = load("grim_srr_depth.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    for ax, probe, title in ((axes[0], "linear", "(a) linear probe trained at each depth"),
                             (axes[1], "elastic", "(b) elastic: head trained at $t=8$")):
        for run in ("v0", "random", "d1", "e2_dense", "lastonly"):
            rs = sorted(select(depth, run=run, readout="f", probe=probe), key=lambda r: int(r["t"]))
            ax.plot([int(r["t"]) for r in rs], [float(r["accuracy"]) for r in rs], label=LABEL[run],
                    **series_style(SLOT[run]))
        if probe == "linear":
            refs = []
            for run, ls in (("t1_deep24", "--"), ("t1_shallow3", ":")):
                acc = float(select(depth, run=run, readout="f", probe="linear", t=1)[0]["accuracy"])
                refs.append(ax.axhline(acc, color=MUTED, linestyle=ls, linewidth=1, label=LABEL[run]))
            ax.legend(handles=refs, loc="lower right")
        ax.set_xlabel("depth $t$ (recurrent steps)")
        ax.set_title(title)
    axes[0].set_ylabel("Tiny-ImageNet probe (%)")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "grim_srr_depth")


def srr_grad_figure():
    runs = {r["run"]: r for r in load("grim_srr_runs.csv")}
    fig, ax = panel()
    pairs = range(2, 9)
    for run in ("v0", "v1", "v2", "d1", "e2_dense"):
        r = runs[run]
        ax.plot(list(pairs), [float(r[f"grad/cos_step_{k}_vs_{k - 1}"]) for k in pairs], label=LABEL[run],
                **series_style(SLOT[run]))
    ax.set_xticks(list(pairs), [f"{k}/{k - 1}" for k in pairs])
    ax.set_xlabel("consecutive steps")
    ax.set_ylabel("gradient cosine")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=6)
    save_panel(fig, "grim_srr_grad")
    print("conflict rate:", json.dumps({run: runs[run].get("grad/conflict_rate") for run in SLOT}))


if __name__ == "__main__":
    ablation_table()
    adapters_figure()
    srr_table()
    srr_depth_figure()
    srr_grad_figure()
