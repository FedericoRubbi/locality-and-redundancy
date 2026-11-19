import json

import matplotlib.pyplot as plt
import numpy as np

from style import FULL, MUTED, load, save, select, series_style, stat, write_table

DATASETS = [("MNIST", "MNIST"), ("F-MNIST", "F-MNIST"), ("CIFAR10", "CIFAR-10"), ("CIFAR100", "CIFAR-100")]
METHODS = [("bp", "BP (probe on layer)", 0), ("itmf", "IT-MF (block head)", 1)]


def figure():
    rows = load("itmf_rerun_blocks.csv")
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.9))
    for ax, (d, name) in zip(axes, DATASETS):
        for m, label, slot in METHODS:
            sel = select(rows, method=m, dataset=d)
            blocks = sorted({int(r["block"]) for r in sel})
            s = np.array([stat(select(sel, block=b), "test_acc")[:2] for b in blocks], dtype=float)
            st = series_style(slot)
            ax.plot(blocks, s[:, 0], label=label, **st)
            ax.fill_between(blocks, s[:, 0] - s[:, 1], s[:, 0] + s[:, 1], color=st["color"], alpha=0.2, linewidth=0)
        ax.set_title(name)
        ax.set_xlabel("block / layer")
        ax.set_xticks([1, 4, 8])
        top = max(float(r["test_acc"]) for r in select(rows, dataset=d))
        ax.set_ylim(0, top * 1.04)
    axes[0].set_ylabel("test accuracy (%)")
    axes[-1].legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    save(fig, "itmf_blocks")


def configs_table():
    rows = select(load("itmf_rerun.csv"), seed=0)
    body = []
    for m, label in (("itmf", r"\itmf{}"), ("bp", "BP")):
        for d, name in DATASETS:
            r = select(rows, method=m, dataset=d)[0]
            c = json.loads(r["config"])
            sel = "test" if "test" in r["selection"] else "val."
            row = [label, name, f"{sel} ({r['selected_trial']})", f"${c['lr']:.1e}$".replace("e-0", r"\mathrm{e}{-}"),
                   f"${c['weight_decay']:.0e}$".replace("e-0", r"\mathrm{e}{-}"), str(c["batch_size"]),
                   f"{c['dropout']:.2f}"]
            if m == "itmf":
                act = {"silu": "SiLU", "gelu": "GELU", "relu": "ReLU"}.get(c["activation"], c["activation"])
                row += [act, f"{c['alpha_max']:.1f}", c["schedule_type"], f"{c['gamma_rec']:.2f}",
                        "yes" if c["tie_weights"] else "no"]
            else:
                row += ["ReLU"] + ["---"] * 4
            body.append(row)
        if m == "itmf":
            body.append("midrule")
    header = [["Method", "Data", "Selected on (trial)", "LR", "WD", "Batch", "Dropout", "Act.",
               r"$\alpha_{\max}$", "Schedule", r"$\gamma$", "Tied"]]
    write_table("itmf_configs", "@{}llllllllllll@{}", header, body, "figures_itmf.py")


def noise_figure():
    rng = np.random.default_rng(0)
    corners = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.75))
    for ax, alpha in zip(axes, [0, 2, 8, 32]):
        conc = np.ones(3) + alpha * np.array([1.0, 0.0, 0.0])
        pts = rng.dirichlet(conc, size=400) @ corners
        ax.add_patch(plt.Polygon(corners, closed=True, fill=False, edgecolor=MUTED, linewidth=0.8))
        ax.scatter(pts[:, 0], pts[:, 1], s=4, color=series_style(1)["color"], alpha=0.5, linewidths=0)
        ax.scatter([corners[0, 0]], [corners[0, 1]], s=16, color=series_style(0)["color"], zorder=3)
        ax.text(0.5, 0.93, r"$y^\ast$", ha="center", va="bottom", fontsize=7)
        ax.set_title(rf"$\alpha_i = {alpha}$")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.02)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    save(fig, "itmf_noise")


if __name__ == "__main__":
    figure()
    noise_figure()
    configs_table()
