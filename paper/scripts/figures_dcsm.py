import matplotlib.pyplot as plt

from style import FULL, MUTED, load, save, select, series_style

EVALS = [
    ("a5l8r648", "SupCon, checkpoint 1", 0),
    ("e6hb8294", "SupCon, checkpoint 2", 1),
    ("em990jzk", "SupCon, checkpoint 2 (re-run)", 2),
    ("i5msnthk", "InfoMax (accuracy not logged)", 3),
]


def figure():
    steps = load("dcsm_eval_steps.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.2))
    for run, label, slot in EVALS:
        rs = sorted(select(steps, run=run), key=lambda r: int(r["step"]))
        st = series_style(slot)
        x = [int(r["step"]) for r in rs]
        if rs[0]["eval/accuracy"]:
            ax1.plot(x, [100 * float(r["eval/accuracy"]) for r in rs], label=label, markevery=100, **st)
        drift = [(int(r["step"]), 1 - float(r["eval/trajectory_cosine_drift"])) for r in rs
                 if r["eval/trajectory_cosine_drift"] and int(r["step"]) > 0]
        ax2.plot(*zip(*drift), label=label, markevery=100, **st)
    ax1.axhline(1.0, color=MUTED, linestyle="--", linewidth=1, label="chance (1%)")
    ax1.set_xlabel("diffusion timestep $t$")
    ax1.set_ylabel("CIFAR-100 accuracy (%)")
    ax1.set_ylim(0, 3)
    ax1.set_title("(a) accuracy of the denoised label")
    ax1.invert_xaxis()
    ax1.legend(loc="upper right", fontsize=6)
    ax2.set_yscale("log")
    ax2.set_xlabel("diffusion timestep $t$")
    ax2.set_ylabel("1 - trajectory cosine")
    ax2.set_title(r"(b) step size: $1-\cos(Y_t, Y_{t-1})$")
    ax2.invert_xaxis()
    ax2.legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, "dcsm_reverse")


if __name__ == "__main__":
    figure()
