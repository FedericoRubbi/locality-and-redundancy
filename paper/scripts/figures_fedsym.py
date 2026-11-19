import matplotlib.pyplot as plt
import numpy as np

from style import FULL, MUTED, PALETTE, load, pm, save, select, series_style, stat, write_table

SRC = "figures_fedsym.py"
DATASETS = [("fmnist", "F-MNIST"), ("cifar10", "CIFAR-10")]
NAMES = {
    "sym": ("Symmetric", "Symmetric"),
    "dropasym": ("Per-unit dropout", "Per-unit dropout"),
    "lrasym": ("Per-unit learning rate", "Per-unit LR"),
    "act_asym": ("Per-unit activation slope", "Per-unit act. slope"),
    "res_id": ("Residual (identity)", "Residual (identity)"),
    "res_rand+act_asym": ("Residual (random) + act.\\ slope", "Residual (random) + act."),
    "wasym": (r"\wasym{}", "W-Asym"),
    "wasym_band": (r"Banded \wasym{}", "Banded W-Asym"),
    "wasym+act_asym": (r"\wasym{} + act.\ slope", "W-Asym + act. slope"),
    "wasym+res_diag": (r"\wasym{} + residual (diag.)", "W-Asym + res. diag"),
    "wasym+res_diag+act_asym": (r"\wasym{} + res.\ (diag.) + act.", "W-Asym + res. + act."),
    "res_diag": ("Residual (diagonal)", "Residual (diag.)"),
    "wasym_band+act_asym": (r"Banded \wasym{} + act.\ slope", "Banded W-Asym + act."),
}
SLOT = {"sym": 0, "wasym": 1, "wasym+act_asym": 2, "res_id": 3, "res_rand+act_asym": 4, "sym@aligned": 5}
MAIN = ["sym", "wasym", "wasym+act_asym", "res_id", "res_rand+act_asym"]

runs, rounds = load("fedsym_runs.csv"), load("fedsym_rounds.csv")


def strong(variant):
    return {"wasym_scale": "4.0"} if "wasym" in variant else {}


def oneshot_table():
    variants = ["sym", "dropasym", "lrasym", "act_asym", "res_id", "wasym", "wasym_band", "wasym+act_asym",
                "wasym+res_diag+act_asym"]
    body = []
    for v in variants:
        row = [NAMES[v][0]]
        for d, _ in DATASETS:
            central = select(runs, group="screen", dataset=d, variant=v, mode="central", **strong(v))
            shot = select(runs, group="screen", dataset=d, variant=v, mode="fedavg", rounds=1, clients=4, alpha="0.0",
                          local_epochs=20, client_init="independent", **strong(v))
            assert len(central) == 3 and len(shot) == 3, (v, d, len(central), len(shot))
            row += [pm(stat(central, "global/linprobe", 100)), pm(stat(shot, "merge/penalty_loss"), 2),
                    pm(stat(shot, "sym/gap_loss"), 2)]
        body.append(row)
        if v in ("res_id",):
            body.append("midrule")
    header = [["", r"\multicolumn{3}{c}{F-MNIST}", r"\multicolumn{3}{c}{CIFAR-10}"],
              [r"\cmidrule(lr){2-4}\cmidrule(lr){5-7} Variant"] + ["Central probe", r"$\MP$", r"$\SG$"] * 2]
    write_table("fedsym_oneshot", "@{}lcccccc@{}", header, body, SRC,
                best={1: "max", 2: "min", 4: "max", 5: "min"})


def strength_figure():
    mechanisms = [
        ("wasym", "wasym_scale", "W-Asym scale $s$ (fraction 0.1)", PALETTE[1], {"wasym_frac": "0.1"}),
        ("dropasym", "drop_mean", "Per-unit dropout, mean rate", PALETTE[6], {}),
        ("lrasym", "lr_ratio", "Per-unit LR, max ratio", PALETTE[7], {}),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.6), sharey="row")
    for i, (d, dname) in enumerate(DATASETS):
        ref_c = select(runs, group="strength", sweep="", dataset=d, variant="sym", mode="central")
        ref_m = select(runs, group="strength", sweep="", dataset=d, variant="sym", mode="fedavg")
        rx, ry = stat(ref_c, "global/linprobe", 100)[0], stat(ref_m, "merge/penalty_loss")[0]
        for j, (v, knob, label, colour, extra) in enumerate(mechanisms):
            ax = axes[i, j]
            pts = []
            levels = sorted({r[knob] for r in select(runs, group="strength", variant=v, dataset=d) if r["sweep"]},
                            key=float)
            for lv in levels:
                c = select(runs, group="strength", dataset=d, variant=v, mode="central", **{knob: lv}, **extra)
                m = select(runs, group="strength", dataset=d, variant=v, mode="fedavg", **{knob: lv}, **extra)
                if c and m:
                    pts.append((stat(c, "global/linprobe", 100)[0], stat(m, "merge/penalty_loss")[0], lv))
            xs, ys, lvs = zip(*pts)
            ax.plot(xs, ys, color=colour, marker="o", zorder=3)
            for x, y, lv in pts:
                ax.annotate(f"{float(lv):g}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6,
                            color=MUTED)
            ref, = ax.plot([rx], [ry], marker="*", color=MUTED, markersize=8, linestyle="none", zorder=4)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(f"{dname}\none-shot $\\mathrm{{MP}}$ (nats)\nlower is better")
            if i == 1:
                ax.set_xlabel("centralized probe (%)\nhigher is better")
            ax.set_ylim(bottom=0)
            ax.margins(x=0.12)
    fig.legend([ref], ["symmetric MLP (reference); point labels: strength"], loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fedsym_strength")


def comm_table():
    eps = [1, 20, 50]
    body = []
    for d, dname in DATASETS:
        body.append([rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{dname}}}}}"])
        for v in MAIN:
            row = [NAMES[v][0]]
            for a in ("0.0", "0.1"):
                for e in eps:
                    rs = select(runs, group="comm", dataset=d, variant=v, alpha=a, local_epochs=e, **strong(v))
                    row.append(pm(stat(rs, "global/linprobe", 100)))
            body.append(row)
    header = [["", r"\multicolumn{3}{c}{IID ($\alpha=0$)}", r"\multicolumn{3}{c}{Dirichlet $\alpha=0.1$}"],
              [r"\cmidrule(lr){2-4}\cmidrule(lr){5-7} Variant"] + [f"$E={e}$" for e in eps] * 2]
    write_table("fedsym_comm", "@{}lcccccc@{}", header, body, SRC, best={c: "max" for c in range(1, 7)})


def comm_figure():
    d, a = "cifar10", "0.1"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.3))
    eps = [1, 2, 5, 10, 20, 50]
    for v in MAIN:
        st = series_style(SLOT[v])
        means, stds = [], []
        for e in eps:
            names = {r["name"] for r in select(runs, group="comm", dataset=d, variant=v, alpha=a, local_epochs=e,
                                               **strong(v))}
            first = [r for r in select(rounds, group="comm", round=1) if r["name"] in names]
            m, s, _ = stat(first, "merge/penalty_loss")
            means.append(m)
            stds.append(s or 0)
        ax1.errorbar(eps, means, yerr=stds, label=NAMES[v][1], capsize=2, **st)
        names = {r["name"] for r in select(runs, group="comm", dataset=d, variant=v, alpha=a, local_epochs=20,
                                           **strong(v))}
        per = select([r for r in rounds if r["name"] in names], group="comm")
        rnds = sorted({int(r["round"]) for r in per})
        curve = np.array([stat(select(per, round=t), "global/linprobe", 100)[:2] for t in rnds], dtype=float)
        ax2.plot(rnds, curve[:, 0], label=NAMES[v][1], **st)
        ax2.fill_between(rnds, curve[:, 0] - curve[:, 1], curve[:, 0] + curve[:, 1], color=st["color"], alpha=0.15,
                         linewidth=0)
    ax1.axhline(0, color=MUTED, linewidth=0.6)
    ax1.set_xscale("log")
    ax1.set_xticks(eps, [str(e) for e in eps])
    ax1.set_xlabel("local epochs per round $E$")
    ax1.set_ylabel("first-merge $\\mathrm{MP}$ (nats)")
    ax1.set_title("(a) penalty of the first FedAvg merge")
    ax2.set_xlabel("round")
    ax2.set_ylabel("global probe (%)")
    ax2.set_title("(b) global model, $E=20$")
    ax2.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    save(fig, "fedsym_comm")


def locality_table():
    variants = ["sym", "dropasym", "lrasym", "act_asym", "res_diag", "wasym", "wasym+act_asym", "wasym+res_diag",
                "wasym_band", "wasym_band+act_asym"]
    body = []
    for v in variants:
        rs = select(runs, group="screen", dataset="cifar10", variant=v, mode="fedavg", rounds=2, local_epochs=20,
                    alpha="0.1", **strong(v))
        assert len(rs) == 3, (v, len(rs))
        body.append([NAMES[v][0], pm(stat(rs, "loc/perm_id_frac", 100)), pm(stat(rs, "loc/loc_ratio"), 2),
                     pm(stat(rs, "loc/loc_cov8", 100)), pm(stat(rs, "sym/gap_agg_full_loss"), 3),
                     pm(stat(rs, "sym/gap_agg_k8_loss"), 3)])
    header = [["Variant", r"Unmoved (\%)", "Locality ratio", r"Within 8 (\%)", r"$\SG$ full", r"$\SG$ local-8"]]
    write_table("fedsym_locality", "@{}lccccc@{}", header, body, SRC)


def agg_table():
    def probes(d, v, a, e, agg):
        rs = select(runs, group="commagg", dataset=d, variant=v, alpha=a, local_epochs=e, agg=agg)
        if not rs and agg == "naive":
            rs = select(runs, group="comm", dataset=d, variant=v, alpha=a, local_epochs=e, **strong(v))
        keys = {(r["group"], r["name"]) for r in rs}
        first = [r for r in select(rounds, round=1) if (r["group"], r["name"]) in keys]
        assert len(first) == len(rs) == 3, (d, v, a, e, agg, len(first), len(rs))
        return pm(stat(first, "global/linprobe", 100)), pm(stat(rs, "global/linprobe", 100))

    body = []
    for (d, dname), (a, aname) in [(d, a) for d in DATASETS for a in (("0.0", "IID"), ("0.1", r"Dirichlet $\alpha=0.1$"))]:
        body.append([rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{dname}}}, {aname}}}"])
        for v, aligned in [("sym", "aligned"), ("wasym", "aligned_local"), ("wasym_band", "aligned_local")]:
            row = [NAMES[v][0]]
            for e in (20, 50):
                (n1, n10), (a1, a10) = probes(d, v, a, e, "naive"), probes(d, v, a, e, aligned)
                row += [n1, a1, n10, a10]
            body.append(row)
    header = [["", r"\multicolumn{4}{c}{$E=20$}", r"\multicolumn{4}{c}{$E=50$}"],
              [r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}", r"\multicolumn{2}{c}{Round 1}", r"\multicolumn{2}{c}{Round 10}",
               r"\multicolumn{2}{c}{Round 1}", r"\multicolumn{2}{c}{Round 10}"],
              [r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9} Variant"]
              + ["Naive", "Aligned"] * 4]
    write_table("fedsym_agg", "@{}lcccccccc@{}", header, body, SRC, best={c: "max" for c in range(1, 9)})


GOSSIP = [("sym", "naive"), ("sym", "aligned"), ("wasym", "naive"), ("wasym+act_asym", "naive"), ("res_id", "naive")]


def gossip_runs(d, v, agg, e=20, g=2):
    return select(runs, group="gossip", dataset=d, variant=v, agg=agg, local_epochs=e, client_init="independent",
                  group_size=g)


def gossip_table():
    body = []
    for v, agg in GOSSIP:
        row = [NAMES[v][0] + (" (aligned merges)" if agg == "aligned" else "")]
        for d, _ in DATASETS:
            rs = gossip_runs(d, v, agg)
            names = {r["name"] for r in rs}
            first = [r for r in select(rounds, group="gossip", round=1) if r["name"] in names]
            row += [pm(stat(first, "gossip/penalty_loss"), 2), pm(stat(rs, "gossip/consensus_dist"), 2),
                    pm(stat(rs, "gossip/post_linprobe", 100))]
        body.append(row)
    header = [["", r"\multicolumn{3}{c}{F-MNIST}", r"\multicolumn{3}{c}{CIFAR-10}"],
              [r"\cmidrule(lr){2-4}\cmidrule(lr){5-7} Variant"] + [r"$\MP$ round 1", "Consensus", "Client probe"] * 2]
    write_table("fedsym_gossip", "@{}lcccccc@{}", header, body, SRC,
                best={1: "min", 3: "max", 4: "min", 6: "max"})


def gossip_figure():
    d = "cifar10"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.3))
    for v, agg in GOSSIP:
        key = "sym@aligned" if agg == "aligned" else v
        st = series_style(SLOT[key])
        label = NAMES[v][1] + (" (aligned)" if agg == "aligned" else "")
        names = {r["name"] for r in gossip_runs(d, v, agg)}
        per = [r for r in select(rounds, group="gossip") if r["name"] in names]
        rnds = sorted({int(r["round"]) for r in per})
        for ax, col, scale in ((ax1, "gossip/penalty_loss", 1), (ax2, "gossip/post_knn", 100)):
            curve = np.array([stat(select(per, round=t), col, scale)[:2] for t in rnds], dtype=float)
            ax.plot(rnds, curve[:, 0], label=label, **st)
            ax.fill_between(rnds, curve[:, 0] - curve[:, 1], curve[:, 0] + curve[:, 1], color=st["color"],
                            alpha=0.15, linewidth=0)
    ax1.axhline(0, color=MUTED, linewidth=0.6)
    ax1.set_xlabel("round")
    ax1.set_ylabel("group-merge $\\mathrm{MP}$ (nats)")
    ax1.set_title("(a) penalty of the pairwise merges")
    ax2.set_xlabel("round")
    ax2.set_ylabel("client kNN accuracy (%)")
    ax2.set_title("(b) models held by the clients")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fedsym_gossip")


if __name__ == "__main__":
    oneshot_table()
    strength_figure()
    comm_table()
    comm_figure()
    locality_table()
    agg_table()
    gossip_table()
    gossip_figure()
