import matplotlib.pyplot as plt

from style import FULL, MUTED, load, save, select, series_style, write_table

SRC = "figures_vit.py"
IN = "ImageNet100"
RUNS = [
    ("ImageNet100_baseline/run_002", "BP ViT", 0),
    ("ImageNet100/run_007_plain_greedyvit", "Greedy ViT", 1),
    ("ImageNet100_infovit_v1/run_005", "InfoViT", 2),
]


def last_epoch(rows):
    return max(int(r["epoch"]) for r in rows)


def first_last_block(run):
    blocks = select(load("vit_blocks.csv"), run=run)
    if not blocks:
        return None, None
    final = select(blocks, epoch=last_epoch(blocks))
    acc = {int(r["block"]): float(r["test_acc"]) for r in final}
    return acc[1], acc[max(acc)]


def final_acc(run):
    rows = select(load("vit_epochs.csv"), run=run)
    return float(select(rows, epoch=last_epoch(rows))[0]["test_acc"]), last_epoch(rows)


def cell(x):
    return "---" if x is None else f"${x:.1f}$"


def table():
    body = []
    for run, label, _ in RUNS:
        b1, bl = first_last_block(run)
        acc, _ = final_acc(run)
        body.append([label, cell(b1), cell(acc)])
    header = [["Model", "Block 1", "Final"]]
    write_table("vit", "@{}lcc@{}", header, body, SRC, best={1: "max", 2: "max"})


def figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.3))
    bp_acc, _ = final_acc(RUNS[0][0])
    ax1.axhline(bp_acc, color=MUTED, linestyle="--", linewidth=1, label="BP ViT (end to end)")
    for run, label, slot in RUNS[1:]:
        blocks = select(load("vit_blocks.csv"), run=run)
        final = sorted(select(blocks, epoch=last_epoch(blocks)), key=lambda r: int(r["block"]))
        ax1.plot([int(r["block"]) for r in final], [float(r["test_acc"]) for r in final], label=label,
                 **series_style(slot))
    for run, label, slot in RUNS:
        rows = sorted(select(load("vit_epochs.csv"), run=run), key=lambda r: int(r["epoch"]))
        ax2.plot([int(r["epoch"]) for r in rows], [float(r["test_acc"]) for r in rows], label=label,
                 markevery=5, **series_style(slot))
    ax1.set_xlabel("block")
    ax1.set_ylabel("ImageNet-100 test accuracy (%)")
    ax1.set_title("(a) per block, after the last epoch")
    ax1.set_ylim(0, 65)
    ax1.legend(loc="lower right", fontsize=6)
    ax2.set_xlabel("epoch")
    ax2.set_title("(b) last block (whole network for BP)")
    ax2.set_ylim(0, 65)
    ax2.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    save(fig, "vit_blocks")


if __name__ == "__main__":
    table()
    figure()
