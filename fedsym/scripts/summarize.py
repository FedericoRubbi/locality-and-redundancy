import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

DEFAULT_KEYS = ["final/global/linprobe", "final/global/knn", "final/global/loss", "final/merge/penalty_loss",
                "final/sym/gap_loss", "final/sym/gap_knn", "final/drift/perm_id_frac", "final/drift/rot_explained",
                "final/barrier/naive_loss", "final/barrier/aligned_loss"]
BY = ["dataset", "mode", "alpha", "local_epochs", "rounds", "clients", "client_init", "agg", "vname"]


def load(path):
    rows = []
    for f in glob.glob(os.path.join(path, "**", "*.json"), recursive=True):
        r = json.load(open(f))
        row = {**r["config"], **{k: v for k, v in r["info"].items() if not isinstance(v, list)}, **r["summary"]}
        for k in ("merge/penalty_loss", "sym/gap_loss", "sym/gap_knn", "drift/perm_id_frac", "drift/rot_explained"):
            vals = [h[k] for h in r["history"] if k in h]
            if vals:
                row[f"avg/{k}"] = float(np.mean(vals))
        toks = row["variant"].split("+")
        if row.get("wasym_scale", 4.0) == 1:
            toks = [t.replace("wasym", "wasym_weak", 1) if t.startswith("wasym") and "_weak" not in t else t
                    for t in toks]
            toks = [t.replace("wasym_weak_band", "wasym_band_weak") for t in toks]
        row["vname"] = "+".join(toks)
        rows.append(row)
    return pd.DataFrame(rows)


def table(df, keys, by):
    keys = [k for k in keys if k in df.columns]
    by = [b for b in by if b in df.columns]
    g = df.groupby(by, dropna=False)[keys]
    mean, std, n = g.mean(), g.std(), g.size()
    out = mean.copy().astype(object)
    for k in keys:
        out[k] = [f"{m:.4f}±{s:.4f}" if not np.isnan(s) else f"{m:.4f}" for m, s in zip(mean[k], std[k])]
    out["n"] = n
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    p.add_argument("--by", nargs="+", default=BY)
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    df = load(a.path)
    if a.csv:
        df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    t = table(df, a.keys, a.by)
    t.columns = [c.replace("final/", "") for c in t.columns]
    print(t.to_string())
