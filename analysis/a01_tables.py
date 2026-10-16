"""Aggregate result records into CSV and Markdown tables.

For every experiment folder under --results, all records are collected,
averaged over seeds per dataset (mean and SD), and averaged over datasets
(grand mean of per-dataset means). Output goes to <results>/tables/.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from core.config import DISPLAY


def cells(mean, sd, digits=3):
    """'m ± s' strings; pandas >= 3 keeps NaN through astype(str), so format
    explicitly and drop the SD when only one seed is available."""
    return [f"{m:.{digits}f} ± {v:.{digits}f}" if np.isfinite(v) else f"{m:.{digits}f}"
            for m, v in zip(mean, sd)]


def load(results, exp):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, exp, "*.json"))):
        if f.endswith("summary.json") or f.endswith("timing.json"):
            continue
        d = json.load(open(f))
        rows.extend(d["records"])
    return pd.DataFrame(rows)


def per_dataset(df, keys, metric_filter=None):
    if metric_filter is not None:
        df = df[df.metric.isin(metric_filter)]
    g = df.groupby(["dataset"] + keys + ["metric"]).value
    return g.mean().rename("mean").to_frame().join(g.std().rename("sd")).join(
        g.count().rename("n_seeds")).reset_index()


def grand(pd_df, keys):
    g = pd_df.groupby(keys + ["metric"])["mean"]
    out = g.mean().rename("grand_mean").to_frame()
    out["grand_median"] = g.median()
    out["n_datasets"] = g.count()
    return out.reset_index()


def fmt_table(pd_df, row, col, metric, digits=3):
    t = pd_df[pd_df.metric == metric]
    t = t.assign(cell=cells(t["mean"], t["sd"], digits))
    piv = t.pivot_table(index=row, columns=col, values="cell", aggfunc="first")
    if row == "dataset":
        piv.index = [DISPLAY.get(i, i) for i in piv.index]
    gm = t.groupby(col)["mean"].mean().round(digits)
    piv.loc["Grand mean"] = gm.astype(str)
    return piv


def save(name, tables, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".md"), "w") as f:
        for title, t in tables:
            f.write(f"## {title}\n\n{t.to_markdown()}\n\n")
            t.to_csv(os.path.join(out_dir, f"{name}__{title.replace(' ', '_')}.csv"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    a = p.parse_args()
    out = os.path.join(a.results, "tables")

    df = load(a.results, "e01_ranking")
    if len(df):
        r = per_dataset(df[df.group == "rank"], ["method"], ["spearman", "prr_mae", "prr_mse"])
        tabs = [(f"rank {m}", fmt_table(r, "dataset", "method", m))
                for m in ("spearman", "prr_mae")]
        tabs.append(("rank grand", grand(r, ["method"]).pivot(
            index="method", columns="metric", values="grand_mean").round(3)))
        o = per_dataset(df[df.group == "ood_pc1_3sd"], ["method"])
        tabs.append(("ood auroc pc1 3sd", fmt_table(o, "dataset", "method", "auroc")))
        acc = per_dataset(df[df.group == "acc"], ["method"])
        tabs.append(("accuracy", acc.pivot_table(index="dataset", columns="metric",
                                                 values="mean").round(4)))
        q = per_dataset(df[df.group == "quartile"], ["method"])
        tabs.append(("quartile MAE (test quartiles)", q[q.method == "testq"].pivot_table(
            index="dataset", columns="metric", values="mean").round(4)))
        ret = df[(df.group.isin(["rank", "rank_ref"])) & df.metric.str.startswith("ret_mae")]
        rr = ret.groupby(["dataset", "method", "metric"]).value.mean().reset_index()
        base = rr[rr.metric == "ret_mae_0.00"].set_index(["dataset", "method"]).value
        rr["rel"] = rr.value / rr.set_index(["dataset", "method"]).index.map(base)
        tabs.append(("retention (MAE relative to no rejection, mean over datasets)",
                     rr.groupby(["method", "metric"]).rel.mean().unstack().round(3)))
        save("e01_ranking", tabs, out)

    df = load(a.results, "e02_conformal")
    if len(df):
        c = per_dataset(df[df.group == "cp"], ["method"])
        tabs = [(m, fmt_table(c, "dataset", "method", m)) for m in
                ("cov_all", "cov_q1", "cov_low_testq", "cov_knn_q4", "wsc",
                 "width_all", "width_q1", "is_all", "inf_frac")]
        g = grand(c, ["method"]).pivot(index="method", columns="metric", values="grand_mean")
        cols = [k for k in ("cov_all", "cov_q1", "cov_q2", "cov_q3", "cov_q4",
                            "cov_low_testq", "cov_knn_q4", "wsc", "inf_frac",
                            "ess_mean", "ess_q1_mean") if k in g]
        tabs.append(("grand", g[cols].round(3)))
        # interval score and width relative to split CP, per dataset then averaged
        cw = c[c.metric.isin(["is_all", "width_all", "width_q1", "is_q1"])]
        sp = cw[cw.method == "Split"].set_index(["dataset", "metric"])["mean"]
        cw = cw.assign(rel=cw["mean"].to_numpy() /
                       sp.reindex(list(zip(cw.dataset, cw.metric))).to_numpy())
        tabs.append(("relative to Split (mean over datasets)",
                     cw.groupby(["method", "metric"]).rel.mean().unstack().round(3)))
        save("e02_conformal", tabs, out)

    df = load(a.results, "e03_extended")
    if len(df):
        tabs = []
        for grp in ("ngboost", "cp_refit", "impl", "impl_cp"):
            x = per_dataset(df[df.group == grp], ["method"])
            tabs.append((grp + " grand", grand(x, ["method"]).pivot(
                index="method", columns="metric", values="grand_mean").round(3)))
        x = per_dataset(df[df.group == "impl"], ["method"])
        tabs.append(("impl spearman_LPDS", fmt_table(x, "dataset", "method", "spearman_LPDS")))
        save("e03_extended", tabs, out)

    df = load(a.results, "e04_sensitivity")
    if len(df):
        x = per_dataset(df, ["method"])
        tabs = [("grand", grand(x, ["method"]).pivot(
            index="method", columns="metric", values="grand_mean").round(3)),
            ("spearman_LPDS", fmt_table(x, "dataset", "method", "spearman_LPDS"))]
        save("e04_sensitivity", tabs, out)

    df = load(a.results, "e05_shift")
    if len(df):
        s = df[df.group == "shift"]
        x = s.groupby(["method", "magnitude", "metric", "dataset"]).value.mean().reset_index()
        gm = x.groupby(["method", "magnitude", "metric"]).value.mean().unstack("metric")
        tabs = [("auroc vs magnitude (grand mean)", gm.round(3))]
        h = x[x.magnitude == 3.0].pivot_table(index=["dataset"], columns=["method", "metric"],
                                               values="value").round(3)
        tabs.append(("magnitude 3 per dataset", h))
        save("e05_shift", tabs, out)

    df = load(a.results, "e06_calib_size")
    if len(df):
        x = df.groupby(["dataset", "method", "n_cal", "metric"]).value.agg(["mean", "std"]).reset_index()
        tabs = []
        for m in ("cov_all", "cov_q1", "cov_low_testq", "width_ratio_low_testq"):
            t = x[x.metric == m]
            t = t.assign(cell=cells(t["mean"], t["std"]))
            tabs.append((m, t.pivot_table(index=["dataset", "n_cal"], columns="method",
                                          values="cell", aggfunc="first")))
        save("e06_calib_size", tabs, out)

    for exp in ("e07_noise_bike", "e07_noise_synthetic", "e08_ring"):
        df = load(a.results, exp)
        if len(df):
            x = df.groupby(["method", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
            t = x.assign(cell=cells(x["mean"], x["std"])).pivot(
                index="method", columns="metric", values="cell")
            save(exp, [("summary", t)], out)
    print("tables written to", out)


if __name__ == "__main__":
    main()
