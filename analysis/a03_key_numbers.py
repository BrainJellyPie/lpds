"""Summary of the main results in Markdown, computed directly from the stored
records.

  python -m analysis.a03_key_numbers --results results [--results_all results_all]

Writes <results>/tables/key_numbers.md.
"""
import argparse
import glob
import json
import os

import pandas as pd

from core.config import DISPLAY


def load(res, exp, group=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(res, exp, "*.json"))):
        d = json.load(open(f))
        if "records" in d:
            rows.extend(r for r in d["records"] if group is None or r["group"] == group)
    return pd.DataFrame(rows)


def ds_mean(df, metric):
    x = df[df.metric == metric]
    return x.groupby(["dataset", "method"]).value.mean().unstack()


def ds_sd(df, metric):
    x = df[df.metric == metric]
    return x.groupby(["dataset", "method"]).value.std().unstack()


def rel_to(dm, ref="Split"):
    return dm.div(dm[ref], axis=0)


def section(title, body):
    return f"## {title}\n\n{body}\n\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--results_all", default=None)
    a = p.parse_args()
    out = []

    rank = load(a.results, "e01_ranking", "rank")
    sp, sd = ds_mean(rank, "spearman"), ds_sd(rank, "spearman")
    order = ["LPDS", "LPDS_stored", "LPDS_absw", "LPDS_varw", "LPDS+Range",
             "IsoForest", "AA-Range", "Mahalanobis", "kNN", "LOF", "VirtualEns",
             "AbsPred", "LeafVar", "IBUG", "ErrModel", "ErrModel+LPDS"]
    order = [m for m in order if m in sp]
    t = sp[order].copy()
    t.index = [DISPLAY.get(i, i) for i in t.index]
    t.loc["Mean"] = sp[order].mean()
    t.loc["Median"] = sp[order].median()
    t.loc["Positive datasets"] = (sp[order] > 0).sum()
    out.append(section("Spearman rho(risk, |error|), mean over 5 seeds (main 9)",
                       t.round(3).to_markdown()))
    s2 = sd[["LPDS"]].copy()
    s2.index = [DISPLAY.get(i, i) for i in s2.index]
    out.append(section("LPDS Spearman SD over seeds", s2.round(3).to_markdown()))
    pr = ds_mean(rank, "prr_mae")[order]
    out.append(section("PRR (MAE), mean over datasets",
                       pr.mean().round(3).to_frame("PRR").T.to_markdown()))

    q = load(a.results, "e01_ranking", "quartile")
    q = q[q.method == "testq"].groupby(["dataset", "metric"]).value.mean().unstack()
    out.append(section("Q1/Q4 MAE ratio by LPDS test quartile",
                       q[["ratio_q1_q4", "monotone_pairs"]].round(3).to_markdown() +
                       f"\n\nmedian ratio = {q.ratio_q1_q4.median():.3f}"))
    acc = load(a.results, "e01_ranking", "acc")
    acc = acc.groupby(["dataset", "metric"]).value.mean().unstack()
    out.append(section("Model accuracy (test) and active-feature fraction",
                       acc[["r2", "rmse", "mae", "active_feature_fraction"]].round(4).to_markdown()))
    ood = ds_mean(load(a.results, "e01_ranking", "ood_pc1_3sd"), "auroc")
    cols = [c for c in ["LPDS", "LPDS+Range", "LPDS+Exceed", "IsoForest", "AA-Range",
                        "Mahalanobis", "kNN", "LOF"] if c in ood]
    out.append(section("AUROC, PC1 shift at 3 sigma (mean over datasets)",
                       ood[cols].mean().round(3).to_frame("AUROC").T.to_markdown()))

    cp = load(a.results, "e02_conformal", "cp")
    methods = ["Split", "WQ", "WLACP_LCP_la_g1", "WLACP_LCP_la_norm_g1",
               "WLACP_LCP_auto", "Mondrian_LPDS4", "LoBoost", "LCP_RBF", "NCP_LPDS",
               "NCP_ErrModel", "NCP_IBUG", "IBUG_Gaussian"]
    rows = {}
    for m in ["cov_all", "cov_q1", "cov_q4", "cov_low_testq", "cov_knn_q4", "wsc", "inf_frac"]:
        rows[m] = ds_mean(cp, m).mean()
    for m in ["is_all", "width_all", "width_q1"]:
        rows[m + "_rel_split"] = rel_to(ds_mean(cp, m)).mean()
    t = pd.DataFrame(rows).loc[[m for m in methods if m in rows["cov_all"].index]]
    out.append(section("Conformal summary (main 9, mean over datasets of 5-seed means)",
                       t.round(3).to_markdown()))
    q1 = ds_mean(cp, "cov_q1")[[m for m in methods if m in rows["cov_all"].index]]
    q1.index = [DISPLAY.get(i, i) for i in q1.index]
    out.append(section("Coverage on the lowest LPDS quartile per dataset", q1.round(3).to_markdown()))

    e10 = os.path.join(a.results, "e10_coverage_check", "summary.json")
    if os.path.exists(e10):
        s = json.load(open(e10))
        t = pd.DataFrame({k: {m: f"{v['mean']:.4f} ± {v['mc_se']:.4f}" for m, v in d.items()}
                          for k, d in s.items()})
        out.append(section("Monte Carlo marginal coverage (target 0.90)", t.to_markdown()))

    ext = load(a.results, "e03_extended")
    if len(ext):
        im = ext[ext.group == "impl"]
        t = ds_mean(im, "spearman_LPDS")
        t.index = [DISPLAY.get(i, i) for i in t.index]
        t.loc["Mean"] = t.mean()
        t.loc["Positive datasets"] = (t.iloc[:-1] > 0).sum()
        out.append(section("LPDS Spearman by implementation", t.round(3).to_markdown()))
        ng = ext[ext.group == "ngboost"].groupby(["dataset", "method", "metric"]).value.mean()
        out.append(section("NGBoost", ng.groupby(["method", "metric"]).mean().unstack().round(3).to_markdown()))
        rf = ext[ext.group == "cp_refit"]
        sp_is = ds_mean(cp, "is_all")["Split"]
        t = {}
        for m in rf.method.unique():
            x = rf[rf.method == m]
            t[m] = {"cov_all": ds_mean(x, "cov_all")[m].mean(),
                    "cov_q1": ds_mean(x, "cov_q1")[m].mean(),
                    "is_rel_split": (ds_mean(x, "is_all")[m] / sp_is).mean()}
        out.append(section("Refitting references", pd.DataFrame(t).T.round(3).to_markdown()))

    sh = load(a.results, "e05_shift", "shift")
    if len(sh):
        x = sh[sh.method == "all:PC1"].groupby(["magnitude", "metric", "dataset"]).value.mean()
        t = x.groupby(["magnitude", "metric"]).mean().unstack()
        cols = [c for c in ["LPDS[sklearn]", "LPDS+Range", "LPDS+Exceed", "IsoForest", "AA-Range",
                            "Mahalanobis", "kNN", "mean_lpds_ratio"] if c in t]
        out.append(section("AUROC vs shift magnitude (PC1, all features)", t[cols].round(3).to_markdown()))

    cs = load(a.results, "e06_calib_size", "calib")
    if len(cs):
        x = cs.groupby(["dataset", "method", "n_cal", "metric"]).value.mean()
        t = x.groupby(["method", "n_cal", "metric"]).mean().unstack()
        out.append(section("Calibration-size sweep (mean over datasets)",
                           t[["cov_all", "cov_q1", "width_ratio_low_testq"]].round(3).to_markdown()))

    for exp in ("e07_noise_bike", "e07_noise_synthetic", "e08_ring"):
        d = load(a.results, exp)
        if len(d):
            g = d.groupby(["method", "metric"]).value.agg(["mean", "std"])
            keep = [m for m in ["r2", "spearman_LPDS", "spearman_IsoForest", "spearman_kNN",
                                "lpds_interior_over_surface", "iso_interior_over_surface",
                                "auroc_LPDS", "auroc_IsoForest"] if m in g.index.get_level_values(1)]
            t = g.reset_index()
            t = t[t.metric.isin(keep)]
            t["cell"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(t["mean"], t["std"])]
            out.append(section(exp, t.pivot(index="method", columns="metric",
                                            values="cell").to_markdown()))

    sens = load(a.results, "e04_sensitivity", "sens")
    if len(sens):
        x = sens.groupby(["dataset", "method", "metric"]).value.mean()
        t = x.groupby(["method", "metric"]).mean().unstack()
        n = sens.groupby("method").dataset.nunique()
        cols = [c for c in ["r2", "n_trees", "spearman_LPDS", "prr_LPDS", "spearman_IsoForest",
                            "spearman_kNN", "Split_cov_q1", "WLACP_LCP_cov_q1",
                            "Mondrian_cov_q1"] if c in t]
        t = t[cols]
        t["n_datasets"] = n
        out.append(section("Sensitivity (mean over datasets)", t.round(3).to_markdown()))
        lp = x.xs("spearman_LPDS", level="metric").unstack("method")
        lp.index = [DISPLAY.get(i, i) for i in lp.index]
        out.append(section("Sensitivity: LPDS Spearman per dataset", lp.round(3).to_markdown()))

    if a.results_all:
        r19 = load(a.results_all, "e01_ranking", "rank")
        s19 = ds_mean(r19, "spearman")
        cols = [c for c in order if c in s19]
        t = s19[cols].copy()
        t.index = [DISPLAY.get(i, i) for i in t.index]
        t.loc["Mean"] = s19[cols].mean()
        t.loc["Median"] = s19[cols].median()
        t.loc["Positive datasets"] = (s19[cols] > 0).sum()
        out.append(section(f"Spearman, all {s19.shape[0]} datasets", t.round(3).to_markdown()))
        c19 = load(a.results_all, "e02_conformal", "cp")
        rows = {}
        for m in ["cov_all", "cov_q1", "cov_knn_q4", "wsc"]:
            rows[m] = ds_mean(c19, m).mean()
        for m in ["is_all", "width_all"]:
            rows[m + "_rel_split"] = rel_to(ds_mean(c19, m)).mean()
        t = pd.DataFrame(rows).loc[[m for m in methods if m in rows["cov_all"].index]]
        out.append(section(f"Conformal summary, all {ds_mean(c19, 'cov_all').shape[0]} datasets",
                           t.round(3).to_markdown()))
        e19 = load(a.results_all, "e03_extended", "impl")
        if len(e19):
            t = ds_mean(e19, "spearman_LPDS")
            t.index = [DISPLAY.get(i, i) for i in t.index]
            t.loc["Mean"] = t.mean()
            t.loc["Positive datasets"] = (t.iloc[:-1] > 0).sum()
            out.append(section(f"LPDS Spearman by implementation, all {len(t) - 2} datasets",
                               t.round(3).to_markdown()))

    path = os.path.join(a.results, "tables", "key_numbers.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Key numbers\n\n" + "".join(out))
    print("written", path)


if __name__ == "__main__":
    main()
