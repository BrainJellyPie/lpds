"""Numbers quoted in the revised manuscript that are not already in the
tables of a01 to a03, written to <results>/tables/paper_numbers.json.

  python -m analysis.a04_paper_numbers --results results --results_all results_all
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.a02_stats import compare
from core.config import DISPLAY


def load(res, exp, group=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(res, exp, "*.json"))):
        d = json.load(open(f))
        if "records" in d:
            rows.extend(r for r in d["records"] if group is None or r["group"] == group)
    return pd.DataFrame(rows)


def ds_mean(df, metric):
    return df[df.metric == metric].groupby(["dataset", "method"]).value.mean().unstack()


def stored_vs_routed(res):
    vals = []
    for f in sorted(glob.glob(os.path.join(res, "e01_ranking", "*.npz"))):
        z = np.load(f)
        vals.append(spearmanr(z["risk__LPDS"], z["risk__LPDS_stored"])[0])
    return {"n": len(vals), "min": float(np.min(vals)), "mean": float(np.mean(vals))}


def cp_frame(res):
    """e02 records plus the refitting references of e03, with the interval
    score divided by that of split CP on the same dataset and seed, and the
    distance of the lowest-quartile coverage to the target."""
    cp = load(res, "e02_conformal", "cp")
    ref = load(res, "e03_extended", "cp_refit")
    df = pd.concat([cp, ref], ignore_index=True)
    rels = []
    for m_in, m_out in (("is_all", "is_rel"), ("width_all", "width_rel"), ("width_q1", "width_q1_rel")):
        piv = df[df.metric == m_in].pivot_table(index=["dataset", "seed"], columns="method", values="value")
        rel = piv.div(piv["Split"], axis=0).stack().rename("value").reset_index()
        rel["metric"] = m_out
        rels.append(rel)
    rel = pd.concat(rels, ignore_index=True)
    cq = df[df.metric == "cov_q1"].copy()
    cq["value"] = -np.abs(cq["value"] - 0.9)
    cq["metric"] = "cov_q1_close"
    return pd.concat([df, rel, cq], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--results_all", default="results_all")
    a = p.parse_args()
    out = {}
    for tag, res in (("core9", a.results), ("all19", a.results_all)):
        o = {}
        o["stored_vs_routed_spearman"] = stored_vs_routed(res)
        rank = load(res, "e01_ranking", "rank")
        sp = ds_mean(rank, "spearman")
        o["mean_abs_spearman"] = sp.abs().mean().to_dict()
        o["mean_spearman"] = sp.mean().to_dict()
        o["mean_spearman_wo_superconduct"] = sp.drop(index="superconduct").mean().to_dict()
        o["positive_datasets"] = (sp > 0).sum().astype(int).to_dict()
        ci = pd.read_csv(os.path.join(res, "tables", "spearman_bootstrap_ci.csv"))
        c = ci[ci.method == "LPDS"]
        o["lpm_ci_above_zero"] = int((c.ci_lo > 0).sum())
        o["lpm_ci_below_zero"] = int((c.ci_hi < 0).sum())
        o["lpm_ci_contains_zero"] = [d for d, lo, hi in zip(c.dataset, c.ci_lo, c.ci_hi) if lo <= 0 <= hi]
        o["lpm_ci_below_zero_names"] = [d for d, hi in zip(c.dataset, c.ci_hi) if hi < 0]
        # retention curves at 20 % rejection
        ret = load(res, "e01_ranking")
        ret = ret[ret.group.isin(["rank", "rank_ref"]) & ret.metric.isin(["ret_mae_0.00", "ret_mae_0.20"])]
        rr = ret.groupby(["dataset", "method", "metric"]).value.mean().unstack()
        o["retained_mae_at_20pct"] = (rr["ret_mae_0.20"] / rr["ret_mae_0.00"]).groupby("method").mean().to_dict()
        # accuracy per library
        imp = load(res, "e03_extended", "impl")
        o["impl_r2"] = ds_mean(imp, "r2").mean().to_dict()
        o["impl_mean_leaves"] = ds_mean(imp, "mean_leaves_per_tree").mean().round(2).to_dict()
        o["impl_positive"] = (ds_mean(imp, "spearman_LPDS") > 0).sum().astype(int).to_dict()
        # conformal: WLACP against every other interval method
        cpf = cp_frame(res)
        comps = ["Split", "WLACP_submitted", "WLACP_LCP_la_g1", "Mondrian_LPDS4", "LoBoost", "LCP_RBF",
                 "NCP_LPDS", "NCP_ErrModel", "NCP_IBUG", "IBUG_Gaussian", "CQR", "CVplus", "NGBoost100",
                 "NGBoostES"]
        st = pd.concat([compare(cpf, "is_rel", "WLACP_LCP_la_norm_g1", comps, larger_better=False),
                        compare(cpf, "cov_q1_close", "WLACP_LCP_la_norm_g1", comps, larger_better=True)])
        st.to_csv(os.path.join(res, "tables", "wlacp_vs_all.csv"), index=False)
        o["wlacp_vs_all"] = st[["metric", "comparator", "n_datasets", "wins", "mean_diff", "median_diff",
                                "p_two_sided", "p_two_sided_holm", "boot_ci_lo", "boot_ci_hi"]].to_dict("records")
        isr = cpf[cpf.metric == "is_rel"].groupby(["dataset", "method"]).value.mean().unstack()
        o["is_rel_mean"] = isr.mean().to_dict()
        cov = cpf[cpf.metric.isin(["cov_all", "cov_q1", "cov_q4", "cov_knn_q4", "wsc", "cov_low_testq"])]
        o["coverage_means"] = cov.groupby(["metric", "dataset", "method"]).value.mean().groupby(
            ["metric", "method"]).mean().unstack(0).to_dict("index")
        q1 = ds_mean(cpf, "cov_q1")
        w = q1["WLACP_LCP_la_norm_g1"]
        o["wlacp_q1_below_target"] = {DISPLAY.get(k, k): round(float(v), 4) for k, v in w[w < 0.9].items()}
        o["wlacp_q1_min"] = round(float(w.min()), 4)
        o["split_q1_below_target_n"] = int((q1["Split"] < 0.9).sum())
        o["split_q1_min"] = round(float(q1["Split"].min()), 4)
        o["width_q1_rel_split"] = ds_mean(cpf, "width_q1_rel").mean().to_dict()
        o["width_all_rel_split"] = ds_mean(cpf, "width_rel").mean().to_dict()
        o["n_datasets"] = int(len(sp))
        out[tag] = o
    # calibration-size sweep for Ailerons (R2-6(d))
    cs = load(a.results, "e06_calib_size", "calib")
    x = cs[(cs.dataset == "Ailerons")]
    g = x.groupby(["method", "n_cal", "metric"]).value.agg(["mean", "std"])
    out["ailerons_calib"] = {f"{m}|{n}|{k}": [round(float(r["mean"]), 4), round(float(r["std"]), 4)]
                             for (m, n, k), r in g.iterrows() if k in ("cov_all", "cov_q1")}
    # tuned-model sizes
    sens = load(a.results, "e04_sensitivity", "sens")
    nt = sens[sens.metric == "n_trees"].groupby(["method", "dataset"]).value.mean().groupby("method").agg(["mean", "min", "max"])
    out["tuned_n_trees"] = nt.loc[[m for m in nt.index if m.endswith("tuned")]].round(1).to_dict("index")
    path = os.path.join(a.results, "tables", "paper_numbers.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
