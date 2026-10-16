"""Dataset-level statistics (R2-3).

For the Spearman correlation and the PRR of e01, and for the low-quartile
coverage of e02:
  * Wilcoxon signed-rank tests on per-dataset means (datasets as units),
    one-sided in the direction fixed before the analysis (LPDS above the
    comparator) and two-sided, with Holm adjustment over comparators;
  * hierarchical bootstrap CI of the grand-mean difference (resample
    datasets, then seeds within each dataset);
  * linear mixed model on the per-(dataset, seed) differences with a random
    intercept for dataset;
  * per-dataset bootstrap CI of the LPDS Spearman correlation obtained by
    resampling test points within each seed (from the stored arrays).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from core.config import DISPLAY

COMPARATORS = ["IsoForest", "AA-Range", "Mahalanobis", "kNN", "LOF", "LeafVar",
               "AbsPred", "VirtualEns", "IBUG", "ErrModel"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (len(p) - i) * p[k])
        adj[k] = min(running, 1.0)
    return adj


def hier_boot(diff, B=10000, seed=0):
    """diff: DataFrame with columns dataset, seed, value."""
    rng = np.random.default_rng(seed)
    groups = [g.value.to_numpy() for _, g in diff.groupby("dataset")]
    D = len(groups)
    stats = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, D, D)
        stats[b] = np.mean([groups[i][rng.integers(0, len(groups[i]), len(groups[i]))].mean()
                            for i in pick])
    return np.percentile(stats, [2.5, 97.5])


def mixed(diff):
    try:
        import statsmodels.formula.api as smf
        m = smf.mixedlm("value ~ 1", diff, groups=diff["dataset"]).fit(reml=True)
        ci = m.conf_int().loc["Intercept"].to_numpy()
        return float(m.params["Intercept"]), float(ci[0]), float(ci[1]), float(m.pvalues["Intercept"])
    except Exception:  # pragma: no cover
        return (np.nan,) * 4


def compare(df, metric, ref, comps, larger_better=True):
    rows = []
    piv = df[df.metric == metric].pivot_table(index=["dataset", "seed"], columns="method",
                                              values="value")
    dm = piv.groupby("dataset").mean()
    for c in comps:
        if c not in piv:
            continue
        d = (piv[ref] - piv[c]) * (1 if larger_better else -1)
        diff = d.rename("value").reset_index()
        dd = (dm[ref] - dm[c]) * (1 if larger_better else -1)
        w1 = wilcoxon(dd, alternative="greater") if np.any(dd != 0) else None
        w2 = wilcoxon(dd) if np.any(dd != 0) else None
        lo, hi = hier_boot(diff)
        est, mlo, mhi, mp = mixed(diff)
        rows.append(dict(metric=metric, reference=ref, comparator=c,
                         n_datasets=len(dd), wins=int((dd > 0).sum()),
                         mean_diff=dd.mean(), median_diff=dd.median(),
                         W=w1.statistic if w1 else np.nan,
                         p_one_sided=w1.pvalue if w1 else np.nan,
                         p_two_sided=w2.pvalue if w2 else np.nan,
                         boot_ci_lo=lo, boot_ci_hi=hi,
                         mixed_est=est, mixed_ci_lo=mlo, mixed_ci_hi=mhi, mixed_p=mp))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_one_sided_holm"] = holm(out.p_one_sided.fillna(1))
        out["p_two_sided_holm"] = holm(out.p_two_sided.fillna(1))
    return out


def spearman_ci(results, B=1000, seed=0):
    rows = []
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(results, "e01_ranking", "*.npz")))
    by_ds = {}
    for f in files:
        ds = os.path.basename(f).rsplit("_s", 1)[0]
        by_ds.setdefault(ds, []).append(np.load(f))
    for ds, zs in by_ds.items():
        methods = [k.split("__", 1)[1] for k in zs[0].files if k.startswith("risk__")]
        for m in methods:
            vals = np.empty(B)
            prepared = [(rankdata(z[f"risk__{m}"]), rankdata(z["err"])) for z in zs]
            for b in range(B):
                acc = []
                for rs_, re_ in prepared:
                    i = rng.integers(0, len(rs_), len(rs_))
                    a_, b_ = rankdata(rs_[i]), rankdata(re_[i])
                    acc.append(0.0 if a_.std() == 0 else np.corrcoef(a_, b_)[0, 1])
                vals[b] = np.mean(acc)
            point = np.mean([0.0 if a.std() == 0 else np.corrcoef(a, e)[0, 1]
                             for a, e in prepared])
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append(dict(dataset=DISPLAY.get(ds, ds), method=m, spearman=point,
                             ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def load(results, exp, group):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, exp, "*.json"))):
        rows.extend(r for r in json.load(open(f))["records"] if r["group"] == group)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    out_dir = os.path.join(a.results, "tables")
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    rank = load(a.results, "e01_ranking", "rank")
    if len(rank):
        for metric in ("spearman", "prr_mae"):
            parts.append(compare(rank, metric, "LPDS", COMPARATORS))
    cp = load(a.results, "e02_conformal", "cp")
    if len(cp):
        cp = cp.assign(value=np.where(cp.metric.str.startswith("cov_"),
                                      -np.abs(cp.value - 0.9), cp.value))
        # interval score in units of the split-CP interval score of the same
        # (dataset, seed), so that datasets with different target scales are
        # comparable
        isc = cp[cp.metric == "is_all"]
        ref = isc[isc.method == "Split"].set_index(["dataset", "seed"]).value
        cp.loc[isc.index, "value"] = isc.value.to_numpy() / ref.reindex(
            list(zip(isc.dataset, isc.seed))).to_numpy()
        for ref in [m for m in cp.method.unique() if m.startswith("WLACP_LCP")]:
            parts.append(compare(cp, "cov_q1", ref, ["Split", "Mondrian_LPDS4", "LoBoost",
                                                     "LCP_RBF", "NCP_LPDS"]))
            parts.append(compare(cp, "is_all", ref, ["Split", "Mondrian_LPDS4", "LoBoost",
                                                     "LCP_RBF", "NCP_LPDS"],
                                 larger_better=False))
    if parts:
        res = pd.concat(parts, ignore_index=True)
        res.to_csv(os.path.join(out_dir, "stats_dataset_level.csv"), index=False)
        with open(os.path.join(out_dir, "stats_dataset_level.md"), "w") as f:
            f.write("cov_q1 comparisons use -|coverage - 0.9| (closer to target is "
                    "better); is_all is the interval score relative to split CP "
                    "(lower is better).\n\n")
            f.write(res.round(4).to_markdown(index=False))
        print(res.round(4).to_string())
    if a.boot > 0 and glob.glob(os.path.join(a.results, "e01_ranking", "*.npz")):
        ci = spearman_ci(a.results, B=a.boot)
        ci.to_csv(os.path.join(out_dir, "spearman_bootstrap_ci.csv"), index=False)
        with open(os.path.join(out_dir, "spearman_bootstrap_ci.md"), "w") as f:
            f.write(ci.round(4).to_markdown(index=False))


if __name__ == "__main__":
    main()
