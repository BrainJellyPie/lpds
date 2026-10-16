"""LaTeX tables of the revised manuscript, generated from the result records.

  python -m analysis.a05_latex_tables --results results --results_all results_all --out <dir>

Writes one .tex file per table into <dir>; the manuscript inputs them.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from analysis.f01_figures import ds_mean, ds_sd, load
from core.config import DISPLAY, EXTRA10, MAIN9
from core.data import load_dataset

L = "LPM"


def tex(s):
    return s.replace("_", r"\_")


def name(ds):
    return DISPLAY.get(ds, ds).replace("_", " ")


def f3(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"${v:.3f}$"


def fp(p):
    if p < 0.001:
        return r"$<0.001$"
    return f"${p:.3f}$"


def write(out, fname, body):
    with open(os.path.join(out, fname), "w") as f:
        f.write(body.strip() + "\n")


DOMAIN = {"Ailerons": "Aircraft control", "elevators": "Aircraft control", "cpu_act": "Computer systems",
          "superconduct": "Materials science", "Bike_Sharing_Demand": "Urban mobility",
          "houses": "Real estate", "MiamiHousing2016": "Real estate", "pol": r"Telecommunications~\cite{weiss1995rule}",
          "sulfur": "Chemical process", "Brazilian_houses": "Real estate", "abalone": "Marine biology",
          "delays_zurich_transport": "Public transport", "diamonds": "Retail pricing",
          "house_16H": "Census", "house_sales": "Real estate", "medical_charges": "Health-care billing",
          "nyc-taxi-green-dec-2016": "Urban mobility", "wine_quality": "Food chemistry",
          "yprop_4_1": "Anonymized features"}


def t_datasets(a):
    imp = load(a.results_all, "e03_extended", "impl")
    aff = imp[(imp.metric == "active_feature_fraction") & (imp.method == "sklearn")]
    aff = aff.groupby("dataset").value.mean()
    rows = []
    for grp, lst in (("Core", MAIN9), ("Extended", EXTRA10)):
        rows.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{grp} datasets}}}} \\")
        for ds in lst:
            X, y, _ = load_dataset(ds, a.data_dir)
            n = len(y)
            ntr = min(int(0.6 * n), 6000)
            rows.append(f"{name(ds)} & {n} & {ntr} & {X.shape[1]} & {aff.loc[ds]:.2f} & {DOMAIN[ds]} \\\\")
    body = r"""
\begin{table}[!t]
\centering\footnotesize
\caption{Regression tasks with numerical features in the tabular benchmark of Grinsztajn et al.~\cite{grinsztajn2022tree}. The nine core datasets form the main testbed, and the remaining ten extend the error-ranking, conformal and cross-library analyses to the full benchmark. Active is the fraction of features used in at least one split of the default model (mean over five seeds).}\label{tbl:datasets}
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrl@{}}
\toprule
Dataset & Rows & $n_\text{train}$ & $d$ & Active & Domain \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
    write(a.out, "tbl_datasets.tex", body)


def t_spearman(a):
    rank = load(a.results, "e01_ranking", "rank")
    sp, sd = ds_mean(rank, "spearman"), ds_sd(rank, "spearman")
    sp19 = ds_mean(load(a.results_all, "e01_ranking", "rank"), "spearman")
    cols = ["LPDS", "IsoForest", "AA-Range", "Mahalanobis", "kNN", "LOF", "VirtualEns", "IBUG", "ErrModel"]
    head = [L, "IsoF.", "AA-R.", "Mahal.", "kNN", "LOF", "VE", "IBUG", "Err.\\ model"]
    lines = []
    for ds in MAIN9:
        cells = [f"${sp.loc[ds, 'LPDS']:.3f} \\pm {sd.loc[ds, 'LPDS']:.3f}$"]
        cells += [f3(sp.loc[ds, c]) for c in cols[1:]]
        lines.append(f"{name(ds)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("Mean & " + " & ".join(f3(sp[c].mean()) for c in cols) + r" \\")
    lines.append("Median & " + " & ".join(f3(sp[c].median()) for c in cols) + r" \\")
    lines.append("Positive & " + " & ".join(f"{int((sp[c] > 0).sum())}/9" for c in cols) + r" \\")
    lines.append(r"\midrule")
    lines.append("Mean, 19 & " + " & ".join(f3(sp19[c].mean()) for c in cols) + r" \\")
    lines.append("Positive, 19 & " + " & ".join(f"{int((sp19[c] > 0).sum())}/19" for c in cols) + r" \\")
    body = r"""
\begin{table*}[!t]
\centering\footnotesize
\caption{Spearman correlation between each risk score and the absolute error of the deployed GBDT (mean over five seeds; for """ + L + r""" also the standard deviation over seeds). The first seven scores use no labels, with VE the virtual ensemble; IBUG uses the training targets and the error model is fitted on the labeled calibration residuals. The last two rows summarize all 19 regression tasks.}\label{tbl:spearman}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccccc|cc@{}}
\toprule
Dataset & """ + " & ".join(head) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    write(a.out, "tbl_spearman.tex", body)


def t_stats(a):
    s9 = pd.read_csv(os.path.join(a.results, "tables", "stats_dataset_level.csv"))
    s19 = pd.read_csv(os.path.join(a.results_all, "tables", "stats_dataset_level.csv"))
    comps = [("AA-Range", "AA-Range"), ("IsoForest", "Isolation Forest"), ("Mahalanobis", "Mahalanobis"),
             ("kNN", "kNN distance"), ("LOF", "LOF"), ("VirtualEns", "Virtual ensemble"),
             ("AbsPred", r"$|\hat{f}(x)|$"), ("LeafVar", "Leaf target variance"), ("IBUG", "IBUG"),
             ("ErrModel", "Error model")]
    lines = []
    for c, lab in comps:
        def g(df, m):
            return df[(df.metric == m) & (df.reference == "LPDS") & (df.comparator == c)].iloc[0]
        a9, b9 = g(s9, "spearman"), g(s9, "prr_mae")
        a19, b19 = g(s19, "spearman"), g(s19, "prr_mae")
        lines.append(f"{lab} & {int(a9.wins)}/9 & {fp(a9.p_two_sided)} & {int(b9.wins)}/9 & {fp(b9.p_two_sided)} & "
                     f"{int(a19.wins)}/19 & ${a19.mean_diff:+.3f}$ & [${a19.boot_ci_lo:+.3f}$, ${a19.boot_ci_hi:+.3f}$] & "
                     f"{fp(a19.p_two_sided_holm)} & {int(b19.wins)}/19 & {fp(b19.p_two_sided_holm)} \\\\")
        if c == "VirtualEns":
            lines.append(r"\midrule")
    body = r"""
\begin{table*}[!t]
\centering\footnotesize
\caption{Dataset-level comparison of """ + L + r""" with each baseline, with datasets as units (per-dataset means over five seeds). Wins count the datasets on which """ + L + r""" attains the higher Spearman correlation or the higher PRR. $p$ is the two-sided Wilcoxon signed-rank $p$-value; on the 19 tasks it is Holm-adjusted over the ten comparators. Mean $\Delta\rho$ is the mean difference in Spearman correlation between """ + L + r""" and the comparator, and the 95\% CI is its hierarchical bootstrap interval, resampling datasets and then seeds. Baselines below the line use labels or the prediction.}\label{tbl:stats}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc|cccccc@{}}
\toprule
 & \multicolumn{4}{c|}{9 core datasets} & \multicolumn{6}{c}{All 19 datasets} \\
 & \multicolumn{2}{c}{Spearman} & \multicolumn{2}{c|}{PRR} & \multicolumn{4}{c}{Spearman} & \multicolumn{2}{c}{PRR} \\
Comparator & Wins & $p$ & Wins & $p$ & Wins & Mean $\Delta\rho$ & 95\% CI & $p_\text{Holm}$ & Wins & $p_\text{Holm}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    write(a.out, "tbl_stats.tex", body)


def t_auroc(a):
    sh = load(a.results, "e05_shift", "shift")
    x = sh[(sh.method == "all:PC1") & (sh.magnitude == 3.0)]
    m = x.groupby(["dataset", "metric"]).value.mean().unstack()
    s = x.groupby(["dataset", "metric"]).value.std().unstack()
    cols = ["LPDS[sklearn]", "LPDS+Range", "IsoForest", "AA-Range", "Mahalanobis", "kNN", "LOF"]
    head = [L, f"{L} + range", "IsoForest", "AA-Range", "Mahal.", "kNN", "LOF"]
    lines = []
    for ds in MAIN9:
        cells = [f"${m.loc[ds, cols[0]]:.3f} \\pm {s.loc[ds, cols[0]]:.3f}$"] + [f3(m.loc[ds, c]) for c in cols[1:]]
        lines.append(f"{name(ds)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("Mean & " + " & ".join(f3(m[c].mean()) for c in cols) + r" \\")
    body = r"""
\begin{table*}[!t]
\centering\footnotesize
\caption{AUROC for separating held-out test points from validation points shifted by three training PC1 standard deviations along PC1 (mean over five seeds; for """ + L + r""" also the standard deviation). """ + L + r""" + range ranks every point with a feature outside its training range above all in-range points and orders each group by """ + L + r""".}\label{tbl:auroc}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccccc@{}}
\toprule
Dataset & """ + " & ".join(head) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    write(a.out, "tbl_auroc.tex", body)


def t_impl(a):
    imp = load(a.results, "e03_extended", "impl")
    imp19 = load(a.results_all, "e03_extended", "impl")
    sp, sd = ds_mean(imp, "spearman_LPDS"), ds_sd(imp, "spearman_LPDS")
    sp19 = ds_mean(imp19, "spearman_LPDS")
    r2, r219 = ds_mean(imp, "r2"), ds_mean(imp19, "r2")
    libs = ["sklearn", "xgb", "lgbm", "lgbm_d4"]
    lines = []
    for ds in MAIN9:
        lines.append(f"{name(ds)} & " + " & ".join(f"${sp.loc[ds, l]:.3f} \\pm {sd.loc[ds, l]:.3f}$" for l in libs) + r" \\")
    lines.append(r"\midrule")
    lines.append("Mean & " + " & ".join(f3(sp[l].mean()) for l in libs) + r" \\")
    lines.append("Positive & " + " & ".join(f"{int((sp[l] > 0).sum())}/9" for l in libs) + r" \\")
    lines.append("Test $R^2$ & " + " & ".join(f3(r2[l].mean()) for l in libs) + r" \\")
    lines.append(r"\midrule")
    lines.append("Mean, 19 & " + " & ".join(f3(sp19[l].mean()) for l in libs) + r" \\")
    lines.append("Positive, 19 & " + " & ".join(f"{int((sp19[l] > 0).sum())}/19" for l in libs) + r" \\")
    lines.append("Test $R^2$, 19 & " + " & ".join(f3(r219[l].mean()) for l in libs) + r" \\")
    body = r"""
\begin{table*}[!t]
\centering\footnotesize
\caption{Spearman correlation of """ + L + r""" with the absolute error by GBDT library (mean $\pm$ SD over five seeds). LightGBM grows 16 leaves per tree leaf-wise without a depth limit; LightGBM-4 limits the depth to 4.}\label{tbl:impl}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
Dataset & scikit-learn & XGBoost & LightGBM & LightGBM-4 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    write(a.out, "tbl_impl.tex", body)


CPN = [("Split", "Split CP"), ("WLACP_submitted", "Weighted quantile"),
       ("WLACP_LCP_la_g1", "WLACP, raw LA"), ("WLACP_LCP_la_norm_g1", r"\textbf{WLACP}"),
       ("WLACP_LCP_auto", "WLACP, kernel by validation"),
       ("Mondrian_LPDS4", f"Mondrian, {L} quartiles"), ("LoBoost", "LoBoost"),
       ("LCP_RBF", "LCP, RBF kernel"), ("NCP_LPDS", f"NCP, {L} scale"),
       ("NCP_ErrModel", "NCP, error-model scale"), ("NCP_IBUG", "NCP, IBUG scale"),
       ("IBUG_Gaussian", "IBUG, Gaussian"), None,
       ("CQR", "CQR"), ("CVplus", "CV+"), ("NGBoost100", "NGBoost, 100 trees"),
       ("NGBoostES", "NGBoost, early stopping")]


def t_conformal(a):
    pn = json.load(open(os.path.join(a.results, "tables", "paper_numbers.json")))
    c9, c19 = pn["core9"], pn["all19"]
    lines = []
    for item in CPN:
        if item is None:
            lines.append(r"\midrule")
            continue
        k, lab = item
        cm, cm19 = c9["coverage_means"][k], c19["coverage_means"][k]
        cells = [f3(cm["cov_all"]), f3(cm["cov_q1"]), f3(cm["cov_q4"]), f3(cm.get("cov_knn_q4")),
                 f3(cm.get("wsc")), f3(c9["is_rel_mean"][k]), f3(c9["width_all_rel_split"][k]),
                 f3(c9["width_q1_rel_split"][k]),
                 f3(cm19["cov_q1"]), f3(c19["is_rel_mean"][k])]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    body = r"""
\begin{table*}[!t]
\centering\footnotesize
\caption{Prediction intervals at the 90\% level (mean over datasets of five-seed means). Q1 and Q4 are the lowest and highest """ + L + r""" quartiles of the test set with thresholds fixed on the validation split, kNN-Q4 is the quartile of test points farthest from the training set in kNN distance, and WSC is the worst-slab coverage. IS is the interval score, Width the mean width and Width Q1 the mean width on Q1, each divided by that of split CP. Methods below the line refit the model, and kNN-Q4 and WSC are evaluated for the methods that keep it.}\label{tbl:conformal}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccccccc|cc@{}}
\toprule
 & \multicolumn{8}{c|}{9 core datasets} & \multicolumn{2}{c}{All 19} \\
Method & Marginal & Q1 & Q4 & kNN-Q4 & WSC & IS & Width & Width Q1 & Q1 & IS \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    write(a.out, "tbl_conformal.tex", body)


def t_cpstats(a):
    def get(res):
        return pd.read_csv(os.path.join(res, "tables", "wlacp_vs_all.csv"))
    s9, s19 = get(a.results), get(a.results_all)
    order = ["Split", "WLACP_submitted", "WLACP_LCP_la_g1", "Mondrian_LPDS4", "LoBoost", "LCP_RBF", "NCP_LPDS",
             "NCP_ErrModel", "NCP_IBUG", "IBUG_Gaussian", "CQR", "CVplus", "NGBoost100", "NGBoostES"]
    lab = dict((k, v) for k, v in [x for x in CPN if x is not None])
    lab.update({"Mondrian_LPDS4": "Mondrian", "NCP_LPDS": f"NCP, {L}", "NCP_ErrModel": "NCP, error model",
                "LCP_RBF": "LCP, RBF", "NCP_IBUG": "NCP, IBUG", "NGBoost100": "NGBoost, 100",
                "NGBoostES": "NGBoost, ES"})
    lines = []
    for c in order:
        r = {}
        for tag, s in (("9", s9), ("19", s19)):
            for m in ("is_rel", "cov_q1_close"):
                r[(tag, m)] = s[(s.metric == m) & (s.comparator == c)].iloc[0]
        i9, i19, q19 = r[("9", "is_rel")], r[("19", "is_rel")], r[("19", "cov_q1_close")]
        lines.append(f"{lab[c]} & {int(i9.wins)}/9 & {fp(i9.p_two_sided)} & {int(i19.wins)}/19 & "
                     f"${i19.median_diff:+.3f}$ & {fp(i19.p_two_sided_holm)} & {int(q19.wins)}/19 & "
                     f"{fp(q19.p_two_sided_holm)} \\\\")
    body = r"""
\begin{table}[!tb]
\centering\footnotesize
\caption{Dataset-level comparison of WLACP with the other interval methods. Wins count the datasets on which WLACP has the lower interval score (IS) or a Q1 coverage closer to 0.90. Median $\Delta$IS is the comparator's relative interval score minus that of WLACP. $p$ is the two-sided Wilcoxon $p$-value, on the 19 tasks Holm-adjusted over the 14 comparators. NCP is normalized conformal prediction, and NGBoost uses 100 trees or early stopping (ES).}\label{tbl:cpstats}
\setlength{\tabcolsep}{1.6pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccccc@{}}
\toprule
 & \multicolumn{2}{c}{IS, 9} & \multicolumn{3}{c}{IS, 19} & \multicolumn{2}{c}{Q1, 19} \\
Comparator & Wins & $p$ & Wins & Med.\ $\Delta$ & $p_\text{Holm}$ & Wins & $p_\text{Holm}$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
    write(a.out, "tbl_cpstats.tex", body)


def t_mc(a):
    mc = json.load(open(os.path.join(a.results, "e10_coverage_check", "summary.json")))
    ns = sorted(mc, key=int)
    rows = [("split", "Split CP"), ("split_uncorrected", r"Split CP, rank $\lceil n(1-\alpha)\rceil$"),
            ("wlacp_submitted", "WQ, raw LA"),
            ("wlacp_testmass_la_g1", "WQ + test mass, raw LA"),
            ("wlacp_testmass_la_norm_g1", "WQ + test mass, normalized"),
            ("wlacp_lcp_la_g1", "WLACP, raw LA"), ("wlacp_lcp_la_norm_g1", r"\textbf{WLACP}, normalized LA"),
            ("wlacp_lcp_la_g2", r"WLACP, LA$^2$"), ("wlacp_lcp_la_g4", r"WLACP, LA$^4$")]
    lines = []
    for k, lab in rows:
        lines.append(f"{lab} & " + " & ".join(f"${mc[n][k]['mean']:.3f}$" for n in ns) + r" \\")
    se_max = max(mc[n][k]["mc_se"] for n in ns for k, _ in rows)
    body = r"""
\begin{table}[!t]
\centering\footnotesize
\caption{Marginal coverage at the 90\% level in the Monte Carlo study (mean over 2000 independent replications of the calibration and test sets for each calibration-set size $n$; every Monte Carlo standard error is at most $""" + f"{se_max:.4f}" + r"""$). WQ is the weighted quantile.}\label{tbl:mc}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
Procedure & """ + " & ".join(f"$n={n}$" for n in ns) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
    write(a.out, "tbl_mc.tex", body)


def t_timing(a):
    t = json.load(open(os.path.join(a.results, "e09_timing", "timing.json")))
    Ns = sorted(t["by_N"], key=int)
    keys = [("LPDS_lookup", L), ("LPDS_with_leaf_extraction", f"{L} + leaves"), ("predict", "Predict"),
            ("IsoForest", "IsoForest"), ("Mahalanobis", "Mahal."), ("kNN", "kNN"), ("LOF", "LOF"),
            ("IBUG", "IBUG")]
    lines = []
    for n in Ns:
        lines.append(f"{int(n)} & " + " & ".join(f"{t['by_N'][n][k]:.2f}" if t['by_N'][n][k] < 100 else f"{t['by_N'][n][k]:.0f}"
                                                for k, _ in keys) + r" \\")
    nc = sorted(t["by_ncal"], key=int)
    ck = [("WLACP_submitted", "Weighted quantile"), ("WLACP_LCP", "WLACP"), ("Mondrian", "Mondrian"),
          ("LoBoost_fit", "LoBoost fit"), ("LoBoost_assign", "LoBoost assign"), ("LA_matrix_MB", "Kernel (MB)")]
    lines2 = []
    for n in nc:
        lines2.append(f"{int(n)} & " + " & ".join(f"{t['by_ncal'][n][k]:.2f}" if t['by_ncal'][n][k] < 100 else f"{t['by_ncal'][n][k]:.0f}"
                                                 for k, _ in ck) + r" \\")
    body = r"""
\begin{table*}[!t]
\centering\footnotesize
\caption{Median time in milliseconds to process 400 queries on Superconduct with $T=100$ trees. The upper part gives the risk scores against the training-set size $N$, with """ + L + r""" timed from the leaf indices alone and including their extraction, and the lower part gives the interval construction against the calibration-set size $n$, with the memory of the query-calibration kernel matrix.}\label{tbl:timing}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rcccccccc@{}}
\toprule
$N$ & """ + " & ".join(h for _, h in keys) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\midrule
$n$ & """ + " & ".join(h for _, h in ck) + r""" & & \\
\midrule
""" + "\n".join(l.replace(r" \\", r" & & \\") for l in lines2) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
    write(a.out, "tbl_timing.tex", body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--results_all", default="results_all")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (t_datasets, t_spearman, t_stats, t_auroc, t_impl, t_conformal, t_cpstats, t_mc, t_timing):
        f(a)
        print("table", f.__name__)


if __name__ == "__main__":
    main()
