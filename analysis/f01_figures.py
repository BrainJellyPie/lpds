"""All figures of the paper, drawn from the stored result records.

  python -m analysis.f01_figures --results results --results_all results_all --out figures

Each figure is written as a 600-dpi PNG together with a .txt file holding
its caption, which the paper uses verbatim.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

from analysis import figstyle as S
from core.config import DISPLAY, MAIN9

NAME = "LPM"     # name of the leaf-population score in the paper


def load(res, exp, group=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(res, exp, "*.json"))):
        d = json.load(open(f))
        if "records" in d:
            rows.extend(r for r in d["records"] if group is None or r["group"] == group)
    return pd.DataFrame(rows)


def ds_mean(df, metric, by=("dataset", "method")):
    x = df[df.metric == metric]
    return x.groupby(list(by)).value.mean().unstack()


def ds_sd(df, metric):
    x = df[df.metric == metric]
    return x.groupby(["dataset", "method"]).value.std().unstack()


def short(ds):
    return DISPLAY.get(ds, ds).replace("_", " ")


def key(m):
    return NAME if m == "LPDS" else m


def new_fig(h, ratios):
    fig = plt.figure(figsize=(S.W, h))
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios)
    return fig, [fig.add_subplot(gs[i]) for i in range(len(ratios))]


def rotate(ax, deg=32):
    for t in ax.get_xticklabels():
        t.set_rotation(deg)
        t.set_ha("right")
        t.set_rotation_mode("anchor")


def stripe(ax, k, horizontal=True):
    if horizontal:
        ax.axhspan(k - 0.5, k + 0.5, color="#f6f6f4", zorder=0, lw=0)
    else:
        ax.axvspan(k - 0.5, k + 0.5, color="#f6f6f4", zorder=0, lw=0)


# --------------------------------------------------------------------------
def fig_ranking(R, RA, out):
    rank = load(R, "e01_ranking", "rank")
    sp, sd = ds_mean(rank, "spearman"), ds_sd(rank, "spearman")
    order = sp["LPDS"].sort_values(ascending=False).index.tolist()
    fig, (ax, bx, cx) = new_fig(2.95, [1.45, 1.0, 1.08])
    leg = ["LPDS", "IsoForest", "AA-Range", "Mahalanobis", "kNN", "LOF", "VirtualEns", "IBUG", "ErrModel"]
    S.fig_legend(fig, [S.handle(S.COLORS[key(m)], S.MARKERS[key(m)]) for m in leg],
                 [S.LABELS[key(m)] for m in leg], ncol=9, size=7.0, spacing=0.85)

    # (a) per-dataset Spearman
    meths = ["LPDS", "IsoForest", "AA-Range", "Mahalanobis", "kNN"]
    offs = S.jitter_offsets(len(meths), 0.64)
    x = np.arange(len(order))
    for i in range(0, len(order), 2):
        stripe(ax, i, horizontal=False)
    for k, m in enumerate(meths):
        c = S.COLORS[key(m)]
        mu, s = sp.loc[order, m].to_numpy(), sd.loc[order, m].to_numpy()
        xx = x + offs[k]
        ax.vlines(xx, mu - s, mu + s, color=c, lw=1.3, alpha=0.45, zorder=3)
        S.dot(ax, xx, mu, c, S.MARKERS[key(m)], s=40 if m == "LPDS" else 24,
              z=5 if m == "LPDS" else 4)
    ax.axhline(0, color=S.INK2, lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([short(d) for d in order])
    rotate(ax, 38)
    ax.set_ylabel("Spearman ρ with |error|")
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.grid(axis="x", visible=False)
    S.panel(ax, "a", "Error ranking per dataset")

    # (b) rejection curves
    ret = load(R, "e01_ranking")
    ret = ret[ret.group.isin(["rank", "rank_ref"]) & ret.metric.str.startswith("ret_mae")]
    rr = ret.groupby(["dataset", "method", "metric"]).value.mean().reset_index()
    base = rr[rr.metric == "ret_mae_0.00"].set_index(["dataset", "method"]).value
    rr["rel"] = rr.value.to_numpy() / base.reindex(list(zip(rr.dataset, rr.method))).to_numpy()
    curve = rr.groupby(["method", "metric"]).rel.mean().unstack()
    fr = np.array([float(c.split("_")[-1]) for c in curve.columns])
    keep = fr <= 0.9 + 1e-9
    fr = fr[keep]
    series = [("random", "Random", S.NEUTRAL, None), ("oracle", "Ideal ordering", S.INK, None),
              ("IsoForest", "Isolation Forest", S.COLORS["IsoForest"], "s"),
              ("VirtualEns", "Virtual ensemble", S.COLORS["VirtualEns"], "h"),
              ("kNN", "kNN distance", S.COLORS["kNN"], "D"),
              ("ErrModel", "Error model", S.COLORS["ErrModel"], "X"),
              ("IBUG", "IBUG", S.COLORS["IBUG"], "P"),
              ("LPDS", NAME, S.COLORS["LPM"], "o")]
    ends, texts, cols, wts = [], [], [], []
    for m, lab, c, mk in series:
        yv = curve.loc[m].to_numpy()[keep]
        if m == "random":
            bx.plot(fr, yv, color=c, lw=1.2, ls=(0, (4, 3)), zorder=2)
        elif m == "oracle":
            bx.plot(fr, yv, color=c, lw=1.2, ls=(0, (1, 1.8)), zorder=2)
        else:
            main = m == "LPDS"
            S.glow_line(bx, fr, yv, c, lw=2.0 if main else 1.35, glow=main, zorder=5 if main else 3)
            S.dot(bx, fr[::3], yv[::3], c, mk, s=22 if main else 13, z=6 if main else 4, ring=0.8)
        ends.append(yv[-1])
        texts.append(lab)
        cols.append(c)
        wts.append("bold" if m == "LPDS" else "medium")
    bx.set_xlim(0, 1.3)
    bx.set_ylim(0, 1.06)
    bx.spines["bottom"].set_bounds(0, 0.9)
    bx.set_xticks([0, 0.3, 0.6, 0.9])
    bx.set_xticklabels(["0", "0.3", "0.6", "0.9"])
    bx.set_xlabel("Fraction of predictions rejected")
    bx.set_ylabel("Retained MAE / MAE of all")
    S.panel(bx, "b", "Rejection curves, 9 core datasets")
    S.spread_labels(bx, 0.9, ends, texts, cols, x_text=0.735, gap_pt=8.2, size=7.2, weights=wts)

    # (c) benchmark-wide paired differences
    r19 = ds_mean(load(RA, "e01_ranking", "rank"), "spearman")
    comps = ["AA-Range", "IsoForest", "Mahalanobis", "LOF", "kNN", "VirtualEns"]
    for k, c in enumerate(comps):
        d = (r19["LPDS"] - r19[c]).to_numpy()
        wins = int((d > 0).sum())
        if k % 2 == 0:
            stripe(cx, k)
        yj = k + np.random.default_rng(k).uniform(-0.22, 0.22, len(d))
        cx.scatter(d, yj, s=14, color=S.COLORS[c], alpha=0.85, edgecolor=S.SURF,
                   linewidth=0.5, zorder=3, marker=S.MARKERS[c])
        cx.plot([np.median(d)] * 2, [k - 0.36, k + 0.36], color=S.INK, lw=1.8, zorder=4,
                solid_capstyle="butt")
        cx.annotate(f"{wins}/{len(d)}", (1.0, k), xycoords=("axes fraction", "data"),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=7.4, color=S.INK, fontweight="semibold")
    cx.annotate("wins", (1.0, -0.8), xycoords=("axes fraction", "data"), xytext=(5, 0),
                textcoords="offset points", ha="left", va="center", fontsize=7.0, color=S.MUTED)
    cx.axvline(0, color=S.INK2, lw=0.8, zorder=2)
    cx.set_yticks(range(len(comps)))
    cx.set_yticklabels([S.LABELS[c] for c in comps])
    cx.set_ylim(len(comps) - 0.5, -1.1)
    cx.set_xlabel(f"ρ({NAME}) − ρ(baseline)")
    cx.grid(axis="y", visible=False)
    S.panel(cx, "c", f"Paired gaps, {len(r19)} tasks")
    cap = (f"Deployment error ranking. (a) Spearman correlation between each risk score and the absolute error "
           f"of the deployed GBDT on the nine core datasets (mean and standard deviation over five seeds), "
           f"sorted by {NAME}. (b) Mean absolute error of the retained predictions after rejecting the "
           f"highest-risk fraction, divided by the error with no rejection and averaged over the nine datasets; "
           f"the ideal ordering rejects by the true error. (c) Per-dataset difference between the Spearman "
           f"correlation of {NAME} and that of each label-free baseline on all {len(r19)} regression tasks of "
           f"the benchmark; the black tick marks the median and the right column counts the datasets on which "
           f"{NAME} attains the higher correlation.")
    S.save(fig, os.path.join(out, "fig_ranking.png"), cap)


# --------------------------------------------------------------------------
def fig_quartile(R, RA, out):
    q = load(R, "e01_ranking", "quartile")
    q = q[q.method == "testq"]
    m = q.groupby(["dataset", "metric"]).value.mean().unstack()
    acc = ds_mean(load(R, "e01_ranking", "acc"), "mae")["GBDT"]
    order = m.ratio_q1_q4.sort_values(ascending=False).index.tolist()
    fig, (ax, bx) = new_fig(2.75, [2.25, 1.0])
    cols = [S.BLUE_RAMP[1], S.BLUE_RAMP[4], S.BLUE_RAMP[8], S.BLUE_RAMP[12]]
    w = 0.19
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(0, 2.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short(d) for d in order])
    rotate(ax, 28)
    ax.set_ylabel("Quartile MAE / test MAE")
    ax.grid(axis="x", visible=False)
    S.panel(ax, "a", f"Error by {NAME} quartile, 9 core datasets")
    for k in range(4):
        ax.scatter([], [], marker="s", s=42, color=cols[k],
                   label=[f"Q1 (lowest {NAME})", "Q2", "Q3", f"Q4 (highest {NAME})"][k])
    ax.legend(ncol=4, loc="upper right", handletextpad=0.2, columnspacing=1.0)
    fig.canvas.draw()
    for i, ds in enumerate(order):
        vals = [m.loc[ds, f"mae_q{k}"] / acc.loc[ds] for k in range(1, 5)]
        for k, v in enumerate(vals):
            S.rbar(ax, i + (k - 1.5) * (w + 0.015), v, w, cols[k], frac=0.32)
        S.label(ax, i, max(vals), f"{m.loc[ds, 'ratio_q1_q4']:.2f}×", S.INK, dx=0, dy=7,
                ha="center", size=7.4, weight="semibold")
    ax.axhline(1, color=S.INK2, lw=0.8, ls=(0, (3, 3)), zorder=2)

    qa = load(RA, "e01_ranking", "quartile")
    qa = qa[(qa.method == "testq") & (qa.metric == "ratio_q1_q4")]
    ra = qa.groupby("dataset").value.mean().sort_values()
    y = np.arange(len(ra))
    v = ra.to_numpy()
    core = [d in MAIN9 for d in ra.index]
    for yy, vv, cc in zip(y, v, core):
        col = S.BLUE if vv > 1 else S.ORANGE
        bx.plot([1, vv], [yy, yy], color=col, lw=1.6, alpha=0.35, zorder=2, solid_capstyle="butt")
        S.dot(bx, [vv], [yy], col, "o", s=26 if cc else 20, z=4)
    bx.axvline(1, color=S.INK2, lw=0.8)
    bx.set_yticks(y)
    bx.set_yticklabels([short(d) for d in ra.index], fontsize=6.7)
    for t, cc in zip(bx.get_yticklabels(), core):
        t.set_color(S.INK if cc else S.MUTED)
        t.set_fontweight("semibold" if cc else "normal")
    bx.set_xscale("log")
    S.log_ticks(bx.xaxis, [0.75, 1, 1.5, 2, 3], ["0.75", "1", "1.5", "2", "3"])
    bx.set_xlim(0.62, 4.2)
    bx.set_ylim(-0.7, len(ra) - 0.3)
    bx.set_xlabel("Q1 / Q4 MAE ratio")
    bx.grid(axis="y", visible=False)
    S.panel(bx, "b", f"All {len(ra)} tasks")
    cap = (f"Prediction error across {NAME} quartiles of the test set. (a) Mean absolute error in each quartile "
           f"divided by the mean absolute error of the whole test set on the nine core datasets (mean over five seeds); the "
           f"number above each group is the ratio between the lowest and the highest quartile. (b) The same ratio "
           f"on all {len(ra)} regression tasks of the benchmark on a logarithmic axis, with the nine core "
           f"datasets in bold.")
    S.save(fig, os.path.join(out, "fig_quartile.png"), cap)


# --------------------------------------------------------------------------
def fig_partition(R, RA, out):
    sens = load(R, "e04_sensitivity", "sens")
    x = sens.groupby(["dataset", "method", "metric"]).value.mean()
    mean = x.groupby(["method", "metric"]).mean().unstack()
    pos = (x.xs("spearman_LPDS", level="metric").unstack("method") > 0).sum()
    fig, (ax, bx, cx) = new_fig(2.9, [1.25, 0.95, 1.12])

    # (a) tile matrix
    rows = [("Trees", "n_estimators", [50, 100, 300, 500], 100),
            ("Depth", "max_depth", [2, 4, 6, 8], 4),
            ("Min. leaf", "min_samples_leaf", [5, 20, 50, 100], 20),
            ("Learn. rate", "learning_rate", [0.02, 0.05, 0.1, 0.2], 0.05)]
    norm = Normalize(0.05, 0.27)
    cmap = LinearSegmentedColormap.from_list("b", S.BLUE_RAMP)

    def tile(i, j, meth, top, default=False, width=0.9, top_size=6.6):
        v = mean.loc[meth, "spearman_LPDS"]
        col = cmap(norm(v))
        ax.add_patch(FancyBboxPatch((j - width / 2, i - 0.42), width, 0.84,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=col, edgecolor=S.INK if default else "none",
                                    lw=1.1 if default else 0, zorder=2))
        dark = norm(v) > 0.55
        ax.text(j, i - 0.21, top, ha="center", va="center", fontsize=top_size,
                color="white" if dark else S.INK2, zorder=3)
        ax.text(j, i + 0.05, f"{v:.3f}", ha="center", va="center", fontsize=8.2,
                color="white" if dark else S.INK, fontweight="semibold", zorder=3)
        ax.text(j, i + 0.28, f"{pos[meth]}/9 > 0", ha="center", va="center", fontsize=6.0,
                color="white" if dark else S.INK2, zorder=3)

    for i, (gname, k_, vals, base) in enumerate(rows):
        for j, v in enumerate(vals):
            meth = "sklearn:base" if v == base else f"sklearn:{k_}={v:g}"
            tile(i, j, meth, f"{v:g}", default=(v == base))
    # the three tuned libraries share the width of the four columns above
    for x_, (meth, lab) in zip([0.17, 1.5, 2.83], [("sklearn:tuned", "scikit-learn"), ("xgb:tuned", "XGBoost"),
                                                   ("lgbm:tuned", "LightGBM")]):
        tile(4, x_, meth, lab, width=1.24, top_size=6.2)
    ax.set_xlim(-0.52, 3.52)
    ax.set_ylim(4.5, -0.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels([r[0] for r in rows] + ["Tuned"])
    for t in ax.get_yticklabels():
        t.set_fontweight("semibold")
        t.set_color(S.INK)
    ax.set_xticks([])
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    S.panel(ax, "a", f"Mean ρ of {NAME}")

    # (b) leaf-wise LightGBM
    leaves = [8, 16, 31, 63, 127]
    ends, texts, colors, wts = [], [], [], []
    for m, lab, col, mk in [("spearman_IsoForest", "Isolation Forest", S.ORANGE, "s"),
                            ("spearman_kNN", "kNN distance", S.AQUA, "D"),
                            ("spearman_LPDS", NAME, S.BLUE, "o")]:
        yv = [mean.loc[f"lgbm:num_leaves={nl}", m] for nl in leaves]
        main = "LPDS" in m
        S.glow_line(bx, leaves, yv, col, lw=2.0 if main else 1.4, glow=main, zorder=5 if main else 3)
        S.dot(bx, leaves, yv, col, mk, s=28 if main else 18, z=6)
        if main:
            for nl, v in zip(leaves, yv):
                S.label(bx, nl, v, f"{v:.2f}", S.INK, dx=0 if nl > 8 else 4, dy=8 if nl > 8 else -9,
                        ha="center" if nl > 8 else "left", size=6.8)
        ends.append(yv[-1])
        texts.append(lab)
        colors.append(col)
        wts.append("bold" if main else "medium")
    bx.set_xscale("log", base=2)
    S.log_ticks(bx.xaxis, leaves, [str(v) for v in leaves])
    bx.set_xlim(6.5, 600)
    bx.spines["bottom"].set_bounds(8, 127)
    bx.set_ylim(-0.02, 0.48)
    bx.set_xlabel("Leaves per tree")
    bx.set_ylabel("Mean Spearman ρ")
    S.panel(bx, "b", "LightGBM, leaf-wise growth")
    S.spread_labels(bx, 127, ends, texts, colors, x_text=0.79, gap_pt=8.5, size=7.0, weights=wts)

    # (c) libraries on all datasets
    imp = ds_mean(load(RA, "e03_extended", "impl"), "spearman_LPDS")
    order = imp["lgbm"].sort_values().index.tolist()
    y = np.arange(len(order))
    for yy, d in zip(y, order):
        a_, b_ = imp.loc[d, "sklearn"], imp.loc[d, "lgbm"]
        cx.plot([a_, b_], [yy, yy], color=S.BLUE if b_ > a_ else S.ORANGE, lw=1.8, alpha=0.3,
                zorder=2, solid_capstyle="butt")
    S.dot(cx, imp.loc[order, "sklearn"].to_numpy(), y, S.NEUTRAL, "o", s=20, label="scikit-learn, depth 4")
    S.dot(cx, imp.loc[order, "lgbm_d4"].to_numpy(), y, S.YELLOW, "^", s=20, label="LightGBM, depth 4")
    S.dot(cx, imp.loc[order, "lgbm"].to_numpy(), y, S.BLUE, "o", s=28, z=5, label="LightGBM, leaf-wise")
    cx.axvline(0, color=S.INK2, lw=0.8)
    cx.set_yticks(y)
    cx.set_yticklabels([short(d) for d in order], fontsize=6.6)
    core = [d in MAIN9 for d in order]
    for t, cc in zip(cx.get_yticklabels(), core):
        t.set_color(S.INK if cc else S.MUTED)
        t.set_fontweight("semibold" if cc else "normal")
    cx.set_ylim(-0.7, len(order) - 0.3)
    cx.set_xlim(-0.22, 0.95)
    cx.set_xticks([0, 0.2, 0.4])
    cx.set_xlabel(f"Spearman ρ of {NAME}")
    cx.grid(axis="y", visible=False)
    cx.legend(loc="lower right", fontsize=6.8, handletextpad=0.1, borderaxespad=0.15, labelspacing=0.3)
    S.panel(cx, "c", f"Libraries, all {len(order)} tasks")
    cap = (f"Dependence of {NAME} on the fitted partition. (a) Mean Spearman correlation between {NAME} and the "
           f"absolute error over the nine core datasets when one scikit-learn hyperparameter departs from the "
           f"default configuration, and for models tuned on the validation split; the bottom line of each tile "
           f"counts the datasets with a positive correlation. (b) LightGBM with leaf-wise growth as the number "
           f"of leaves per tree increases, with two ambient-space scores evaluated against the errors of the "
           f"same models. (c) Per-dataset correlation of {NAME} under scikit-learn, LightGBM with leaf-wise "
           f"growth and LightGBM limited to depth 4 on all {len(order)} regression tasks (mean over five "
           f"seeds), with the nine core datasets in bold.")
    S.save(fig, os.path.join(out, "fig_partition.png"), cap)


# --------------------------------------------------------------------------
def fig_shift(R, out):
    sh = load(R, "e05_shift", "shift")
    x = sh.groupby(["method", "magnitude", "metric", "dataset"]).value.mean()
    g = x.groupby(["method", "magnitude", "metric"]).mean()
    mags = sorted(sh.magnitude.unique())
    fig, (ax, bx, cx) = new_fig(2.85, [1.3, 1.0, 0.95])
    series = [("LPDS[sklearn]", NAME, S.BLUE, "o"), ("LPDS+Range", f"{NAME} + range check", S.VIOLET, "P"),
              ("IsoForest", "Isolation Forest", S.ORANGE, "s"), ("AA-Range", "AA-Range", S.YELLOW, "^"),
              ("kNN", "kNN distance", S.AQUA, "D"), ("Mahalanobis", "Mahalanobis", S.MAGENTA, "v")]
    S.fig_legend(fig, [S.handle(c, mk) for _, _, c, mk in series], [lab for _, lab, _, _ in series], ncol=6)
    for k_, lab, col, mk in series[::-1]:
        yv = [g.loc[("all:PC1", mm, k_)] for mm in mags]
        main = k_ == "LPDS[sklearn]"
        S.glow_line(ax, mags, yv, col, lw=2.0 if main else 1.3, glow=main, zorder=5 if main else 3)
        S.dot(ax, mags, yv, col, mk, s=24 if main else 14, z=6 if main else 4, ring=0.8)
    ax.set_xscale("log")
    S.log_ticks(ax.xaxis, [0.5, 1, 3, 10, 30, 100, 300])
    ax.set_xlabel("Shift along PC1 (training SD)")
    ax.set_ylabel("AUROC, shifted vs test")
    ax.set_ylim(0.5, 1.02)
    ax.axvspan(30, 330, color="#f3f6fb", zorder=0, lw=0)
    ax.text(100, 0.53, f"{NAME} plateau", ha="center", va="bottom", fontsize=7.0, color=S.INK2)
    S.panel(ax, "a", "Detection as the shift grows")

    ends, texts, colors = [], [], []
    for d, col, mk in [("PC1", S.BLUE_RAMP[7], "o"), ("PC2", S.BLUE_RAMP[4], "s"),
                       ("PC3", S.BLUE_RAMP[10], "D"), ("Diag", S.BLUE_RAMP[12], "^")]:
        yv = [g.loc[(f"all:{d}", mm, "mean_lpds_ratio")] for mm in mags]
        S.glow_line(bx, mags, yv, col, lw=1.6, glow=False)
        S.dot(bx, mags, yv, col, mk, s=16, ring=0.8)
        ends.append(yv[-1])
        texts.append(d)
        colors.append(col)
    bx.axhline(1, color=S.INK2, lw=0.8, ls=(0, (3, 3)))
    bx.set_xscale("log")
    S.log_ticks(bx.xaxis, [0.5, 1, 3, 10, 30, 300])
    bx.set_xlim(0.4, 1500)
    bx.spines["bottom"].set_bounds(0.5, 300)
    bx.set_ylim(0, 1.08)
    bx.set_xlabel("Shift (training PC1 SD)")
    bx.set_ylabel(f"Mean {NAME}, shifted / test")
    S.panel(bx, "b", f"{NAME} saturates")
    S.spread_labels(bx, 300, ends, texts, colors, x_text=0.86, gap_pt=8.0, size=7.2)

    dirs = ["PC1", "PC2", "PC3", "Diag"]
    trio = [("LPDS[sklearn]", S.BLUE, "o"), ("LPDS+Range", S.VIOLET, "P"), ("IsoForest", S.ORANGE, "s")]
    offs = [-0.24, 0, 0.24]
    for j, d in enumerate(dirs):
        if j % 2 == 0:
            stripe(cx, j, horizontal=False)
        for (k_, col, mk), o in zip(trio, offs):
            v = g.loc[(f"all:{d}", 3.0, k_)]
            cx.plot([j + o, j + o], [0.5, v], color=col, lw=2.2, alpha=0.35, zorder=2, solid_capstyle="butt")
            S.dot(cx, [j + o], [v], col, mk, s=30 if k_.startswith("LPDS[") else 22, z=5)
            if k_ == "LPDS[sklearn]":
                S.label(cx, j + o, 0.5, f"{v:.2f}", S.INK, dx=0, dy=6, ha="center", size=6.6, weight="semibold",
                        va="bottom")
    cx.set_xticks(range(len(dirs)))
    cx.set_xticklabels(dirs)
    cx.set_xlim(-0.5, len(dirs) - 0.5)
    cx.set_ylim(0.5, 1.02)
    cx.set_ylabel("AUROC at 3 SD")
    cx.grid(axis="x", visible=False)
    S.panel(cx, "c", "Direction, equal length")
    cap = (f"Detection of synthetic extrapolation, averaged over the nine core datasets and five seeds. "
           f"(a) AUROC between held-out test points and validation points shifted along the first principal "
           f"component (PC1) of the standardized training features, as a function of the shift in units of the "
           f"training PC1 standard deviation. (b) Mean {NAME} of the shifted points divided by the mean {NAME} of "
           f"the test points for four shift directions. (c) AUROC at a shift of three PC1 standard deviations "
           f"for each direction, with the same displacement length for all directions; the value of {NAME} is "
           f"printed.")
    S.save(fig, os.path.join(out, "fig_shift.png"), cap)


# --------------------------------------------------------------------------
def fig_heatmaps(R, out):
    sh = load(R, "e05_shift", "shift")
    sh = sh[(sh.magnitude == 3.0) & sh.method.str.startswith("all:")]
    x = sh.groupby(["dataset", "method", "metric"]).value.mean()
    dirs = ["PC1", "PC2", "PC3", "Diag"]
    order = list(MAIN9)
    norm = Normalize(0.4, 1.0)
    cmap = LinearSegmentedColormap.from_list("b", ["#f2f6fc"] + S.BLUE_RAMP[1:])
    fig = plt.figure(figsize=(S.W, 2.6))
    gs = fig.add_gridspec(1, 3)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for p, (lib, lab) in enumerate([("sklearn", "scikit-learn"), ("xgb", "XGBoost"), ("lgbm", "LightGBM")]):
        ax = axes[p]
        M = np.array([[x.loc[(ds, f"all:{d}", f"LPDS[{lib}]")] for d in dirs] for ds in order])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.add_patch(FancyBboxPatch((j - 0.45, i - 0.42), 0.9, 0.84,
                                            boxstyle="round,pad=0,rounding_size=0.14",
                                            facecolor=cmap(norm(M[i, j])), edgecolor="none"))
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.2,
                        color="white" if norm(M[i, j]) > 0.55 else S.INK, fontweight="medium")
        ax.set_xlim(-0.5, len(dirs) - 0.5)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xticks(range(len(dirs)))
        ax.set_xticklabels(dirs)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([short(d) for d in order] if p == 0 else [])
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(rf"$\mathbf{{{'abc'[p]}}}$   {lab}  ·  mean {M.mean():.3f}", loc="left",
                     fontsize=9.0, fontweight="semibold", pad=16)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, shrink=0.85, aspect=28, pad=0.015)
    cb.outline.set_visible(False)
    cb.set_label("AUROC", color=S.INK2)
    cb.ax.tick_params(labelsize=7.2)
    cap = (f"AUROC of {NAME} for synthetic extrapolation at three training PC1 standard deviations along four "
           f"directions with equal displacement length, for (a) scikit-learn, (b) XGBoost and (c) LightGBM "
           f"ensembles on the nine core datasets (mean over five seeds).")
    S.save(fig, os.path.join(out, "fig_heatmaps.png"), cap)


# --------------------------------------------------------------------------
CP = {"Split": ("Split CP", S.NEUTRAL, "o"),
      "Split_uncorrected": ("Split CP, rank n(1−α)", S.PALE, "o"),
      "WQ": ("Weighted quantile", S.LIGHT_BLUE, "s"),
      "WLACP_LCP_la_g1": ("WLACP, raw LA kernel", S.RED, "h"),
      "Mondrian_LPDS4": (f"Mondrian, {NAME}", S.ORANGE, "s"),
      "LoBoost": ("LoBoost", S.AQUA, "D"),
      "LCP_RBF": ("LCP, RBF kernel", S.YELLOW, "^"),
      "NCP_LPDS": (f"NCP, {NAME} scale", S.MAGENTA, "v"),
      "NCP_ErrModel": ("NCP, error-model scale", S.VIOLET, "X"),
      "WLACP_LCP_la_norm_g1": ("WLACP", S.BLUE, "o")}
MC_KEY = {"split": "Split", "split_uncorrected": "Split_uncorrected", "wq": "WQ",
          "wlacp_lcp_la_g1": "WLACP_LCP_la_g1", "wlacp_lcp_la_norm_g1": "WLACP_LCP_la_norm_g1"}


def cp_legend(fig, keys, ncol, extra=None):
    h = [S.handle(CP[k][1], CP[k][2]) for k in keys]
    lab = [CP[k][0] for k in keys]
    if extra:
        h += [e[0] for e in extra]
        lab += [e[1] for e in extra]
    S.fig_legend(fig, h, lab, ncol=ncol, size=7.0, spacing=1.0)


def fig_conformal(R, RA, out):
    cp = load(R, "e02_conformal", "cp")
    fig, (ax, bx, cx) = new_fig(3.05, [1.08, 1.32, 0.98])
    cp_legend(fig, ["WLACP_LCP_la_norm_g1", "WLACP_LCP_la_g1", "WQ", "Split", "Split_uncorrected",
                    "Mondrian_LPDS4", "LoBoost", "LCP_RBF", "NCP_LPDS", "NCP_ErrModel"], ncol=5)
    qs = ["q1", "q2", "q3", "q4"]
    for k_ in ["Split", "WQ", "LoBoost", "Mondrian_LPDS4", "WLACP_LCP_la_norm_g1"]:
        lab, col, mk = CP[k_]
        yv = [ds_mean(cp, f"cov_{q}")[k_].mean() for q in qs]
        main = k_ == "WLACP_LCP_la_norm_g1"
        S.glow_line(ax, range(4), yv, col, lw=2.0 if main else 1.3, glow=main, zorder=5 if main else 3)
        S.dot(ax, range(4), yv, col, mk, s=30 if main else 18, z=6 if main else 4)
    S.hline(ax, 0.9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    ax.set_xlabel(f"{NAME} quartile of the test point")
    ax.set_ylabel("Coverage")
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylim(0.83, 0.97)
    S.panel(ax, "a", f"Coverage by {NAME} quartile")

    keys = ["WLACP_LCP_la_norm_g1", "NCP_ErrModel", "WLACP_LCP_la_g1", "WQ", "LoBoost",
            "NCP_LPDS", "Mondrian_LPDS4", "LCP_RBF"]
    pn = json.load(open(os.path.join(R, "tables", "paper_numbers.json")))
    for k, k_ in enumerate(keys):
        lab, col, mk = CP[k_]
        v9 = pn["core9"]["is_rel_mean"][k_]
        v19 = pn["all19"]["is_rel_mean"][k_]
        if k % 2 == 0:
            stripe(bx, k)
        bx.plot([v9, v19], [k, k], color=col, lw=2.0, alpha=0.35, zorder=2, solid_capstyle="butt")
        S.dot(bx, [v9], [k], col, "o", s=34, z=5)
        S.dot(bx, [v19], [k], col, "D", s=22, z=5)
        S.label(bx, 0.762, k, f"{v9:.3f}", S.INK, dx=0, ha="left", size=7.0,
                weight="bold" if k_ == "WLACP_LCP_la_norm_g1" else "medium")
    bx.axvline(1, color=S.INK2, lw=0.8)
    bx.set_yticks(range(len(keys)))
    bx.set_yticklabels([CP[k][0] for k in keys], fontsize=7.2)
    bx.set_ylim(len(keys) - 0.5, -0.5)
    bx.set_xlim(0.76, 1.01)
    bx.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    bx.set_xlabel("Interval score / split CP")
    bx.grid(axis="y", visible=False)
    bx.legend([S.handle(S.INK2, "o", ms=5.5), S.handle(S.INK2, "D", ms=4.5)], ["9 core", "all 19"],
              loc="lower left", bbox_to_anchor=(0.2, 0.0), fontsize=7.0, handletextpad=0.1,
              borderaxespad=0.2, ncol=2, columnspacing=0.8)
    S.panel(bx, "b", "Interval score, lower is better")

    mc = json.load(open(os.path.join(R, "e10_coverage_check", "summary.json")))
    ns = sorted(int(k) for k in mc)
    for mk_ in ["split_uncorrected", "wq", "split", "wlacp_lcp_la_g1", "wlacp_lcp_la_norm_g1"]:
        lab, col, mk = CP[MC_KEY[mk_]]
        mu = np.array([mc[str(n)][mk_]["mean"] for n in ns])
        se = np.array([mc[str(n)][mk_]["mc_se"] for n in ns])
        main = mk_ == "wlacp_lcp_la_norm_g1"
        S.soft_band(cx, ns, mu - 2 * se, mu + 2 * se, col, alpha=0.16)
        S.glow_line(cx, ns, mu, col, lw=2.0 if main else 1.3, glow=main, zorder=5 if main else 3)
        S.dot(cx, ns, mu, col, mk, s=26 if main else 16, z=6 if main else 4)
    S.hline(cx, 0.9)
    cx.set_xscale("log")
    S.log_ticks(cx.xaxis, ns, [str(n) for n in ns])
    cx.set_xlim(16, 250)
    cx.set_xlabel("Calibration-set size n")
    cx.set_ylabel("Marginal coverage")
    cx.set_ylim(0.85, 0.915)
    S.panel(cx, "c", "Coverage at small n")
    cap = (f"Conformal prediction without refitting at the 90\\% level. (a) Coverage in each {NAME} quartile of "
           f"the test set, with quartile thresholds fixed on the validation split (nine core datasets, five "
           f"seeds); the dashed line marks the nominal level. (b) Mean interval score divided by that of split "
           f"conformal prediction (NCP, normalized conformal prediction; Mondrian bins defined by {NAME} quartiles), "
           f"on the nine core "
           f"datasets (circles, value printed) and on all 19 regression tasks (diamonds). (c) Marginal coverage "
           f"in a Monte Carlo study with 2000 independent replications of the calibration and test sets for each "
           f"size; bands show two standard errors.")
    S.save(fig, os.path.join(out, "fig_conformal.png"), cap)


# --------------------------------------------------------------------------
def fig_calibration(R, out):
    cs = load(R, "e06_calib_size", "calib")
    x = cs.groupby(["dataset", "method", "n_cal", "metric"]).value.mean()
    fig, axes = new_fig(2.75, [1, 1, 1])
    keys = ["Split", "WQ", "Mondrian_LPDS4", "WLACP_LCP_la_g1", "WLACP_LCP_la_norm_g1"]
    cp_legend(fig, keys[::-1], ncol=5)
    for p, (metric, ylab, title) in enumerate([("cov_q1", "Coverage, lowest quartile", f"Lowest {NAME} quartile"),
                                               ("cov_all", "Coverage, full test set", "Full test set"),
                                               ("width_ratio_low_testq", "Width / split CP width",
                                                "Width, lowest quartile")]):
        ax = axes[p]
        for k_ in keys:
            lab, col, mk = CP[k_]
            t = x.xs((k_, metric), level=("method", "metric")).unstack("n_cal")
            ns = [n for n in t.columns if t[n].notna().sum() >= 5]
            mu = t[ns].mean()
            main = k_ == "WLACP_LCP_la_norm_g1"
            if not (metric.startswith("width") and k_ == "Split"):
                S.soft_band(ax, ns, t[ns].quantile(0.25), t[ns].quantile(0.75), col, alpha=0.08)
            S.glow_line(ax, ns, mu.to_numpy(), col, lw=2.0 if main else 1.3, glow=main, zorder=5 if main else 3)
            S.dot(ax, ns, mu.to_numpy(), col, mk, s=24 if main else 15, z=6 if main else 4)
        if metric.startswith("cov"):
            S.hline(ax, 0.9)
        ax.set_xscale("log")
        S.log_ticks(ax.xaxis, [100, 200, 500, 1000, 2000], ["100", "200", "500", "1k", "2k"])
        ax.set_xlabel("Calibration-set size")
        ax.set_ylabel(ylab)
        if metric == "width_ratio_low_testq":
            ax.set_yscale("log")
            S.log_ticks(ax.yaxis, [1, 1.5, 2, 3, 5], ["1", "1.5", "2", "3", "5"])
        S.panel(ax, "abc"[p], title)
    cap = (f"Effect of the calibration-set size, averaged over the nine core datasets (eight at 2000 points) and five seeds; bands show "
           f"the interquartile range over datasets and the dashed line marks the nominal level. (a) Coverage on "
           f"the lowest {NAME} quartile, (b) coverage on "
           f"the full test set and (c) mean interval width on the 25\\% of test points with the lowest {NAME}, "
           f"divided by that of split conformal prediction, on a logarithmic axis.")
    S.save(fig, os.path.join(out, "fig_calibration.png"), cap)


# --------------------------------------------------------------------------
def fig_ring(R, out):
    z = np.load(os.path.join(R, "e08_ring", "ring_s42.npz"))
    summ = load(R, "e08_ring", "ring").groupby("metric").value.agg(["mean", "std"])
    fig, (ax, bx, cx) = new_fig(2.35, [1, 1, 1.1])
    g = z["grid"]
    ext = [g[0], g[-1], g[0], g[-1]]
    th = np.linspace(0, 2 * np.pi, 400)
    cm_lp = LinearSegmentedColormap.from_list("lp", ["#ffffff"] + S.BLUE_RAMP)
    cm_iso = LinearSegmentedColormap.from_list("iso", ["#ffffff"] + S.ORANGE_RAMP)
    for p, (axx, field, cmap, title, lab) in enumerate(
            [(ax, "lpds_grid", cm_lp, f"{NAME} over the plane", NAME),
             (bx, "iso_grid", cm_iso, "Isolation Forest score", "anomaly score")]):
        F = z[field]
        im = axx.imshow(F, origin="lower", extent=ext, cmap=cmap, interpolation="bilinear",
                        vmin=np.percentile(F, 1), vmax=np.percentile(F, 99.5))
        axx.scatter(z["X_train"][::2, 0], z["X_train"][::2, 1], s=0.6, color=S.INK, alpha=0.28, linewidth=0)
        for r in (1.0, 1.5):
            axx.plot(r * np.cos(th), r * np.sin(th), color=S.INK, lw=0.7, ls=(0, (3, 2)))
        axx.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color=S.INK, lw=1.0, ls=(0, (1, 1.2)))
        axx.set_xticks([])
        axx.set_yticks([])
        axx.grid(False)
        for s_ in axx.spines.values():
            s_.set_visible(False)
        axx.set_aspect("equal")
        S.panel(axx, "ab"[p], title)
        cb = fig.colorbar(im, ax=axx, shrink=0.82, aspect=22, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.6)
        cb.set_label(lab, fontsize=7.2)
    pos = 0
    for k_, lab, s_, i_, col in [("LPDS", NAME, z["lpds_surface"], z["lpds_interior"], S.BLUE),
                                 ("ISO", "Isolation Forest", z["iso_surface"], z["iso_interior"], S.ORANGE)]:
        sc = np.median(s_)
        for k, (vals, nm) in enumerate([(s_ / sc, "ring"), (i_ / sc, "hole")]):
            xp = pos + k * 0.95
            parts = cx.violinplot([vals], positions=[xp], widths=0.8, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(col)
                b.set_alpha(0.22 if nm == "ring" else 0.55)
                b.set_edgecolor("none")
            cx.plot([xp] * 2, np.percentile(vals, [25, 75]), color=col, lw=3.0, solid_capstyle="round", zorder=4)
            S.dot(cx, [xp], [np.median(vals)], S.SURF, "o", s=14, z=6, ring=0)
            cx.annotate(nm, (xp, 0), xycoords=("data", "axes fraction"), xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top", fontsize=7.2, color=S.INK2)
        cx.annotate(lab, (pos + 0.475, 0), xycoords=("data", "axes fraction"), xytext=(0, -15),
                    textcoords="offset points", ha="center", va="top", fontsize=7.6, color=S.INK,
                    fontweight="semibold")
        pos += 2.3
    cx.set_xticks([])
    cx.set_xlim(-0.6, pos - 0.95)
    cx.axhline(1, color=S.INK2, lw=0.8, ls=(0, (3, 3)))
    cx.set_ylabel("Score / median ring score")
    cx.grid(axis="x", visible=False)
    S.panel(cx, "c", "Ring versus hollow interior")
    ar_l, ar_i = summ.loc["auroc_LPDS", "mean"], summ.loc["auroc_IsoForest", "mean"]
    cx.text(0.99, 0.99, f"AUROC, hole vs ring\n{NAME}  {ar_l:.2f}\nIsolation Forest  {ar_i:.2f}",
            transform=cx.transAxes, ha="right", va="top", fontsize=6.9, color=S.INK2, linespacing=1.35)
    cap = (f"Two-dimensional ring benchmark. Training points (black) are uniform on the annulus between radii "
           f"1.0 and 1.5 (dashed circles), and interior test points are uniform on the disc of radius 0.5 "
           f"(dotted circle). (a) {NAME} and (b) the Isolation Forest anomaly score over the plane for one seed. "
           f"(c) Distributions of both scores on held-out ring points and interior points, divided by the median "
           f"score on ring points; the AUROC for separating interior from ring points is the mean over five "
           f"seeds.")
    S.save(fig, os.path.join(out, "fig_ring.png"), cap)


# --------------------------------------------------------------------------
def fig_noise(R, out):
    syn = load(R, "e07_noise_synthetic", "noise")
    fig, (ax, bx, cx) = new_fig(2.75, [1.0, 0.9, 1.3])
    s = syn[syn.metric == "spearman_LPDS"].groupby("method").value.agg(["mean", "std"])
    types = [("sparse", S.VIOLET, "^"), ("homo", S.YELLOW, "o"), ("dense", S.MAGENTA, "v")]
    lv = [0.0, 0.1, 0.25, 0.5, 1.0]
    ends, colors = [], []
    for typ, col, mk in types:
        mu = np.array([s.loc["homo_0", "mean"] if l == 0 else s.loc[f"{typ}_{l:g}", "mean"] for l in lv])
        sd = np.array([s.loc["homo_0", "std"] if l == 0 else s.loc[f"{typ}_{l:g}", "std"] for l in lv])
        S.soft_band(ax, lv, mu - sd, mu + sd, col, alpha=0.13)
        S.glow_line(ax, lv, mu, col, lw=1.8, glow=False)
        S.dot(ax, lv, mu, col, mk, s=22)
        ends.append(mu[-1])
        colors.append(col)
    ax.plot([-0.05, 1.0], [0, 0], color=S.INK2, lw=0.8, zorder=2)
    ax.set_xlim(-0.05, 1.45)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "1"])
    ax.set_xlabel("Noise level (× signal SD)")
    ax.set_ylabel(f"Spearman ρ of {NAME}")
    S.panel(ax, "a", "Synthetic, noise placement")
    S.spread_labels(ax, 1.0, ends, ["sparse regions", "uniform", "dense regions"], colors,
                    x_text=0.73, gap_pt=8.5, size=7.0)

    lvl = syn[syn.level == 0.25].groupby(["noise", "metric"]).value.mean().unstack()
    trio = [("spearman_LPDS", NAME, S.BLUE, "o"), ("spearman_IsoForest", "Isolation Forest", S.ORANGE, "s"),
            ("spearman_kNN", "kNN distance", S.AQUA, "D")]
    noises = ["sparse", "homo", "dense"]
    for j, nz in enumerate(noises):
        if j % 2 == 0:
            stripe(bx, j, horizontal=False)
        for (m, lab, col, mk), o in zip(trio, [-0.22, 0, 0.22]):
            v = lvl.loc[nz, m]
            bx.plot([j + o] * 2, [0, v], color=col, lw=2.2, alpha=0.35, solid_capstyle="butt", zorder=2)
            S.dot(bx, [j + o], [v], col, mk, s=26 if m == "spearman_LPDS" else 20, z=5)
    bx.axhline(0, color=S.INK2, lw=0.8)
    bx.set_xticks(range(3))
    bx.set_xticklabels(["sparse", "uniform", "dense"])
    bx.set_xlim(-0.5, 2.5)
    bx.set_xlabel("Noise placement, level 0.25")
    bx.set_ylabel("Spearman ρ with |error|")
    bx.grid(axis="x", visible=False)
    bx.legend([S.handle(c, mk) for _, _, c, mk in trio], [lab for _, lab, _, _ in trio],
              loc="upper right", fontsize=6.8, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.3)
    S.panel(bx, "b", "Three scores, level 0.25")

    bk = load(R, "e07_noise_bike", "noise")
    b = bk.groupby(["method", "metric"]).value.agg(["mean", "std"])
    items = [("original", "Original"), ("perm_resid", "Permuted"), ("gauss_0", "Smooth"),
             ("gauss_0.25", "+0.25 SD"), ("gauss_0.5", "+0.5 SD"), ("gauss_1", "+1 SD"), ("gauss_2", "+2 SD")]
    cx.set_xlim(-0.6, len(items) - 0.4)
    cx.set_ylim(-0.21, 0.1)
    cx.set_xticks(range(len(items)))
    cx.set_xticklabels([lab for _, lab in items], fontsize=6.9)
    rotate(cx, 35)
    cx.set_yticks([-0.2, -0.15, -0.1, -0.05, 0, 0.05])
    cx.set_ylabel(f"Spearman ρ of {NAME}")
    cx.grid(axis="x", visible=False)
    S.panel(cx, "c", "Bike Sharing target variants")
    fig.canvas.draw()
    for k, (k_, lab) in enumerate(items):
        mu, sd = b.loc[(k_, "spearman_LPDS"), "mean"], b.loc[(k_, "spearman_LPDS"), "std"]
        col = S.ORANGE if k_ == "original" else (S.BLUE if k_ == "perm_resid" else S.BLUE_RAMP[4 + k])
        S.rbar(cx, k, mu, 0.6, col, frac=0.3)
        cx.vlines(k, mu - sd, mu + sd, color=S.INK2, lw=0.9, zorder=4)
        r2 = b.loc[(k_, "r2"), "mean"]
        cx.text(k, 0.083, f"{r2:.2f}", ha="center", va="center", fontsize=6.7, color=S.INK2)
    cx.text(-0.5, 0.083, "R²", ha="right", va="center", fontsize=6.8, color=S.INK, fontweight="semibold")
    cx.axhline(0, color=S.INK2, lw=0.8)
    cap = (f"Effect of the noise structure on the association between risk scores and absolute error (five "
           f"seeds). (a) Mean and standard deviation of the Spearman correlation of {NAME} on synthetic data "
           f"with skewed features, as noise of increasing level is added uniformly, in the densest regions or in "
           f"the sparsest regions of the feature space. (b) Mean correlation of {NAME} and two ambient-space "
           f"scores at noise level 0.25. (c) Bike Sharing with the original target, a cross-fitted smooth "
           f"signal plus randomly permuted residuals, the smooth signal alone and the smooth signal plus "
           f"Gaussian noise whose standard deviation is a multiple of that of the original residuals; the "
           f"test $R^2$ of the deployed model is printed above each bar.")
    S.save(fig, os.path.join(out, "fig_noise.png"), cap)


# --------------------------------------------------------------------------
def _timing_panel(ax, xs, data, ser, xlim, bounds, ylim, yt):
    ends, texts, cols, wts = [], [], [], []
    for k_, lab, col, mk in ser:
        yv = [data[str(n)][k_] for n in xs]
        main = k_ in ("LPDS_lookup", "WLACP_LCP")
        S.glow_line(ax, xs, yv, col, lw=2.0 if main else 1.3, glow=main, zorder=5 if main else 3)
        S.dot(ax, xs, yv, col, mk, s=20 if main else 13, z=6 if main else 4, ring=0.8)
        ends.append(yv[-1])
        texts.append(lab)
        cols.append(col)
        wts.append("bold" if main else "medium")
    ax.set_xscale("log")
    ax.set_yscale("log")
    S.log_ticks(ax.xaxis, xs, [f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n) for n in xs])
    S.log_ticks(ax.yaxis, yt, [f"{v:g}" for v in yt])
    ax.set_xlim(*xlim)
    ax.spines["bottom"].set_bounds(*bounds)
    ax.set_ylim(*ylim)
    return ends, texts, cols, wts


def fig_timing(R, out):
    t = json.load(open(os.path.join(R, "e09_timing", "timing.json")))
    fig, (ax, bx, cx) = new_fig(2.6, [1.25, 0.95, 1.05])
    Ns = sorted(int(k) for k in t["by_N"])
    ser = [("IBUG", "IBUG", S.GREEN, "P"), ("LOF", "LOF", S.COLORS["LOF"], "<"),
           ("kNN", "kNN distance", S.AQUA, "D"), ("IsoForest", "Isolation Forest", S.ORANGE, "s"),
           ("Mahalanobis", "Mahalanobis", S.MAGENTA, "v"),
           ("predict", "GBDT prediction", S.NEUTRAL, "o"),
           ("LPDS_with_leaf_extraction", f"{NAME} + leaf indices", S.LIGHT_BLUE, "o"),
           ("LPDS_lookup", f"{NAME}", S.BLUE, "o")]
    lab = _timing_panel(ax, Ns, t["by_N"], ser, (400, 16000 * 9), (500, 16000), (0.08, 1500),
                        [0.1, 1, 10, 100, 1000])
    ax.set_xlabel("Training-set size N")
    ax.set_ylabel("Time for 400 queries (ms)")
    S.panel(ax, "a", "Scoring cost versus N")
    S.spread_labels(ax, 16000, *lab[:3], gap_pt=8.0, size=6.9, weights=lab[3])

    Ts = sorted(int(k) for k in t["by_T"])
    ser = [("LPDS_with_leaf_extraction", f"{NAME} + leaf indices", S.LIGHT_BLUE, "o"),
           ("predict", "GBDT prediction", S.NEUTRAL, "o"), ("LPDS_lookup", NAME, S.BLUE, "o")]
    lab = _timing_panel(bx, Ts, t["by_T"], ser, (8, 400 * 9), (10, 400), (0.02, 15), [0.1, 1, 10])
    bx.set_xlabel("Number of trees T")
    bx.set_ylabel("Time for 400 queries (ms)")
    S.panel(bx, "b", "Cost versus T, N = 6000")
    S.spread_labels(bx, 400, *lab[:3], gap_pt=8.0, size=6.9, weights=lab[3])
    S.log_ticks(bx.xaxis, [10, 30, 100, 400])

    ncs = sorted(int(k) for k in t["by_ncal"])
    ser = [("WLACP_LCP", "WLACP", S.BLUE, "o"), ("WQ", "Weighted quantile", S.LIGHT_BLUE, "s"),
           ("LoBoost_fit", "LoBoost fit", S.AQUA, "D"), ("Mondrian", "Mondrian", S.ORANGE, "s")]
    lab = _timing_panel(cx, ncs, t["by_ncal"], ser, (170, 2000 * 6), (200, 2000), (0.012, 3000),
                        [0.1, 1, 10, 100, 1000])
    cx.set_xlabel("Calibration-set size n")
    cx.set_ylabel("Time for 400 queries (ms)")
    S.panel(cx, "c", "Interval construction")
    S.spread_labels(cx, 2000, *lab[:3], gap_pt=8.0, size=6.9, weights=lab[3])
    cap = (f"Inference time on Superconduct with $T = 100$ trees unless varied, taken as the "
           f"median of repeated measurements. (a) Time to score 400 queries as the training-set size grows; {NAME} is "
           f"timed both from precomputed leaf indices and including the extraction of leaf indices. (b) {NAME} "
           f"and prediction time as the number of trees grows. (c) Time to build intervals for 400 queries as "
           f"the calibration-set size grows. All axes are logarithmic.")
    S.save(fig, os.path.join(out, "fig_timing.png"), cap)


FIGS = ["ranking", "quartile", "partition", "shift", "heatmaps", "conformal", "calibration",
        "ring", "noise", "timing"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--results_all", default="results_all")
    p.add_argument("--out", default="figures")
    p.add_argument("--only", default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S.setup()
    R, RA, out = a.results, a.results_all, a.out
    calls = {"ranking": lambda: fig_ranking(R, RA, out), "quartile": lambda: fig_quartile(R, RA, out),
             "partition": lambda: fig_partition(R, RA, out), "shift": lambda: fig_shift(R, out),
             "heatmaps": lambda: fig_heatmaps(R, out), "conformal": lambda: fig_conformal(R, RA, out),
             "calibration": lambda: fig_calibration(R, out), "ring": lambda: fig_ring(R, out),
             "noise": lambda: fig_noise(R, out), "timing": lambda: fig_timing(R, out)}
    for k in FIGS:
        if a.only and k not in a.only.split(","):
            continue
        calls[k]()
        print("figure", k)


if __name__ == "__main__":
    main()
