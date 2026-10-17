"""E02. Conformal intervals that keep the deployed model (scikit-learn main
model).

Methods: split conformal prediction, the plain weighted quantile with and
without the test-point mass, WLACP with four leaf-agreement kernels (raw,
per-tree normalized, squared and fourth power) and with a kernel chosen on
the validation split, Mondrian conformal prediction on LPM quartiles,
normalized conformal prediction with an LPM, error-model or IBUG scale,
localized conformal prediction with an RBF kernel, LoBoost, and the Gaussian
intervals of IBUG.
Metrics: marginal coverage, coverage and width per LPM quartile, coverage on
the quartile of test points farthest from the training set in kNN distance
(a subgroup defined without LPM), worst-slab coverage, the interval score,
effective sample sizes of the kernel weights and the fraction of unbounded
intervals.

LPM quartile thresholds and all tuned hyperparameters come from the
validation split, so every procedure is fixed before calibration.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import pairwise_distances

from core import conformal as C
from core import metrics as M
from core.config import (ALPHA, LOBOOST_M_GRID, NCP_GAMMA_GRID, RBF_BW_GRID,
                         WLACP_KERNELS)
from core.runner import Context, base_parser, rec, run
from core.scores import IBUG, GeometryScores, error_model


def job(ds, seed, args):
    a = ALPHA
    c = Context(ds, seed, args.data_dir, quick=args.quick)
    lf, v = c.leafs, c.view
    r_cal, y_te, y_va = c.resid["cal"], c.y_test, c.y_val
    p_te, p_va = c.pred["test"], c.pred["val"]
    out, arrays = [], {}

    # groups on the test set, all with thresholds fixed on the val split
    g_te = c.lpds_groups(c.lpds["test"])
    g_cal = c.lpds_groups(c.lpds["cal"])
    geo = GeometryScores(c.X_train, c.std(c.X_train), seed)
    knn_val = geo.all(c.X_val, c.std(c.X_val))["kNN"]
    knn_te = geo.all(c.X_test, c.std(c.X_test))["kNN"]
    lp_te = c.lpds["test"]
    groups = {f"q{k + 1}": g_te == k for k in range(4)}
    groups["low_testq"] = lp_te <= np.quantile(lp_te, 0.25)   # lowest 25% of test points
    groups["knn_q4"] = knn_te > np.quantile(knn_val, 0.75)
    Xw = c.std(c.X_test)
    y_range = c.y_train.max() - c.y_train.min()

    def evaluate(name, lo, hi, extra=None):
        m = M.interval_metrics(lo, hi, y_te, a, groups, wsc_X=Xw,
                               wsc_dirs=args.wsc_dirs, seed=seed,
                               cap=(p_te, y_range))
        if extra:
            m.update(extra)
        rec(out, ds, seed, "cp", name, m)
        arrays[f"lo__{name}"] = lo.astype(np.float32)
        arrays[f"hi__{name}"] = hi.astype(np.float32)

    def val_is(lo, hi):
        lo = np.maximum(lo, p_va - y_range)
        hi = np.minimum(hi, p_va + y_range)
        return float(np.mean(M.interval_score(lo, hi, y_va, a)))

    # --------------------------------------------------------- standard
    q = C.split_quantile(r_cal, a)
    evaluate("Split", *C.sym(p_te, q))
    evaluate("Split_uncorrected", *C.sym(p_te, C.split_quantile(r_cal, a, False)))

    # ------------------------------------------------------------ WLACP
    best_val, best_tag = np.inf, None
    lcp_q = {}
    for kind, g in WLACP_KERNELS:
        tag = f"{kind}_g{g:g}"
        Htc = lf.kernel(c.L["test"], c.L["cal"], kind, g)
        Hcc = lf.kernel(c.L["cal"], c.L["cal"], kind, g)
        htt = lf.self_kernel(c.L["test"], kind, g)
        e = M.ess(Htc)
        diag = {"ess_mean": e.mean(), "ess_median": np.median(e),
                "ess_q1_mean": e[groups["q1"]].mean() if groups["q1"].any() else np.nan,
                "zero_weight_frac": float(np.mean(Htc.sum(axis=1) <= 0)),
                "max_weight_mean": Htc.max(axis=1).mean()}
        if kind == "la" and g == 1.0:
            evaluate("WQ", *C.sym(p_te, C.weighted_quantile(r_cal, Htc, a)), diag)
        evaluate(f"WLACP_testmass_{tag}", *C.sym(p_te, C.wlacp_testmass(r_cal, Htc, htt, a)))
        qt = C.wlacp_lcp(r_cal, Hcc, Htc, htt, a)
        lcp_q[tag] = qt
        evaluate(f"WLACP_LCP_{tag}", *C.sym(p_te, qt), diag)
        # kernel selection on the validation split (interval score)
        Hvc = lf.kernel(c.L["val"], c.L["cal"], kind, g)
        qv = C.wlacp_lcp(r_cal, Hcc, Hvc, lf.self_kernel(c.L["val"], kind, g), a)
        s = val_is(*C.sym(p_va, qv))
        if s < best_val:
            best_val, best_tag = s, tag
    evaluate("WLACP_LCP_auto", *C.sym(p_te, lcp_q[best_tag]))
    rec(out, ds, seed, "diag", "WLACP_LCP_auto", {"kernel_" + best_tag: 1.0})

    # ------------------------------------------- LCP with an RBF kernel
    Zc, Zt, Zv = c.std(c.X_cal), c.std(c.X_test), c.std(c.X_val)
    Dcc = pairwise_distances(Zc)
    med = np.median(Dcc[np.triu_indices_from(Dcc, 1)])
    Dtc, Dvc = pairwise_distances(Zt, Zc), pairwise_distances(Zv, Zc)
    best_val, best_h = np.inf, None
    for m_ in RBF_BW_GRID:
        h = m_ * med
        qv = C.wlacp_lcp(r_cal, np.exp(-Dcc ** 2 / (2 * h * h)),
                         np.exp(-Dvc ** 2 / (2 * h * h)), np.ones(len(Zv)), a)
        s = val_is(*C.sym(p_va, qv))
        if s < best_val:
            best_val, best_h = s, h
    qt = C.wlacp_lcp(r_cal, np.exp(-Dcc ** 2 / (2 * best_h ** 2)),
                     np.exp(-Dtc ** 2 / (2 * best_h ** 2)), np.ones(len(Zt)), a)
    evaluate("LCP_RBF", *C.sym(p_te, qt), {"bandwidth_over_median": best_h / med})

    # --------------------------------------------------------- Mondrian
    evaluate("Mondrian_LPDS4", *C.sym(p_te, C.mondrian(r_cal, g_cal, g_te, a, 4)))

    # ---------------------------------------------- normalized CP (LPDS)
    med_lp = np.median(c.lpds["val"])
    best_val, best_g = np.inf, None
    for gm in NCP_GAMMA_GRID:
        sc = lambda lp: (lp / med_lp) ** (-gm)
        qv = C.normalized(r_cal, sc(c.lpds["cal"]), sc(c.lpds["val"]), a)
        s = val_is(*C.sym(p_va, qv))
        if s < best_val:
            best_val, best_g = s, gm
    sc = lambda lp: (lp / med_lp) ** (-best_g)
    evaluate("NCP_LPDS", *C.sym(p_te, C.normalized(r_cal, sc(c.lpds["cal"]),
                                                  sc(lp_te), a)),
             {"gamma": best_g})

    # ----------------------------------- normalized CP (error model, IBUG)
    _, oof, (fold_te,) = error_model(c.X_cal, r_cal, [c.X_test], seed)
    floor = 0.1 * np.median(oof[oof > 0]) if np.any(oof > 0) else 1e-8
    evaluate("NCP_ErrModel", *C.sym(p_te, C.normalized(
        r_cal, np.maximum(oof, floor), np.maximum(fold_te, floor), a)))
    ibug = IBUG(lf, c.y_train).fit(c.L["val"], y_va, p_va)
    s_cal, s_te = ibug.scale(c.L["cal"]), ibug.scale(c.L["test"])
    evaluate("IBUG_Gaussian", *C.gaussian(p_te, s_te, a), {"k": ibug.k})
    evaluate("NCP_IBUG", *C.sym(p_te, C.normalized(r_cal, s_cal, s_te, a)))

    # ---------------------------------------------------------- LoBoost
    vals = v.leaf_values_flat()
    tree_var = vals[lf.flat(c.L["cal"])].var(axis=0)
    best_val, best_m = np.inf, None
    for m_ in LOBOOST_M_GRID:
        if m_ * 2 > len(r_cal):
            continue
        lb = C.LoBoost(m_min=m_, weights=tree_var).fit(c.L["cal"], r_cal, a)
        s = val_is(*C.sym(p_va, lb.halfwidth(c.L["val"])))
        if s < best_val:
            best_val, best_m = s, m_
    lb = C.LoBoost(m_min=best_m, weights=tree_var).fit(c.L["cal"], r_cal, a)
    evaluate("LoBoost", *C.sym(p_te, lb.halfwidth(c.L["test"])),
             {"m_min": best_m, "n_regions": len(lb.sizes)})
    return out, arrays


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--wsc_dirs", type=int, default=500)
    run(job, "e02_conformal", p.parse_args())
