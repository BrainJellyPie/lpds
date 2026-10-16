"""E01. Error ranking under the main protocol (sklearn, Table 2 settings).

Covers: R2-2(a) rejection curves and PRR, R2-2(b) no-retraining baselines,
R2-2(c) labeled error model, R1-Q1 / R2-1 IBUG and virtual ensembles,
R2-7(a) model accuracy, R2-4 active-feature fraction and main-shift AUROC,
R2-7(f) contribution-weighted LPDS, R2-7(g) LPDS from stored counts,
R2-8(a) / minor 2 quartile errors. Per-point arrays are stored for the
bootstrap and mixed-model analysis of R2-3.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from core import metrics as M
from core.config import N_EXT, SHIFT_MAG
from core.runner import Context, base_parser, rec, run
from core.scores import (IBUG, GeometryScores, error_model, lpds_hybrids,
                         virtual_ensemble_var)
from core.shifts import directions, shift


def job(ds, seed, args):
    c = Context(ds, seed, args.data_dir, quick=args.quick)
    lf, v = c.leafs, c.view
    out = []
    err = c.resid["test"]
    # ---------------------------------------------------------- accuracy
    yt = c.y_test
    rec(out, ds, seed, "acc", "GBDT", {
        "r2": 1 - np.mean((yt - c.pred["test"]) ** 2) / np.var(yt),
        "rmse": np.sqrt(np.mean((yt - c.pred["test"]) ** 2)),
        "mae": np.mean(err),
        "r2_train": 1 - np.mean((c.y_train - c.pred["train"]) ** 2) / np.var(c.y_train),
        "fit_sec": c.fit_time,
        "active_feature_fraction": v.active_feature_fraction(c.d),
        "n_train": len(c.y_train), "n_cal": len(c.y_cal), "n_test": len(yt)})

    # -------------------------------------------------------- risk scores
    def all_scores(X, L, lp):
        Xs = c.std(X)
        s = {"LPDS": -lp}
        s["LPDS_stored"] = -lf.lpds(L, counts=v.stored_counts_flat())
        vals = v.leaf_values_flat()
        s["LPDS_varw"] = -lf.lpds(L, tree_weights=lf.tree_output_variance(vals))
        s["LPDS_absw"] = -lf.lpds(L, tree_weights=np.abs(vals[lf.flat(L)]))
        s.update(geo.all(X, Xs))
        s.update(lpds_hybrids(lp, c.lpds["val"], geo, X))
        s["LeafVar"] = lf.leaf_target_var(c.y_train, L)
        s["AbsPred"] = np.abs(v.predict(X))
        s["VirtualEns"] = virtual_ensemble_var(v, X)
        s["IBUG"] = ibug.scale(L)
        return s

    geo = GeometryScores(c.X_train, c.std(c.X_train), seed)
    ibug = IBUG(lf, c.y_train).fit(c.L["val"], c.y_val, c.pred["val"])
    risks = all_scores(c.X_test, c.L["test"], c.lpds["test"])
    # labeled regime: small GBDT fitted to the calibration residuals
    feats_cal = np.c_[c.X_cal, c.lpds["cal"], np.log(c.lpds["cal"])]
    feats_te = np.c_[c.X_test, c.lpds["test"], np.log(c.lpds["test"])]
    (em,), _, _ = error_model(c.X_cal, c.resid["cal"], [c.X_test], seed, cross_fit=False)
    (eml,), _, _ = error_model(feats_cal, c.resid["cal"], [feats_te], seed, cross_fit=False)
    risks["ErrModel"] = em
    risks["ErrModel+LPDS"] = eml
    rec(out, ds, seed, "diag", "IBUG", {"k": ibug.k, "gamma": ibug.gamma})

    arrays = {"err": err.astype(np.float32), "lpds": c.lpds["test"]}
    for name, s in risks.items():
        mm, mo, mr = M.retention_curve(s, err)
        rec(out, ds, seed, "rank", name, {
            "spearman": M.spearman(s, err),
            "prr_mae": M.prr(s, err, 1), "prr_mse": M.prr(s, err, 2),
            **{f"ret_mae_{f:.2f}": x for f, x in zip(M.RET_FRACS, mm)}})
        arrays[f"risk__{name}"] = s.astype(np.float64)
    for f, x, r in zip(M.RET_FRACS, mo, mr):
        rec(out, ds, seed, "rank_ref", "oracle", {f"ret_mae_{f:.2f}": x})
        rec(out, ds, seed, "rank_ref", "random", {f"ret_mae_{f:.2f}": r})

    # -------------------------------------------------- quartile errors
    lp = c.lpds["test"]
    q_test = np.searchsorted(np.quantile(lp, [0.25, 0.5, 0.75]), lp, side="right")
    q_val = c.lpds_groups(lp)
    for tag, q in (("testq", q_test), ("valq", q_val)):
        mae = [err[q == g].mean() if np.any(q == g) else np.nan for g in range(4)]
        rec(out, ds, seed, "quartile", tag, {
            **{f"mae_q{g + 1}": mae[g] for g in range(4)},
            "ratio_q1_q4": mae[0] / mae[3],
            "monotone_pairs": int(sum(mae[g] > mae[g + 1] for g in range(3)))})

    # -------------------------------------------- binary detection, PC1 3 sigma
    rng = np.random.default_rng(seed + 1)
    src = c.X_val[rng.choice(len(c.X_val), N_EXT, replace=len(c.X_val) < N_EXT)]
    dirs, ref = directions(c.std(c.X_train))
    Xs = shift(src, c.scaler, dirs["PC1"], SHIFT_MAG, ref)
    Ls = v.leaves(Xs)
    rs = all_scores(Xs, Ls, lf.lpds(Ls))
    for name in rs:
        rec(out, ds, seed, "ood_pc1_3sd", name, {"auroc": M.auroc(risks[name], rs[name])})
    return out, arrays


if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run(job, "e01_ranking", args)
