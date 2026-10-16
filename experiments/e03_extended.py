"""E03. Refitting references and cross-implementation study, on the same
datasets and the same five seeds as the main tables (R2 minor 6).

Refitting references (they fit additional models, unlike LPDS and WLACP):
  NGBoost with 100 trees (submitted setting) and with early stopping on the
  validation split (R2 minor 7), CV+ (R2-6(e), minor 8), CQR (R2-6(b)).
Cross-implementation (R2-8(c)(d), minor 5): scikit-learn, XGBoost,
  LightGBM (leaf-wise, no depth limit) and LightGBM with max_depth = 4.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from core import conformal as C
from core import metrics as M
from core.config import ALPHA, GBDT_MAIN
from core.models import fit_model
from core.runner import Context, base_parser, rec, run
from core.scores import GeometryScores


def ngboost(c, seed, n_estimators, early):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    m = NGBRegressor(Dist=Normal, n_estimators=n_estimators,
                     learning_rate=GBDT_MAIN["learning_rate"],
                     minibatch_frac=GBDT_MAIN["subsample"],
                     Base=DecisionTreeRegressor(max_depth=GBDT_MAIN["max_depth"]),
                     random_state=seed, verbose=False)
    t0 = time.time()
    if early:
        m.fit(c.X_train, c.y_train, X_val=c.X_val, Y_val=c.y_val,
              early_stopping_rounds=50)
    else:
        m.fit(c.X_train, c.y_train)
    fit_sec = time.time() - t0
    d = m.pred_dist(c.X_test)
    mu, sd = d.params["loc"], d.params["scale"]
    return mu, sd, fit_sec, getattr(m, "best_val_loss_itr", None)


def job(ds, seed, args):
    a = ALPHA
    c = Context(ds, seed, args.data_dir, quick=args.quick)
    out = []
    y, p = c.y_test, c.pred["test"]
    gbdt_err = c.resid["test"]
    g_te = c.lpds_groups(c.lpds["test"])
    groups = {f"q{k + 1}": g_te == k for k in range(4)}
    y_range = c.y_train.max() - c.y_train.min()

    def cp_eval(name, lo, hi, center, extra=None):
        m = M.interval_metrics(lo, hi, y, a, groups, cap=(center, y_range))
        if extra:
            m.update(extra)
        rec(out, ds, seed, "cp_refit", name, m)

    # ------------------------------------------------------------ NGBoost
    for tag, n_est, early in (("NGBoost100", 100, False),
                              ("NGBoostES", args.ngb_max_iter, True)):
        mu, sd, fit_sec, best = ngboost(c, seed, n_est, early)
        own = np.abs(y - mu)
        rec(out, ds, seed, "ngboost", tag, {
            "rmse": np.sqrt(np.mean((y - mu) ** 2)),
            "r2": 1 - np.mean((y - mu) ** 2) / np.var(y),
            "nll": -norm.logpdf(y, mu, sd).mean(),
            "spearman_own": M.spearman(sd, own),
            "spearman_gbdt": M.spearman(sd, gbdt_err),
            "prr_gbdt": M.prr(sd, gbdt_err),
            "fit_sec": fit_sec, "best_iter": best if best is not None else n_est})
        cp_eval(tag, *C.gaussian(mu, sd, a), mu)

    # ---------------------------------------------------------------- CV+
    kf = KFold(5, shuffle=True, random_state=seed)
    fold_of = np.empty(len(c.y_train), dtype=int)
    resid = np.empty(len(c.y_train))
    preds = []
    t0 = time.time()
    for k, (tr, te) in enumerate(kf.split(c.X_train)):
        mk = fit_model("sklearn", c.X_train[tr], c.y_train[tr], seed)
        fold_of[te] = k
        resid[te] = np.abs(c.y_train[te] - mk.predict(c.X_train[te]))
        preds.append(mk.predict(c.X_test))
    lo, hi = C.cv_plus(np.array(preds), fold_of, resid, a)
    cp_eval("CVplus", lo, hi, p, {"fit_sec": time.time() - t0})

    # ---------------------------------------------------------------- CQR
    t0 = time.time()
    qs = []
    for qa in (a / 2, 1 - a / 2):
        mq = GradientBoostingRegressor(
            loss="quantile", alpha=qa, n_estimators=GBDT_MAIN["n_estimators"],
            max_depth=GBDT_MAIN["max_depth"],
            min_samples_leaf=GBDT_MAIN["min_samples_leaf"],
            learning_rate=GBDT_MAIN["learning_rate"],
            subsample=GBDT_MAIN["subsample"], random_state=seed)
        mq.fit(c.X_train, c.y_train)
        qs.append((mq.predict(c.X_cal), mq.predict(c.X_test)))
    lo, hi = C.cqr(qs[0][0], qs[1][0], c.y_cal, qs[0][1], qs[1][1], a)
    cp_eval("CQR", lo, hi, p, {"fit_sec": time.time() - t0})

    # ------------------------------------------- cross-implementation
    geo = GeometryScores(c.X_train, c.std(c.X_train), seed)
    gte = geo.all(c.X_test, c.std(c.X_test))
    for lib in args.libs.split(","):
        if lib == "sklearn":
            cc = c
        else:
            cc = Context.__new__(Context)
            cc.__dict__.update({k: getattr(c, k) for k in (
                "dataset", "seed", "idx", "X_all", "y_all", "X_train", "y_train",
                "X_val", "y_val", "X_cal", "y_cal", "X_test", "y_test", "d",
                "scaler")})
            cc.lib = lib
            t0 = time.time()
            cc.view = fit_model(lib, c.X_train, c.y_train, seed)
            cc.fit_time = time.time() - t0
            cc.refresh()
        lf, v = cc.leafs, cc.view
        err = cc.resid["test"]
        lp = cc.lpds["test"]
        m = {"r2": 1 - np.mean((y - cc.pred["test"]) ** 2) / np.var(y),
             "rmse": np.sqrt(np.mean((y - cc.pred["test"]) ** 2)),
             "spearman_LPDS": M.spearman(-lp, err),
             "spearman_LPDS_stored": M.spearman(
                 -lf.lpds(cc.L["test"], counts=v.stored_counts_flat()), err),
             "prr_LPDS": M.prr(-lp, err),
             "active_feature_fraction": v.active_feature_fraction(c.d),
             "mean_leaves_per_tree": float(np.mean([
                 np.count_nonzero(lf.counts[o:o + n])
                 for o, n in zip(lf.offsets, lf.n_ids)])),
             "fit_sec": cc.fit_time}
        for name, s in gte.items():
            m[f"spearman_{name}"] = M.spearman(s, err)
        rec(out, ds, seed, "impl", lib, m)
        # conformal summary under this implementation
        gq = cc.lpds_groups(lp)
        grp = {f"q{k + 1}": gq == k for k in range(4)}
        r_cal = cc.resid["cal"]
        Htc = lf.kernel(cc.L["test"], cc.L["cal"])
        Hcc = lf.kernel(cc.L["cal"], cc.L["cal"])
        pt = cc.pred["test"]
        for name, q in (
                ("Split", C.split_quantile(r_cal, a)),
                ("WLACP_LCP_la_g1", C.wlacp_lcp(r_cal, Hcc, Htc,
                                                lf.self_kernel(cc.L["test"]), a)),
                ("Mondrian_LPDS4", C.mondrian(r_cal, cc.lpds_groups(cc.lpds["cal"]),
                                              gq, a, 4))):
            lo, hi = C.sym(pt, q)
            mm = M.interval_metrics(lo, hi, y, a, grp, cap=(pt, y_range))
            rec(out, ds, seed, "impl_cp", f"{lib}:{name}", mm)
    return out, None


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--libs", default="sklearn,xgb,lgbm,lgbm_d4")
    p.add_argument("--ngb_max_iter", type=int, default=2000)
    run(job, "e03_extended", p.parse_args())
