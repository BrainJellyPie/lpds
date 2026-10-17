"""E04. Sensitivity of LPM to the fitted partition, and tuned models.

Configurations, all fitted on the same split as the main experiments:
  sklearn one-at-a-time variations around Table 2 (number of trees, depth,
  minimum leaf size, learning rate); LightGBM number of leaves; and models
  tuned by random search on the validation split (LightGBM and XGBoost with
  early stopping, scikit-learn over a coarser grid).
For each configuration: test R^2 / RMSE, LPDS Spearman and PRR, the
ambient-space baselines against the same model's errors. With --with_conformal,
also split CP, WLACP with the raw leaf-agreement kernel and Mondrian coverage;
these supplementary records are not used in the paper.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import numpy as np

from core import conformal as C
from core import metrics as M
from core.config import ALPHA, GBDT_MAIN
from core.models import fit_model
from core.runner import Context, base_parser, rec, run
from core.scores import GeometryScores

OAT = [("n_estimators", [50, 300, 500]), ("max_depth", [2, 6, 8]),
       ("min_samples_leaf", [5, 50, 100]), ("learning_rate", [0.02, 0.1, 0.2])]
LGBM_LEAVES = [8, 16, 31, 63, 127]


def configs(args, rng):
    out = [("sklearn", "base", {})]
    for k, vals in OAT:
        for v in vals:
            out.append(("sklearn", f"{k}={v}", {k: v}))
    for nl in LGBM_LEAVES:
        out.append(("lgbm", f"num_leaves={nl}", {"num_leaves": nl}))
    return out


def sample_params(lib, rng):
    if lib == "lgbm":
        return {"num_leaves": int(rng.choice([8, 16, 31, 63, 127, 255])),
                "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
                "min_samples_leaf": int(rng.choice([5, 10, 20, 50, 100])),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample": float(rng.uniform(0.6, 1.0)),
                "n_estimators": 3000, "max_depth": -1}
    if lib == "xgb":
        return {"max_depth": int(rng.integers(2, 11)),
                "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
                "min_samples_leaf": int(rng.choice([1, 5, 10, 20, 50])),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample": float(rng.uniform(0.6, 1.0)),
                "n_estimators": 3000}
    return {"max_depth": int(rng.integers(3, 9)),
            "learning_rate": float(rng.choice([0.02, 0.05, 0.1, 0.2])),
            "min_samples_leaf": int(rng.choice([5, 10, 20, 50, 100])),
            "subsample": float(rng.choice([0.6, 0.8, 1.0])),
            "n_estimators": int(rng.choice([100, 300, 600]))}


def tune(c, lib, n_trials, seed):
    rng = np.random.default_rng(seed + 17)
    best = (np.inf, None)
    for _ in range(n_trials):
        prm = sample_params(lib, rng)
        es = 100 if lib in ("lgbm", "xgb") else None
        v = fit_model(lib, c.X_train, c.y_train, seed, prm, c.X_val, c.y_val, es)
        rmse = np.sqrt(np.mean((c.y_val - v.predict(c.X_val)) ** 2))
        if rmse < best[0]:
            best = (rmse, (prm, v))
    return best[1]


def job(ds, seed, args):
    a = ALPHA
    base = Context(ds, seed, args.data_dir, quick=args.quick)
    geo = GeometryScores(base.X_train, base.std(base.X_train), seed)
    gte = geo.all(base.X_test, base.std(base.X_test))
    y = base.y_test
    y_range = base.y_train.max() - base.y_train.min()
    out = []

    def evaluate(lib, tag, view, params, fit_sec):
        c = Context.__new__(Context)
        c.__dict__.update({k: getattr(base, k) for k in (
            "dataset", "seed", "idx", "X_all", "y_all", "X_train", "y_train",
            "X_val", "y_val", "X_cal", "y_cal", "X_test", "y_test", "d", "scaler")})
        c.lib, c.view = lib, view
        c.refresh()
        err = c.resid["test"]
        lp = c.lpds["test"]
        m = {"r2": 1 - np.mean((y - c.pred["test"]) ** 2) / np.var(y),
             "rmse": np.sqrt(np.mean((y - c.pred["test"]) ** 2)),
             "n_trees": view.T, "fit_sec": fit_sec,
             "spearman_LPDS": M.spearman(-lp, err), "prr_LPDS": M.prr(-lp, err),
             "active_feature_fraction": view.active_feature_fraction(c.d)}
        for name, s in gte.items():
            m[f"spearman_{name}"] = M.spearman(s, err)
        for k, val in params.items():
            if isinstance(val, (int, float)):
                m[f"param_{k}"] = val
        if args.with_conformal:
            g = c.lpds_groups(lp)
            grp = {f"q{k + 1}": g == k for k in range(4)}
            r_cal, pt = c.resid["cal"], c.pred["test"]
            Htc = c.leafs.kernel(c.L["test"], c.L["cal"])
            Hcc = c.leafs.kernel(c.L["cal"], c.L["cal"])
            for name, q in (("Split", C.split_quantile(r_cal, a)),
                            ("WLACP_LCP", C.wlacp_lcp(r_cal, Hcc, Htc,
                                                      c.leafs.self_kernel(c.L["test"]), a)),
                            ("Mondrian", C.mondrian(r_cal, c.lpds_groups(c.lpds["cal"]),
                                                    g, a, 4))):
                mm = M.interval_metrics(*C.sym(pt, q), y, a, grp, cap=(pt, y_range))
                m.update({f"{name}_{k}": v for k, v in mm.items()
                          if k in ("cov_all", "cov_q1", "width_all", "is_all")})
        rec(out, ds, seed, "sens", f"{lib}:{tag}", m)

    rng = np.random.default_rng(seed)
    for lib, tag, prm in configs(args, rng):
        if lib == "sklearn" and tag == "base":
            evaluate(lib, tag, base.view, GBDT_MAIN, base.fit_time)
            continue
        t0 = time.time()
        v = fit_model(lib, base.X_train, base.y_train, seed, prm)
        evaluate(lib, tag, v, {**GBDT_MAIN, **prm}, time.time() - t0)
    for lib, n in (("lgbm", args.trials_lgbm), ("xgb", args.trials_xgb),
                   ("sklearn", args.trials_sklearn)):
        if n <= 0:
            continue
        t0 = time.time()
        prm, v = tune(base, lib, n, seed)
        evaluate(lib, "tuned", v, prm, time.time() - t0)
    return out, None


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--with_conformal", action="store_true")
    p.add_argument("--trials_lgbm", type=int, default=30)
    p.add_argument("--trials_xgb", type=int, default=30)
    p.add_argument("--trials_sklearn", type=int, default=10)
    run(job, "e04_sensitivity", p.parse_args())
