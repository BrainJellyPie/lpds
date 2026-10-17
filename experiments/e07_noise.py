"""E07. Target-noise experiments (Section VI-G and Fig. 8 of the paper).

(1) Bike_Sharing variants under the main protocol, five seeds. A smooth
    signal f is estimated once by 5-fold cross-fitted LightGBM on all rows.
      original      the real target (equals the main-table setting)
      perm_resid    f + randomly permuted residuals (same residual
                    distribution, heteroscedastic structure removed)
      gauss_<lam>   f + lam * sd(residual) * N(0, 1)
(2) Synthetic data with controlled support and noise structure:
    n = 10000 points, X ~ Beta(1.2, 3.0)^8 (dense near 0, sparse near 1), and
    the Friedman-type signal f = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4
    + 5 x5. Gaussian noise has standard deviation level * sd(f) times 1
    (uniform), 0.25 + 1.5 * rank (dense) or 0.25 + 1.5 * (1 - rank) (sparse),
    where rank in [0, 1] is the density rank of the point (1 = densest).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from core import metrics as M
from core.config import SPLIT, SPLIT_QUICK
from core.data import load_dataset, make_split
from core.leaves import LeafStructure
from core.models import fit_model
from core.runner import base_parser, rec, run, resolve_seeds
from core.scores import GeometryScores

LAMS = [0.0, 0.25, 0.5, 1.0, 2.0]
SYN_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]


def evaluate(X, y, seed, quick, out, ds, variant, **extra):
    idx = make_split(len(y), seed, SPLIT_QUICK if quick else SPLIT)
    Xtr, ytr, Xte, yte = X[idx["train"]], y[idx["train"]], X[idx["test"]], y[idx["test"]]
    v = fit_model("sklearn", Xtr, ytr, seed)
    lf = LeafStructure(v.n_ids, v.leaves(Xtr))
    lp = lf.lpds(v.leaves(Xte))
    pred = v.predict(Xte)
    err = np.abs(yte - pred)
    sc = StandardScaler().fit(Xtr)
    geo = GeometryScores(Xtr, sc.transform(Xtr), seed)
    # lpds_resid_corr duplicates spearman_LPDS and is kept to match the stored records
    m = {"r2": 1 - np.mean((yte - pred) ** 2) / np.var(yte),
         "spearman_LPDS": M.spearman(-lp, err), "prr_LPDS": M.prr(-lp, err),
         "lpds_resid_corr": M.spearman(-lp, err)}
    for name, s in geo.all(Xte, sc.transform(Xte)).items():
        m[f"spearman_{name}"] = M.spearman(s, err)
    rec(out, ds, seed, "noise", variant, m, **extra)


def smooth_signal(X, y, seed=0):
    f = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        v = fit_model("lgbm", X[tr], y[tr], seed,
                      {"n_estimators": 500, "num_leaves": 63, "learning_rate": 0.05,
                       "min_samples_leaf": 20, "subsample": 0.8})
        f[te] = v.predict(X[te])
    return f


def synthetic(n, d, noise, level, rng):
    X = rng.beta(1.2, 3.0, size=(n, d))
    f = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4])
    logdens = beta_dist.logpdf(X, 1.2, 3.0).sum(axis=1)
    rank = np.argsort(np.argsort(logdens)) / (n - 1)          # 1 = densest
    base = level * f.std()
    if noise == "homo":
        sd = np.full(n, base)
    elif noise == "dense":
        sd = base * (0.25 + 1.5 * rank)
    else:
        sd = base * (0.25 + 1.5 * (1 - rank))
    return X, f + sd * rng.standard_normal(n)


def job_bike(ds, seed, args):
    out = []
    X, y, _ = load_dataset("Bike_Sharing_Demand", args.data_dir)
    cache = os.path.join(args.data_dir, "Bike_Sharing_Demand_smooth.npy")
    if os.path.exists(cache):
        f = np.load(cache)
    else:
        f = smooth_signal(X, y)
        np.save(cache, f)
    res = y - f
    rng = np.random.default_rng(seed + 5)
    evaluate(X, y, seed, args.quick, out, "Bike_Sharing_Demand", "original")
    evaluate(X, f + rng.permutation(res), seed, args.quick, out,
             "Bike_Sharing_Demand", "perm_resid")
    for lam in LAMS:
        yv = f + lam * res.std() * rng.standard_normal(len(y))
        evaluate(X, yv, seed, args.quick, out, "Bike_Sharing_Demand",
                 f"gauss_{lam:g}", lam=lam)
    return out, None


def job_syn(ds, seed, args):
    out = []
    for noise in ("homo", "dense", "sparse"):
        for lev in SYN_LEVELS:
            if lev == 0.0 and noise != "homo":
                continue
            rng = np.random.default_rng(seed)
            X, y = synthetic(10000, 8, noise, lev, rng)
            evaluate(X, y, seed, args.quick, out, "synthetic", f"{noise}_{lev:g}",
                     noise=noise, level=lev)
    return out, None


if __name__ == "__main__":
    p = base_parser(__doc__)
    a = p.parse_args()
    seeds = resolve_seeds(a.seeds)
    run(job_bike, "e07_noise_bike", a, jobs=[("Bike_Sharing_Demand", s) for s in seeds])
    run(job_syn, "e07_noise_synthetic", a, jobs=[("synthetic", s) for s in seeds],
        prefetch=False)   # synthetic data, nothing to download
