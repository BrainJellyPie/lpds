"""E05. Binary detection of synthetic extrapolation as a function of shift
magnitude and direction (R2-4, R2-7(c), R2-8(c), Figure 3).

Source points are validation-split points (never used to fit the model).
Directions PC1, PC2, PC3 and Diag use equal-displacement normalization (see
core/shifts.py). Each shift is applied either to all features or only to
continuous features (no binary or small-cardinality integer features).
The leaf-based scores are computed for every implementation in --libs; the
geometry baselines do not depend on the model.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from core import metrics as M
from core.config import N_EXT
from core.models import fit_model
from core.leaves import LeafStructure
from core.runner import Context, base_parser, rec, run
from core.scores import GeometryScores, lpds_hybrids
from core.shifts import continuous_mask, directions, shift

MAGS = [0.5, 1, 2, 3, 5, 10, 30, 100, 300]


def job(ds, seed, args):
    c = Context(ds, seed, args.data_dir, quick=args.quick)
    out = []
    geo = GeometryScores(c.X_train, c.std(c.X_train), seed)
    libs = {}
    for lib in args.libs.split(","):
        v = c.view if lib == "sklearn" else fit_model(lib, c.X_train, c.y_train, seed)
        lf = LeafStructure(v.n_ids, v.leaves(c.X_train))
        lp_val = lf.lpds(v.leaves(c.X_val))
        libs[lib] = (v, lf, lp_val)
    rng = np.random.default_rng(seed + 1)
    src = c.X_val[rng.choice(len(c.X_val), N_EXT, replace=len(c.X_val) < N_EXT)]
    dirs, ref = directions(c.std(c.X_train))
    cmask = continuous_mask(c.X_train)
    rec(out, ds, seed, "diag", "features",
        {"continuous_fraction": cmask.mean(), "d": c.d})

    def scores(X):
        s = geo.all(X, c.std(X))
        for lib, (v, lf, lp_val) in libs.items():
            lp = lf.lpds(v.leaves(X))
            s[f"LPDS[{lib}]"] = -lp
            if lib == "sklearn":
                s.update(lpds_hybrids(lp, lp_val, geo, X))
                s["_mean_lpds"] = lp
        return s

    base = scores(c.X_test)
    for mask_name, mask in (("all", None), ("continuous", cmask)):
        for dname, dvec in dirs.items():
            for mag in MAGS:
                Xs = shift(src, c.scaler, dvec, mag, ref, mask)
                if Xs is None:
                    continue
                s = scores(Xs)
                m = {name: M.auroc(base[name], s[name]) for name in s
                     if not name.startswith("_")}
                m["mean_lpds_ratio"] = s["_mean_lpds"].mean() / base["_mean_lpds"].mean()
                m["frac_out_of_range"] = geo.out_of_range(Xs).any(axis=1).mean()
                rec(out, ds, seed, "shift", f"{mask_name}:{dname}", m, magnitude=mag)
    return out, None


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--libs", default="sklearn,xgb,lgbm")
    run(job, "e05_shift", p.parse_args())
