"""E08. Two-dimensional ring benchmark (Fig. 9 of the paper).

Training and surface-test points: radius uniform on [R_IN, R_OUT], angle
uniform on [0, 2 pi). Interior points (hollow region): radius uniform on
[0, R_HOLE]. Target: y = sin(3 theta) + 0.5 r + N(0, NOISE^2).
Model: the main scikit-learn configuration of Table 2.
The constants below define the benchmark of the paper.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import IsolationForest

from core import metrics as M
from core.leaves import LeafStructure
from core.models import fit_model
from core.runner import base_parser, rec, resolve_seeds, run

N_TRAIN, N_TEST, N_INT = 2000, 400, 400
R_IN, R_OUT, R_HOLE, NOISE = 1.0, 1.5, 0.5, 0.1


def ring(n, r_lo, r_hi, rng):
    r = np.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2, n))     # uniform over area
    th = rng.uniform(0, 2 * np.pi, n)
    X = np.c_[r * np.cos(th), r * np.sin(th)]
    y = np.sin(3 * th) + 0.5 * r + NOISE * rng.standard_normal(n)
    return X, y


def job(ds, seed, args):
    rng = np.random.default_rng(seed)
    Xtr, ytr = ring(N_TRAIN, R_IN, R_OUT, rng)
    Xs, _ = ring(N_TEST, R_IN, R_OUT, rng)
    Xi, _ = ring(N_INT, 0.0, R_HOLE, rng)
    v = fit_model("sklearn", Xtr, ytr, seed)
    lf = LeafStructure(v.n_ids, v.leaves(Xtr))
    lp_s, lp_i = lf.lpds(v.leaves(Xs)), lf.lpds(v.leaves(Xi))
    iso = IsolationForest(n_estimators=100, random_state=seed).fit(Xtr)
    a_s, a_i = -iso.score_samples(Xs), -iso.score_samples(Xi)
    out = []
    rec(out, "ring", seed, "ring", "summary", {
        "lpds_interior_over_surface": lp_i.mean() / lp_s.mean(),
        "iso_interior_over_surface": a_i.mean() / a_s.mean(),
        "auroc_LPDS": M.auroc(-lp_s, -lp_i), "auroc_IsoForest": M.auroc(a_s, a_i)})
    arrays = dict(X_surface=Xs, X_interior=Xi, lpds_surface=lp_s,
                  lpds_interior=lp_i, iso_surface=a_s, iso_interior=a_i,
                  X_train=Xtr)
    # score fields on a regular grid, for the figure
    g = np.linspace(-1.75, 1.75, 241)
    G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    arrays.update(grid=g, lpds_grid=lf.lpds(v.leaves(G)).reshape(len(g), len(g)),
                  iso_grid=(-iso.score_samples(G)).reshape(len(g), len(g)))
    return out, arrays


if __name__ == "__main__":
    a = base_parser(__doc__).parse_args()
    import core.runner as R
    R.load_dataset = lambda *x, **k: None
    run(job, "e08_ring", a, jobs=[("ring", s) for s in resolve_seeds(a.seeds)])
