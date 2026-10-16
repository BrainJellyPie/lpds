"""E06. Calibration-set size sweep with five seeds (R2-6(d), Tables 10-11).

The train, validation and test splits are those of the main protocol; the
calibration set of size n_cal is drawn from the main calibration split plus
the rows left unused by the main split. Sizes the dataset cannot supply are
skipped. Full-test coverage is reported for every method.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from core import conformal as C
from core import metrics as M
from core.config import ALPHA
from core.runner import Context, base_parser, rec, run

NCALS = [100, 200, 500, 1000, 2000]


def job(ds, seed, args):
    a = ALPHA
    c = Context(ds, seed, args.data_dir, quick=args.quick)
    lf, v = c.leafs, c.view
    pool = np.r_[c.idx["cal"], c.idx["rest"]]
    rng = np.random.default_rng(seed + 3)
    pool = rng.permutation(pool)
    y, pt = c.y_test, c.pred["test"]
    g_te = c.lpds_groups(c.lpds["test"])
    lp = c.lpds["test"]
    groups = {f"q{k + 1}": g_te == k for k in range(4)}
    groups["low_testq"] = lp <= np.quantile(lp, 0.25)
    y_range = c.y_train.max() - c.y_train.min()
    out = []
    for n_cal in NCALS:
        if n_cal > len(pool):
            continue
        sel = pool[:n_cal]
        Xc, yc = c.X_all[sel], c.y_all[sel]
        Lc = v.leaves(Xc)
        rc = np.abs(yc - v.predict(Xc))
        Htc = lf.kernel(c.L["test"], Lc)
        Hcc = lf.kernel(Lc, Lc)
        htt = lf.self_kernel(c.L["test"])
        q_std = C.split_quantile(rc, a)
        res = {"Split": q_std,
               "WLACP_submitted": C.wlacp_submitted(rc, Htc, a),
               "WLACP_LCP_la_g1": C.wlacp_lcp(rc, Hcc, Htc, htt, a),
               "Mondrian_LPDS4": C.mondrian(rc, c.lpds_groups(lf.lpds(Lc)), g_te, a, 4)}
        Hn = lf.kernel(c.L["test"], Lc, "la_norm")
        res["WLACP_LCP_la_norm_g1"] = C.wlacp_lcp(
            rc, lf.kernel(Lc, Lc, "la_norm"), Hn, lf.self_kernel(c.L["test"], "la_norm"), a)
        e = M.ess(Htc)
        for name, q in res.items():
            m = M.interval_metrics(*C.sym(pt, q), y, a, groups, cap=(pt, y_range))
            qa = np.broadcast_to(q, pt.shape)
            low = groups["low_testq"]
            m["width_ratio_low_testq"] = float(np.mean(np.minimum(qa[low], y_range)) /
                                               min(q_std, y_range))
            if name == "WLACP_LCP_la_g1":
                m["max_la_mean"] = Htc.max(axis=1).mean()
                m["ess_mean"] = e.mean()
            rec(out, ds, seed, "calib", name, m, n_cal=n_cal)
    return out, None


if __name__ == "__main__":
    run(job, "e06_calib_size", base_parser(__doc__).parse_args())
