"""E09. Inference-time scaling (R1-Q1, R2-1, Tables 12-14).

Superconduct, T = 100, 400 queries. For training-set sizes N, the time to
score the queries with LPDS (count lookups, and lookups plus leaf
extraction), IBUG (affinity to every training point, top-k, spread), kNN
distance, LOF, Isolation Forest and Mahalanobis. For calibration sizes,
the time of submitted WLACP, level-adjusted WLACP (including the
calibration-calibration kernel), Mondrian and LoBoost assignment.
Run single-threaded; medians over repeats are stored.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import platform
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from core import conformal as C
from core.config import ALPHA
from core.data import load_dataset
from core.leaves import LeafStructure
from core.models import fit_model
from core.scores import IBUG, GeometryScores

N_LIST = [500, 1000, 2000, 4000, 8000, 16000]
NCAL_LIST = [200, 500, 1000, 2000]


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0   # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--n_query", type=int, default=400)
    a = p.parse_args()
    X, y, _ = load_dataset("superconduct", a.data_dir)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    q_idx = perm[:a.n_query]
    cal_pool = perm[a.n_query:a.n_query + max(NCAL_LIST)]
    tr_pool = perm[a.n_query + max(NCAL_LIST):]
    Xq = X[q_idx]
    res = {"machine": platform.processor() or platform.machine(),
           "python": platform.python_version(), "by_N": {}, "by_ncal": {}}
    for N in N_LIST:
        if N > len(tr_pool):
            continue
        tr = tr_pool[:N]
        v = fit_model("sklearn", X[tr], y[tr], 42)
        Ltr = v.leaves(X[tr])
        lf = LeafStructure(v.n_ids, Ltr)
        Lq = v.leaves(Xq)
        sc = StandardScaler().fit(X[tr])
        geo = GeometryScores(X[tr], sc.transform(X[tr]), 42)
        Xqs = sc.transform(Xq)
        ib = IBUG(lf, y[tr])
        ib.k, ib.gamma = 61, 1.0
        r = {"predict": timeit(lambda: v.predict(Xq), a.reps),
             "LPDS_lookup": timeit(lambda: lf.lpds(Lq), a.reps),
             "LPDS_with_leaf_extraction": timeit(lambda: lf.lpds(v.leaves(Xq)), a.reps),
             "IBUG": timeit(lambda: ib.scale(Lq), a.reps),
             "kNN": timeit(lambda: geo.nn.kneighbors(Xqs), a.reps),
             "LOF": timeit(lambda: geo.lof.score_samples(Xqs), a.reps),
             "IsoForest": timeit(lambda: geo.iso.score_samples(Xqs), a.reps),
             "Mahalanobis": timeit(lambda: np.einsum(
                 "ij,jk,ik->i", Xqs - geo.mu, geo.prec, Xqs - geo.mu), a.reps)}
        res["by_N"][N] = r
        print("N", N, {k: round(x, 2) for k, x in r.items()})
    # number of trees, at the main training-set size
    res["by_T"] = {}
    for T in [10, 25, 50, 100, 200, 400]:
        vT = fit_model("sklearn", X[tr_pool[:6000]], y[tr_pool[:6000]], 42,
                       {"n_estimators": T})
        lfT = LeafStructure(vT.n_ids, vT.leaves(X[tr_pool[:6000]]))
        LqT = vT.leaves(Xq)
        r = {"LPDS_lookup": timeit(lambda: lfT.lpds(LqT), a.reps),
             "LPDS_with_leaf_extraction": timeit(lambda: lfT.lpds(vT.leaves(Xq)), a.reps),
             "predict": timeit(lambda: vT.predict(Xq), a.reps)}
        res["by_T"][T] = r
        print("T", T, {k: round(x, 2) for k, x in r.items()})
    v = fit_model("sklearn", X[tr_pool[:6000]], y[tr_pool[:6000]], 42)
    lf = LeafStructure(v.n_ids, v.leaves(X[tr_pool[:6000]]))
    Lq = v.leaves(Xq)
    for n_cal in NCAL_LIST:
        cidx = cal_pool[:n_cal]
        Lc = v.leaves(X[cidx])
        rc = np.abs(y[cidx] - v.predict(X[cidx]))
        g_c = np.searchsorted(np.quantile(lf.lpds(Lc), [.25, .5, .75]), lf.lpds(Lc))
        g_q = np.searchsorted(np.quantile(lf.lpds(Lc), [.25, .5, .75]), lf.lpds(Lq))

        def sub():
            C.wlacp_submitted(rc, lf.kernel(Lq, Lc), ALPHA)

        def lcp():
            C.wlacp_lcp(rc, lf.kernel(Lc, Lc), lf.kernel(Lq, Lc), lf.self_kernel(Lq), ALPHA)

        lb = C.LoBoost(m_min=50).fit(Lc, rc, ALPHA)
        r = {"WLACP_submitted": timeit(sub, a.reps),
             "WLACP_LCP": timeit(lcp, max(1, a.reps // 2)),
             "Mondrian": timeit(lambda: C.mondrian(rc, g_c, g_q, ALPHA, 4), a.reps),
             "LoBoost_fit": timeit(lambda: C.LoBoost(m_min=50).fit(Lc, rc, ALPHA), a.reps),
             "LoBoost_assign": timeit(lambda: lb.halfwidth(Lq), a.reps),
             "LA_matrix_MB": lf.kernel(Lq, Lc).nbytes / 2 ** 20}
        res["by_ncal"][n_cal] = r
        print("n_cal", n_cal, {k: round(x, 2) for k, x in r.items()})
    os.makedirs(os.path.join(a.out, "e09_timing"), exist_ok=True)
    with open(os.path.join(a.out, "e09_timing", "timing.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
