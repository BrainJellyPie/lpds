"""Monte Carlo check of finite-sample marginal coverage (Proposition for the
level-adjusted WLACP). A GBDT is fitted once on a training sample; then
fresh calibration and test samples are drawn repeatedly from the same
heteroscedastic distribution with sparse and dense regions. Small
calibration sizes make finite-sample effects visible.

Output: results/e10_coverage_check/summary.json with mean coverage and its
Monte Carlo standard error for every procedure and kernel.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json

import numpy as np

from core import conformal as C
from core.config import WLACP_KERNELS
from core.leaves import LeafStructure
from core.models import fit_model


def sample(n, rng, d=5):
    X = rng.beta(1.2, 3.0, size=(n, d))
    f = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3]
    sd = 0.5 + 3.0 * X[:, 0] ** 2            # larger noise in the sparse region
    return X, f + sd * rng.standard_normal(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--n_cal", type=int, nargs="+", default=[20, 50, 200])
    p.add_argument("--n_test", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    Xtr, ytr = sample(3000, rng)
    view = fit_model("sklearn", Xtr, ytr, seed=0)
    leafs = LeafStructure(view.n_ids, view.leaves(Xtr))
    summary = {}
    for n_cal in a.n_cal:
        cov = {}
        for rep in range(a.reps):
            Xc, yc = sample(n_cal, rng)
            Xt, yt = sample(a.n_test, rng)
            Lc, Lt = view.leaves(Xc), view.leaves(Xt)
            rc = np.abs(yc - view.predict(Xc))
            rt = np.abs(yt - view.predict(Xt))
            res = {"split": np.full(a.n_test, C.split_quantile(rc, a.alpha)),
                   "split_uncorrected": np.full(a.n_test, C.split_quantile(rc, a.alpha, corrected=False))}
            for kind, g in WLACP_KERNELS:
                tag = f"{kind}_g{g:g}"
                Htc = leafs.kernel(Lt, Lc, kind, g)
                Hcc = leafs.kernel(Lc, Lc, kind, g)
                htt = leafs.self_kernel(Lt, kind, g)
                if kind == "la" and g == 1.0:
                    res["wlacp_submitted"] = C.wlacp_submitted(rc, Htc, a.alpha)
                res[f"wlacp_testmass_{tag}"] = C.wlacp_testmass(rc, Htc, htt, a.alpha)
                res[f"wlacp_lcp_{tag}"] = C.wlacp_lcp(rc, Hcc, Htc, htt, a.alpha)
            for k, q in res.items():
                cov.setdefault(k, []).append(np.mean(rt <= q))
        summary[n_cal] = {k: {"mean": float(np.mean(v)),
                              "mc_se": float(np.std(v) / np.sqrt(len(v)))}
                          for k, v in cov.items()}
        print(f"n_cal={n_cal}")
        for k, v in summary[n_cal].items():
            print(f"  {k:32s} {v['mean']:.4f} +- {v['mc_se']:.4f}")
    os.makedirs(os.path.join(a.out, "e10_coverage_check"), exist_ok=True)
    with open(os.path.join(a.out, "e10_coverage_check", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
