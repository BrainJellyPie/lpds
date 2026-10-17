"""Job orchestration: argument parsing, context building, resumable
parallel execution, and storage of flat result records."""
import argparse
import json
import os
import time
import traceback

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import ALL19, EXTRA10, MAIN9, SEEDS, SPLIT, SPLIT_QUICK
from .data import load_dataset, make_split
from .leaves import LeafStructure
from .models import fit_model


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--datasets", default="main",
                   help="main | all | extra | comma-separated names")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true",
                   help="small splits for a smoke test")
    p.add_argument("--overwrite", action="store_true")
    return p


def resolve_datasets(arg):
    table = {"main": MAIN9, "all": ALL19, "extra": EXTRA10}
    return table.get(arg, [s for s in arg.split(",") if s])


def resolve_seeds(arg):
    return [int(s) for s in str(arg).split(",") if s]


class Context:
    """Data split, fitted model and leaf structure for one (dataset, seed)."""

    def __init__(self, dataset, seed, data_dir, quick=False, lib="sklearn",
                 params=None, split_cfg=None):
        X, y, _ = load_dataset(dataset, data_dir)
        self.dataset, self.seed, self.lib = dataset, seed, lib
        cfg = split_cfg or (SPLIT_QUICK if quick else SPLIT)
        self.idx = make_split(len(y), seed, cfg)
        self.X_all, self.y_all = X, y
        for k in ("train", "val", "cal", "test"):
            setattr(self, f"X_{k}", X[self.idx[k]])
            setattr(self, f"y_{k}", y[self.idx[k]])
        self.d = X.shape[1]
        self.scaler = StandardScaler().fit(self.X_train)
        t0 = time.time()
        self.view = fit_model(lib, self.X_train, self.y_train, seed, params)
        self.fit_time = time.time() - t0
        self.refresh()

    def refresh(self):
        v = self.view
        self.L = {k: v.leaves(getattr(self, f"X_{k}"))
                  for k in ("train", "val", "cal", "test")}
        self.pred = {k: v.predict(getattr(self, f"X_{k}"))
                     for k in ("train", "val", "cal", "test")}
        self.leafs = LeafStructure(v.n_ids, self.L["train"])
        self.lpds = {k: self.leafs.lpds(self.L[k])
                     for k in ("val", "cal", "test")}
        self.resid = {k: np.abs(getattr(self, f"y_{k}") - self.pred[k])
                      for k in ("val", "cal", "test")}

    def std(self, X):
        return self.scaler.transform(X)

    def lpds_groups(self, lp):
        """LPDS quartile index 0..3 using thresholds fixed on the validation
        split (group 0 = lowest support)."""
        th = np.quantile(self.lpds["val"], [0.25, 0.5, 0.75])
        return np.searchsorted(th, lp, side="right")


def rec(out, ds, seed, group, method, metrics, **extra):
    for k, v in metrics.items():
        r = dict(dataset=ds, seed=seed, group=group, method=method,
                 metric=k, value=None if v is None else float(v))
        r.update(extra)
        out.append(r)


def _run_one(fn, ds, seed, args, path):
    if os.path.exists(path + ".json") and not args.overwrite:
        return "skip"
    t0 = time.time()
    try:
        records, arrays = fn(ds, seed, args)
    except Exception:
        with open(path + ".err", "w") as f:
            f.write(traceback.format_exc())
        return "error"
    with open(path + ".json", "w") as f:
        json.dump({"dataset": ds, "seed": seed, "elapsed_sec": time.time() - t0,
                   "records": records}, f)
    if arrays:
        np.savez_compressed(path + ".npz", **arrays)
    return f"{time.time() - t0:.1f}s"


def write_env(out_root):
    """Record the Python and package versions once per output folder."""
    import importlib
    import platform
    path = os.path.join(out_root, "env.json")
    if os.path.exists(path):
        return
    info = {"python": platform.python_version(), "system": platform.system()}
    for m in ("numpy", "scipy", "pandas", "sklearn", "xgboost", "lightgbm",
              "ngboost", "statsmodels", "joblib"):
        try:
            info[m] = importlib.import_module(m).__version__
        except Exception:
            info[m] = None
    os.makedirs(out_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=1)


def run(fn, exp_name, args, jobs=None, prefetch=True):
    out_dir = os.path.join(args.out, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    write_env(args.out)
    if jobs is None:
        jobs = [(ds, s) for ds in resolve_datasets(args.datasets)
                for s in resolve_seeds(args.seeds)]
    if prefetch:
        for ds in sorted({j[0] for j in jobs}):
            load_dataset(ds, args.data_dir)   # download / cache once
    paths = [os.path.join(out_dir, f"{ds}_s{s}") for ds, s in jobs]
    if args.n_jobs == 1:
        status = [_run_one(fn, ds, s, args, p) for (ds, s), p in zip(jobs, paths)]
    else:
        from joblib import Parallel, delayed
        status = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_run_one)(fn, ds, s, args, p) for (ds, s), p in zip(jobs, paths))
    for (ds, s), st in zip(jobs, status):
        print(f"[{exp_name}] {ds} seed={s}: {st}")
    if "error" in status:
        raise SystemExit(f"[{exp_name}] some jobs failed; see the .err files in {out_dir}")
