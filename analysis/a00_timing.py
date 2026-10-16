"""Per-job wall time of every experiment, read from the stored results, and
a projection of the total run time for a given number of parallel jobs.

  python -m analysis.a00_timing --results results --n_jobs 16
"""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--n_jobs", type=int, default=16)
    p.add_argument("--n_datasets", type=int, default=9)
    p.add_argument("--n_seeds", type=int, default=5)
    a = p.parse_args()
    rows = []
    for f in glob.glob(os.path.join(a.results, "*", "*.json")):
        d = json.load(open(f))
        if "elapsed_sec" in d:
            rows.append(dict(exp=os.path.basename(os.path.dirname(f)),
                             dataset=d["dataset"], seed=d["seed"], sec=d["elapsed_sec"]))
    df = pd.DataFrame(rows)
    per = df.groupby(["exp", "dataset"]).sec.mean().unstack()
    print((per / 60).round(2).to_string(), "\n(minutes per job)\n")
    summ = df.groupby("exp").sec.agg(["mean", "max"])
    # e07 and e08 run one job per seed; the others one per (dataset, seed).
    # The mean is over the measured datasets only, so measure every dataset
    # at least once for a fair projection.
    jobs = pd.Series({e: a.n_seeds if e.startswith(("e07", "e08")) else
                      a.n_datasets * a.n_seeds for e in summ.index})
    summ["jobs"] = jobs
    summ["core_hours"] = summ["mean"] * jobs / 3600
    summ["wall_hours_at_n_jobs"] = summ["core_hours"] / jobs.clip(upper=a.n_jobs)
    print(summ.round(2).to_string())
    print(f"\ntotal: {summ.core_hours.sum():.1f} core-hours, about "
          f"{summ.wall_hours_at_n_jobs.sum():.1f} h wall time with {a.n_jobs} jobs")


if __name__ == "__main__":
    main()
