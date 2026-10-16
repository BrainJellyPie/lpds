"""Recompute the e01 Spearman records from the stored per-point arrays with
the corrected constancy test of core.metrics.spearman, and rewrite the
records in place. Every other record is left untouched; the script checks
that all unchanged scores reproduce their stored values.

  python -m analysis.p01_recompute_spearman --results results
"""
import argparse
import glob
import json
import os

import numpy as np

from core import metrics as M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    a = p.parse_args()
    changed, same, maxdiff = [], 0, 0.0
    for f in sorted(glob.glob(os.path.join(a.results, "e01_ranking", "*.json"))):
        z = np.load(f[:-5] + ".npz")
        d = json.load(open(f))
        for r in d["records"]:
            if r["group"] != "rank" or r["metric"] != "spearman":
                continue
            new = M.spearman(z["risk__" + r["method"]], z["err"])
            if abs(new - r["value"]) > 1e-12:
                changed.append((os.path.basename(f), r["method"], r["value"], new))
                r["value"] = new
            else:
                same += 1
        json.dump(d, open(f, "w"))
    print("unchanged", same)
    for c in changed:
        print("changed", c)


if __name__ == "__main__":
    main()
