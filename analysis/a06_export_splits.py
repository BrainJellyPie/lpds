"""Export the data splits used in the paper.

  python -m analysis.a06_export_splits --out splits

For every dataset of the benchmark and every random split, writes
<out>/<dataset>_s<seed>.npz with the row indices of the training,
validation, calibration and test sets. Indices refer to the rows of the
dataset CSV after rows with missing values are dropped (core/data.py).
Rows in none of the four sets are not used by the main protocol. The
experiments regenerate the same splits from the seed (core.data.make_split),
so these files serve as a reference and are not read by any script.
"""
import argparse
import os

import numpy as np

from core.config import ALL19, SEEDS
from core.data import load_dataset, make_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="splits")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for ds in ALL19:
        _, y, _ = load_dataset(ds, a.data_dir)
        for seed in SEEDS:
            idx = make_split(len(y), seed)
            np.savez_compressed(os.path.join(a.out, f"{ds}_s{seed}.npz"),
                                **{k: idx[k].astype(np.int64) for k in ("train", "val", "cal", "test")},
                                n_rows=np.int64(len(y)))
        print("splits", ds)


if __name__ == "__main__":
    main()
