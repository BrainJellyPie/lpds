"""Dataset loading and splitting."""
import os

import numpy as np
import pandas as pd

from .config import SPLIT

HF_REPO = "inria-soda/tabular-benchmark"


def _download(name, data_dir):
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(HF_REPO, f"reg_num/{name}.csv", repo_type="dataset")
    os.makedirs(data_dir, exist_ok=True)
    df = pd.read_csv(src)
    df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)


def load_dataset(name, data_dir="data"):
    """Return X (n, d), y (n,), feature names.

    The target is the last column, rows with missing values are dropped and
    all numeric features are kept, as in Table 1 of the manuscript. A cached
    .npz copy is written next to the CSV for fast reloading.
    """
    cache = os.path.join(data_dir, f"{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], list(z["cols"])
    csv = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(csv):
        _download(name, data_dir)
    df = pd.read_csv(csv).dropna()
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    cols = np.array(df.columns[:-1], dtype=object)
    try:
        np.savez(cache, X=X, y=y, cols=cols)
    except OSError:
        pass
    return X, y, list(cols)


def make_split(n, seed, cfg=SPLIT):
    """Disjoint train / val / cal / test index sets plus the unused rest.

    Fractions are applied to n and then capped, which reproduces the
    sizes in Table 1 (for example n_train = 4915 on CPU_act).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = min(int(cfg["train"] * n), cfg["cap_train"])
    n_va = min(int(cfg["val"] * n), cfg["cap_val"])
    n_ca = min(int(cfg["cal"] * n), cfg["cap_cal"])
    n_te = min(int(cfg["test"] * n), cfg["cap_test"])
    cut = np.cumsum([n_tr, n_va, n_ca, n_te])
    return dict(train=perm[:cut[0]], val=perm[cut[0]:cut[1]],
                cal=perm[cut[1]:cut[2]], test=perm[cut[2]:cut[3]],
                rest=perm[cut[3]:])
