"""Unit checks:  python -m tests.test_core"""
import numpy as np

from core import conformal as C
from core.leaves import LeafStructure
from core.models import fit_model


def data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    return X, X[:, 0] ** 2 + np.sin(X[:, 1]) + 0.3 * rng.normal(size=n)


def test_lpm_equals_mean_leaf_agreement():
    """LPDS equals the mean leaf agreement with the training points."""
    X, y = data(1500)
    for lib in ("sklearn", "xgb", "lgbm"):
        v = fit_model(lib, X[:1000], y[:1000], 0)
        lf = LeafStructure(v.n_ids, v.leaves(X[:1000]))
        Lq = v.leaves(X[1000:])
        assert np.allclose(lf.kernel(Lq, lf.L_train).mean(1), lf.lpds(Lq))


def test_stored_counts_without_subsampling():
    """With subsample = 1 the counts stored in the model equal routed counts."""
    X, y = data(1500)
    for lib in ("sklearn", "xgb", "lgbm"):
        v = fit_model(lib, X[:1000], y[:1000], 0, {"subsample": 1.0})
        lf = LeafStructure(v.n_ids, v.leaves(X[:1000]))
        Lq = v.leaves(X[1000:])
        assert np.allclose(lf.lpds(Lq, counts=v.stored_counts_flat()), lf.lpds(Lq))


def test_lcp_reduces_to_split_for_constant_kernel():
    """With a constant kernel, level-adjusted WLACP is split conformal."""
    rng = np.random.default_rng(1)
    for n in (9, 50, 333):
        r = np.abs(rng.normal(size=n))
        q = C.wlacp_lcp(r, np.ones((n, n)), np.ones((7, n)), np.ones(7), 0.1)
        assert np.all(q == C.split_quantile(r, 0.1))


def test_mondrian_and_cvplus_shapes():
    rng = np.random.default_rng(2)
    r = np.abs(rng.normal(size=200))
    g = rng.integers(0, 4, 200)
    assert C.mondrian(r, g, np.array([0, 1, 2, 3]), 0.1, 4).shape == (4,)
    lo, hi = C.cv_plus(rng.normal(size=(5, 30)), rng.integers(0, 5, 200), r, 0.1)
    assert lo.shape == (30,) and np.all(lo <= hi)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
