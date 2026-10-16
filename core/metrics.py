"""Evaluation metrics."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

RET_FRACS = np.round(np.arange(0.0, 0.951, 0.05), 2)


def spearman(score, err):
    """Spearman correlation; 0 for a score that takes a single value (for
    example AA-Range when every test point lies inside the training range).
    Constancy is tested on the values themselves, so scores on a very small
    scale (such as variances of targets near 1e-4) are not mistaken for
    constant ones."""
    score = np.asarray(score, dtype=float)
    if np.ptp(score) == 0:
        return 0.0
    return float(spearmanr(score, err).statistic)


def _retention_auc(order_err, n):
    # error-retention curve: reject the first k of `order_err`, replace them
    # by zero error; area under the mean-error curve over k = 0..n-1
    tail = np.cumsum(order_err[::-1])[::-1]
    return tail.mean() / n


def prr(score, err, power=1):
    """Prediction rejection ratio (Malinin et al.) on the error-retention
    curve. 1 = oracle ordering, 0 = random ordering."""
    e = np.abs(err) ** power
    n = len(e)
    rnd = e.mean() * (n + 1) / (2 * n)
    model = _retention_auc(e[np.argsort(-score, kind="stable")], n)
    orc = _retention_auc(np.sort(e)[::-1], n)
    return float((rnd - model) / (rnd - orc)) if rnd > orc else 0.0


def retention_curve(score, err, fracs=RET_FRACS):
    """MAE of the retained predictions after rejecting the highest-risk
    fraction f, for each f in fracs (with oracle and random references)."""
    n = len(err)
    e = np.abs(err)
    order = np.argsort(-score, kind="stable")
    orc = np.sort(e)[::-1]
    out_m, out_o = [], []
    for f in fracs:
        k = int(round(f * n))
        out_m.append(e[order[k:]].mean())
        out_o.append(orc[k:].mean())
    return np.array(out_m), np.array(out_o), np.full(len(fracs), e.mean())


def auroc(score_in, score_out):
    y = np.r_[np.zeros(len(score_in)), np.ones(len(score_out))]
    s = np.r_[score_in, score_out]
    if np.allclose(s, s[0]):
        return 0.5
    return float(roc_auc_score(y, s))


def interval_score(lo, hi, y, alpha):
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def ess(W):
    s1 = W.sum(axis=1)
    s2 = (W ** 2).sum(axis=1)
    return np.where(s2 > 0, s1 ** 2 / np.maximum(s2, 1e-300), 0.0)


def _wsc_direction(z, cover, delta):
    n = len(z)
    order = np.argsort(z)
    c = cover[order].astype(float)
    cs = np.r_[0.0, np.cumsum(c)]
    m = max(int(np.ceil(delta * n)), 1)
    a = np.arange(n)[:, None]
    b = np.arange(n)[None, :]
    length = b - a + 1
    cov = (cs[b + 1] - cs[a]) / np.maximum(length, 1)
    cov = np.where(length >= m, cov, np.inf)
    k = np.argmin(cov)
    ia, ib = divmod(k, n)
    return cov.flat[k], z[order[ia]], z[order[ib]]


def wsc_unbiased(X, cover, delta=0.1, M=500, search_frac=0.25, seed=0):
    """Worst-slab coverage (Cauchois et al., JMLR 2021; Romano et al., 2020).

    Slabs {x : a <= v'x <= b} holding at least a fraction delta of the points
    are searched over M random directions on one part of the test set, and
    the coverage of the selected slab is evaluated on the other part.
    """
    rng = np.random.default_rng(seed)
    n = len(cover)
    perm = rng.permutation(n)
    ns = int(search_frac * n)
    s_idx, e_idx = perm[:ns], perm[ns:]
    V = rng.standard_normal((M, X.shape[1]))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    best = (np.inf, None, None, None)
    for v in V:
        c, a, b = _wsc_direction(X[s_idx] @ v, cover[s_idx], delta)
        if c < best[0]:
            best = (c, v, a, b)
    _, v, a, b = best
    z = X[e_idx] @ v
    sel = (z >= a) & (z <= b)
    return float(cover[e_idx][sel].mean()) if sel.any() else float("nan")


def interval_metrics(lo, hi, y, alpha, groups, wsc_X=None, wsc_dirs=None,
                     seed=0, cap=None):
    """Coverage, width and interval score overall and per group.

    groups: dict name -> boolean mask over the test points.
    cap: (center, R). An infinite interval carries no more information than
    the whole target range, so for width and interval score each endpoint is
    clipped to center -+ R (R = training target range). Coverage uses the
    unclipped interval and inf_frac records how often it was infinite.
    """
    inf_frac = float(np.mean(~np.isfinite(hi - lo)))
    cover = (y >= lo) & (y <= hi)
    if cap is not None:
        center, R = cap
        lo = np.maximum(lo, center - R)
        hi = np.minimum(hi, center + R)
    width = hi - lo
    isc = interval_score(lo, hi, y, alpha)
    out = {"cov_all": cover.mean(), "width_all": np.mean(width),
           "width_median": np.median(width), "is_all": isc.mean(),
           "inf_frac": inf_frac}
    for g, mask in groups.items():
        if mask.any():
            out[f"cov_{g}"] = cover[mask].mean()
            out[f"width_{g}"] = np.mean(width[mask])
            out[f"is_{g}"] = isc[mask].mean()
    if wsc_X is not None:
        out["wsc"] = wsc_unbiased(wsc_X, cover, M=wsc_dirs or 500, seed=seed)
    return {k: float(v) for k, v in out.items()}
