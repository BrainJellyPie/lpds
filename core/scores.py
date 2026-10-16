"""Risk scores (higher value = higher expected error) and IBUG.

Label-free at prediction time:
  LPDS variants, Isolation Forest, AA-Range, Mahalanobis, kNN distance, LOF,
  within-leaf target variance, |prediction|, virtual-ensemble variance, IBUG.
Fitted on labeled calibration residuals:
  ErrModel (small GBDT on |residual|), ErrModel+LPDS.
"""
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .config import ISO_TREES, KNN_K, LOF_K, MAHAL_REG, IBUG_K_GRID


class GeometryScores:
    """Model-agnostic scores fitted on the standardized training features."""

    def __init__(self, Xtr_raw, Xtr_s, seed):
        self.lo = Xtr_raw.min(axis=0)
        self.hi = Xtr_raw.max(axis=0)
        self.rng_ = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.iso = IsolationForest(n_estimators=ISO_TREES, random_state=seed,
                                   n_jobs=1).fit(Xtr_s)
        self.mu = Xtr_s.mean(axis=0)
        cov = np.cov(Xtr_s, rowvar=False) + MAHAL_REG * np.eye(Xtr_s.shape[1])
        self.prec = np.linalg.pinv(cov)
        self.nn = NearestNeighbors(n_neighbors=KNN_K).fit(Xtr_s)
        self.lof = LocalOutlierFactor(n_neighbors=LOF_K, novelty=True).fit(Xtr_s)

    def out_of_range(self, X_raw):
        return (X_raw < self.lo) | (X_raw > self.hi)

    def exceedance(self, X_raw):
        """Distance beyond the training range, summed over features, in units
        of each feature's training range."""
        e = np.maximum(X_raw - self.hi, 0) + np.maximum(self.lo - X_raw, 0)
        return (e / self.rng_).sum(axis=1)

    def all(self, X_raw, X_s):
        D = X_s - self.mu
        dist, _ = self.nn.kneighbors(X_s)
        return {
            "IsoForest": -self.iso.score_samples(X_s),
            "AA-Range": self.out_of_range(X_raw).mean(axis=1),
            "Mahalanobis": np.sqrt(np.einsum("ij,jk,ik->i", D, self.prec, D)),
            "kNN": dist.mean(axis=1),
            "LOF": -self.lof.score_samples(X_s),
        }


def ecdf(ref):
    ref = np.sort(ref)
    return lambda v: np.searchsorted(ref, v, side="right") / len(ref)


def lpds_hybrids(lpds_q, lpds_val, geo, X_raw):
    """LPDS combined with a range check (Reviewer 2, comment 4).

    LPDS+Range  queries with any feature outside the training range rank
                above all in-range queries; within each group, by LPDS.
    LPDS+Exceed adds the summed range exceedance, so the score keeps growing
                as a query moves further outside the training data.
    """
    F = ecdf(lpds_val)
    base = 1.0 - F(lpds_q)
    oor = geo.out_of_range(X_raw).any(axis=1).astype(float)
    return {"LPDS+Range": base + oor, "LPDS+Exceed": base + geo.exceedance(X_raw)}


def virtual_ensemble_var(view, X, step=5):
    """Variance across truncated sub-ensembles of the second half of the
    boosting trajectory (virtual ensemble, Malinin et al., ICLR 2021)."""
    iters = list(range(view.T // 2, view.T + 1, step))
    if iters[-1] != view.T:
        iters.append(view.T)
    return view.staged(X, iters).var(axis=0)


def _err_model(seed):
    return GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                     min_samples_leaf=20, learning_rate=0.05,
                                     subsample=0.8, random_state=seed)


def error_model(X_cal, r_cal, X_q_list, seed, K=5, cross_fit=True):
    """Small GBDT fitted to |calibration residual|.

    Returns predictions for each array in X_q_list from a model fitted on the
    full calibration set, together with cross-fitted (out-of-fold)
    predictions on the calibration points and the fold-averaged predictions
    on each query array, which normalized conformal prediction needs.
    """
    full = _err_model(seed).fit(X_cal, r_cal)
    full_preds = [full.predict(Xq) for Xq in X_q_list]
    if not cross_fit:
        return full_preds, None, None
    oof = np.zeros(len(r_cal))
    fold_preds = [np.zeros(len(Xq)) for Xq in X_q_list]
    for tr, te in KFold(K, shuffle=True, random_state=seed).split(X_cal):
        m = _err_model(seed).fit(X_cal[tr], r_cal[tr])
        oof[te] = m.predict(X_cal[te])
        for k, Xq in enumerate(X_q_list):
            fold_preds[k] += m.predict(Xq) / K
    return full_preds, oof, fold_preds


class IBUG:
    """Instance-based uncertainty for GBDTs (Brophy and Lowd, NeurIPS 2022).

    Affinity = number of trees in which the query shares a leaf with a
    training instance. The predictive distribution is Normal with location
    equal to the GBDT prediction and scale equal to the standard deviation of
    the targets of the k highest-affinity training instances, followed by a
    multiplicative variance calibration. k and the calibration factor are
    selected by Gaussian NLL on the validation split, as in the original
    method. This re-implementation reads leaf assignments from any library;
    the official package (github.com/jjbrophy47/ibug) can be swapped in.
    """

    def __init__(self, leafs, y_train, k_grid=IBUG_K_GRID, chunk=512):
        self.leafs = leafs
        self.y = y_train
        self.k_grid = [k for k in k_grid if k <= len(y_train)]
        self.chunk = chunk
        self.min_scale = 1e-6 * max(np.std(y_train), 1e-12)

    def _topk_scales(self, Lq, ks):
        kmax = max(ks)
        out = np.empty((len(ks), Lq.shape[0]))
        for s in range(0, Lq.shape[0], self.chunk):
            A = self.leafs.affinity_counts(Lq[s:s + self.chunk], self.leafs.L_train)
            idx = np.argpartition(-A, kmax - 1, axis=1)[:, :kmax]
            a = np.take_along_axis(A, idx, axis=1)
            order = np.argsort(-a, axis=1, kind="stable")
            yk = self.y[np.take_along_axis(idx, order, axis=1)]
            for j, k in enumerate(ks):
                out[j, s:s + self.chunk] = yk[:, :k].std(axis=1)
        return np.maximum(out, self.min_scale)

    def fit(self, L_val, y_val, pred_val):
        sc = self._topk_scales(L_val, self.k_grid)
        resid = y_val - pred_val
        nll = [-norm.logpdf(resid, scale=s).mean() for s in sc]
        j = int(np.argmin(nll))
        self.k = self.k_grid[j]
        best, self.gamma = np.inf, 1.0
        for g in np.logspace(-1, 1, 41):
            v = -norm.logpdf(resid, scale=g * sc[j]).mean()
            if v < best:
                best, self.gamma = v, g
        return self

    def scale(self, Lq):
        return self.gamma * self._topk_scales(Lq, [self.k])[0]
