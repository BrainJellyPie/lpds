"""Conformal procedures. All scores are absolute residuals unless stated.

Interval-returning functions give (lo, hi) arrays for the query points.
"""
import math

import numpy as np
from scipy.stats import norm


def conf_index(n, alpha):
    """1-based rank of the split-conformal quantile, ceil((n+1)(1-alpha))."""
    return int(math.ceil((n + 1) * (1 - alpha)))


def split_quantile(r, alpha, corrected=True):
    r = np.sort(r)
    n = len(r)
    k = conf_index(n, alpha) if corrected else int(math.ceil(n * (1 - alpha)))
    return np.inf if k > n else r[k - 1]


def sym(pred, q):
    q = np.broadcast_to(q, pred.shape)
    return pred - q, pred + q


# ---------------------------------------------------------------- WLACP
def _sorted(r):
    order = np.argsort(r, kind="mergesort")
    return order, r[order]


def wlacp_submitted(r_cal, H_tc, alpha):
    """WLACP as in Definition 5 of the submitted manuscript: weights
    normalized over the calibration points only, level 1 - alpha, and the
    all-zero fallback to uniform weights."""
    order, rs = _sorted(r_cal)
    W = H_tc[:, order]
    tot = W.sum(axis=1)
    cum = np.cumsum(W, axis=1)
    thr = (1 - alpha) * tot
    k = np.argmax(cum >= thr[:, None] * (1 - 1e-12), axis=1)
    q = rs[k]
    zero = tot <= 0
    if zero.any():
        q[zero] = split_quantile(r_cal, alpha, corrected=False)
    return q


def wlacp_testmass(r_cal, H_tc, h_tt, alpha):
    """Weighted quantile that also carries the test-point mass H(x*, x*) at
    +infinity (form of Tibshirani et al. 2019), without level adjustment."""
    order, rs = _sorted(r_cal)
    W = H_tc[:, order]
    tot = W.sum(axis=1) + h_tt
    cum = np.cumsum(W, axis=1)
    thr = (1 - alpha) * tot
    reach = cum >= thr[:, None] * (1 - 1e-12)
    k = np.argmax(reach, axis=1)
    q = rs[k].astype(float)
    q[~reach.any(axis=1)] = np.inf
    return q


def wlacp_lcp(r_cal, H_cc, H_tc, h_tt, alpha, chunk=16, block=64):
    """Leaf-agreement localized conformal prediction with the level
    adjustment of Guan (Biometrika, 2023).

    For each point i of the augmented sample (calibration points and the test
    point), S_i is the localized weight of the points whose score is strictly
    below V_i, using the weights H(x_i, .) normalized over all n+1 points:
        S_i = sum_j H(x_i, x_j) 1[V_j < V_i] / sum_j H(x_i, x_j).
    Guan's adjusted level is the smallest tau with
        #{i : V_i <= Q(tau; F_i)} = #{i : S_i < tau} >= (1 - alpha)(n + 1),
    so the test value v is accepted when S_{n+1}(v) is not above the K-th
    smallest S, K = ceil((1 - alpha)(n + 1)). The S_i are a symmetric
    function of the exchangeable sample whenever H does not depend on the
    calibration or test data, which gives
        P(Y* in C(X*)) >= 1 - alpha        (finite-sample, marginal).
    The leaf-agreement kernels depend only on the trees fitted on the
    training split, so the condition holds.

    The set is evaluated exactly on every interval between consecutive
    calibration scores; the returned half-width is the supremum of the
    accepted values (convex hull, which can only add coverage).
    """
    n = len(r_cal)
    K = conf_index(n, alpha)
    order, rs = _sorted(r_cal)
    if K > n:
        return np.full(H_tc.shape[0], np.inf)
    Hcc = H_cc[np.ix_(order, order)]
    less = rs[None, :] < rs[:, None]
    a = (Hcc * less).sum(axis=1)          # mass strictly below, from cal points
    s0 = Hcc.sum(axis=1)                  # includes H(x_i, x_i)
    Htc = H_tc[:, order]
    p_idx = np.arange(n)[:, None]
    q = np.empty(Htc.shape[0])
    for st in range(0, Htc.shape[0], chunk):
        b = Htc[st:st + chunk]
        m = b.shape[0]
        D = s0[None, :] + b
        hv = (a[None, :] + b) / D         # S_i when v < V_i
        lv = a[None, :] / D               # S_i when v > V_i
        denom = b.sum(axis=1) + h_tt[st:st + chunk]
        c = np.concatenate([np.zeros((m, 1)), np.cumsum(b, axis=1)],
                           axis=1) / denom[:, None]      # S_{n+1} per slot
        # Scan slots from the top in blocks and keep the highest accepted
        # slot of every row. Slot 0 is always accepted (S_{n+1} = 0 there),
        # so every row is resolved. Exact, and usually only the top ~alpha
        # fraction of the slots is visited.
        last = np.full(m, -1)
        open_ = np.ones(m, dtype=bool)
        end = n + 1
        while open_.any():
            s_rng = np.arange(max(0, end - block), end)
            rows = np.nonzero(open_)[0]
            cb = c[np.ix_(rows, s_rng)]
            below = (p_idx < s_rng[None, :])[None]
            cnt = np.where(below, lv[rows][:, :, None] < cb[:, None, :],
                           hv[rows][:, :, None] < cb[:, None, :]).sum(axis=1)
            ok = cnt <= K - 1
            hit = ok.any(axis=1)
            j = len(s_rng) - 1 - np.argmax(ok[:, ::-1], axis=1)
            last[rows[hit]] = s_rng[j[hit]]
            open_[rows[hit]] = False
            end = s_rng[0]
            if end == 0:
                break
        q[st:st + chunk] = np.where(last < n, rs[np.clip(last, 0, n - 1)], np.inf)
    return q


# ------------------------------------------------------------- baselines
def mondrian(r_cal, g_cal, g_q, alpha, n_groups):
    """Mondrian (group-conditional) split conformal prediction."""
    qg = np.array([split_quantile(r_cal[g_cal == g], alpha)
                   if np.any(g_cal == g) else np.inf for g in range(n_groups)])
    return qg[g_q]


def normalized(r_cal, s_cal, s_q, alpha):
    """Normalized conformal prediction (Papadopoulos et al., JAIR 2011)."""
    return s_q * split_quantile(r_cal / s_cal, alpha)


def gaussian(mu, sigma, alpha):
    z = norm.ppf(1 - alpha / 2)
    return mu - z * sigma, mu + z * sigma


def cqr(qlo_cal, qhi_cal, y_cal, qlo_q, qhi_q, alpha):
    """Conformalized quantile regression (Romano et al., 2019)."""
    E = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
    Q = split_quantile(E, alpha)
    return qlo_q - Q, qhi_q + Q


def cv_plus(fold_pred_q, fold_of_train, resid_train, alpha):
    """CV+ (Barber et al., 2021).

    fold_pred_q: (K, n_q) predictions of the K leave-fold-out models.
    fold_of_train: fold index of every training point; resid_train: its
    out-of-fold absolute residual.
    """
    n = len(resid_train)
    mu = fold_pred_q[fold_of_train]                     # (n, n_q)
    low = mu - resid_train[:, None]
    up = mu + resid_train[:, None]
    k_lo = int(math.floor(alpha * (n + 1)))
    k_hi = conf_index(n, alpha)
    lo = (np.partition(low, k_lo - 1, axis=0)[k_lo - 1] if k_lo >= 1
          else np.full(low.shape[1], -np.inf))
    hi = (np.partition(up, k_hi - 1, axis=0)[k_hi - 1] if k_hi <= n
          else np.full(up.shape[1], np.inf))
    return lo, hi


class LoBoost:
    """Re-implementation of LoBoost (Santos et al., arXiv:2602.22432).

    Calibration points are grouped by the prefix of their leaf sequence over
    the first k trees. A region is refined with the next tree while it holds
    at least n_part points; children with fewer than m_min points are merged
    into the most similar retained region under the weighted Hamming distance
    d(R, x) = sum_{t < k_R} w_t 1[l_t(x) != l_t(R)], with decreasing tree
    weights w_t = Var[h_t] on the calibration set (or rho**t). Each region
    uses the split-conformal quantile of its own residuals, which gives
    region-conditional finite-sample coverage.
    """

    def __init__(self, m_min=50, n_part=None, weights=None):
        self.m_min = m_min
        self.n_part = n_part or 2 * m_min
        self.w = weights

    def fit(self, L_cal, r_cal, alpha):
        self.T = L_cal.shape[1]
        if self.w is None:
            self.w = 0.9 ** np.arange(self.T)
        prefixes, members, orphans = [], [], []

        stack = [(np.arange(len(r_cal)), ())]
        while stack:
            idx, pre = stack.pop()
            k = len(pre)
            if len(idx) >= self.n_part and k < self.T:
                vals, inv = np.unique(L_cal[idx, k], return_inverse=True)
                sizes = np.bincount(inv)
                dense = np.nonzero(sizes >= self.m_min)[0]
                if len(dense) == 0:
                    prefixes.append(pre)
                    members.append(list(idx))
                    continue
                for g in range(len(vals)):
                    sel = idx[inv == g]
                    if sizes[g] >= self.m_min:
                        stack.append((sel, pre + (int(vals[g]),)))
                    else:
                        orphans.extend(sel.tolist())
            else:
                prefixes.append(pre)
                members.append(list(idx))
        self.prefixes = prefixes
        self._build_prefix_arrays()
        if orphans:
            orphans = np.array(orphans)
            reg = self.assign(L_cal[orphans])
            for i, rg in zip(orphans, reg):
                members[rg].append(int(i))
        self.q = np.array([split_quantile(r_cal[np.array(m)], alpha)
                           for m in members])
        self.sizes = np.array([len(m) for m in members])
        return self

    def _build_prefix_arrays(self):
        R = len(self.prefixes)
        kmax = max((len(p) for p in self.prefixes), default=0)
        self.P = np.full((R, max(kmax, 1)), -1, dtype=np.int64)
        self.depth = np.array([len(p) for p in self.prefixes])
        for r, p in enumerate(self.prefixes):
            if p:
                self.P[r, :len(p)] = p

    def assign(self, L):
        kmax = self.P.shape[1]
        out = np.empty(L.shape[0], dtype=np.int64)
        w = self.w[:kmax]
        valid = self.P >= 0
        for s in range(0, L.shape[0], 512):
            Lq = L[s:s + 512, :kmax]
            mism = (Lq[:, None, :] != self.P[None, :, :]) & valid[None]
            d = (mism * w[None, None, :]).sum(axis=2)
            # ties: prefer the deeper (more specific) region
            d = d - 1e-9 * self.depth[None, :]
            out[s:s + 512] = np.argmin(d, axis=1)
        return out

    def halfwidth(self, L):
        return self.q[self.assign(L)]
