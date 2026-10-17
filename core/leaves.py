"""Leaf populations, LPDS variants and leaf-agreement kernels."""
import numpy as np
import scipy.sparse as sp


class LeafStructure:
    """Per-leaf training populations of a fitted ensemble.

    Leaf ids of tree t are offset into one flat index so that all trees share
    a single count vector. Counts are obtained by routing the full training
    set (Definition 2 of the paper).
    """

    def __init__(self, n_ids, L_train):
        self.n_ids = np.asarray(n_ids, dtype=np.int64)
        self.T = len(self.n_ids)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_ids)[:-1]])
        self.n_total = int(self.n_ids.sum())
        self.N = L_train.shape[0]
        self.L_train = L_train
        self.counts = np.bincount(self.flat(L_train).ravel(),
                                  minlength=self.n_total).astype(float)

    def flat(self, L):
        return L + self.offsets[None, :]

    # ---------------------------------------------------------------- LPDS
    def lpds(self, L, counts=None, tree_weights=None):
        """LPM, named LPDS in the code (Eqs. 3 and 4 of the paper). With `counts` given (e.g. counts stored in the model
        file), each tree is normalized by its own total so that in-bag counts
        are expressed as fractions of that tree's subsample. `tree_weights`
        may be a vector (T,) or a per-query matrix (n, T)."""
        if counts is None:
            frac = self.counts[self.flat(L)] / self.N
        else:
            tot = np.add.reduceat(counts, self.offsets)
            frac = counts[self.flat(L)] / tot[None, :]
        if tree_weights is None:
            return frac.mean(axis=1)
        w = np.asarray(tree_weights, dtype=float)
        if w.ndim == 1:
            w = np.broadcast_to(w, frac.shape)
        return (frac * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-300)

    def tree_output_variance(self, leaf_values_flat):
        """Variance over the training set of each tree's output, Var[h_t]."""
        vals = leaf_values_flat[self.flat(self.L_train)]
        return vals.var(axis=0)

    def leaf_target_var(self, y_train, L):
        """Mean over trees of the within-leaf variance of training targets."""
        yc = y_train - y_train.mean()
        idx = self.flat(self.L_train).ravel()
        cnt = np.maximum(self.counts, 1.0)
        s1 = np.bincount(idx, weights=np.repeat(yc, self.T), minlength=self.n_total)
        s2 = np.bincount(idx, weights=np.repeat(yc ** 2, self.T), minlength=self.n_total)
        var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0)
        return var[self.flat(L)].mean(axis=1)

    # ------------------------------------------------------------- kernels
    def onehot(self, L, weights=None):
        n = L.shape[0]
        rows = np.repeat(np.arange(n), self.T)
        cols = self.flat(L).ravel()
        data = np.ones(n * self.T) if weights is None else weights[cols]
        return sp.csr_matrix((data, (rows, cols)), shape=(n, self.n_total))

    def kernel(self, L1, L2, kind="la", gamma=1.0):
        """Leaf-agreement kernel between two sets of leaf assignments.

        la       LA(x, z) = (1/T) sum_t 1[l_t(x) = l_t(z)]            (Eq. 5)
        la_norm  (1/T) sum_t 1[l_t(x) = l_t(z)] / n_t(l_t(x)), per-tree
                 normalization by the training population of the shared leaf (Eq. 6).
        Both are symmetric and depend only on the training split, so they are
        fixed with respect to calibration and test data. gamma sharpens.
        """
        A1 = self.onehot(L1)
        if kind == "la":
            A2 = self.onehot(L2)
        elif kind == "la_norm":
            A2 = self.onehot(L2, weights=1.0 / np.maximum(self.counts, 1.0))
        else:
            raise ValueError(kind)
        K = (A1 @ A2.T).toarray() / self.T
        if gamma != 1.0:
            K = K ** gamma
        return K

    def self_kernel(self, L, kind="la", gamma=1.0):
        if kind == "la":
            k = np.ones(L.shape[0])
        else:
            k = (1.0 / np.maximum(self.counts, 1.0))[self.flat(L)].mean(axis=1)
        return k ** gamma

    def affinity_counts(self, Lq, Lref):
        """Number of trees in which a query shares a leaf with each reference
        point (the IBUG affinity), shape (n_q, n_ref), integer valued."""
        return (self.onehot(Lq) @ self.onehot(Lref).T).toarray()
