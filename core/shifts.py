"""Synthetic covariate shifts used for the binary detection evaluation.

Directions are unit vectors in the standardized feature space of the
training split: the first three principal components (PC1, PC2, PC3) and
the diagonal (1, ..., 1)/sqrt(d). Equal-displacement normalization: every
direction moves a point by the same Euclidean distance in standardized
space, equal to m times the standard deviation of the training data along
PC1, so m = 3 on PC1 reproduces the main 3-sigma shift of the manuscript and
the other directions are compared at the same displacement.
"""
import numpy as np


def directions(Xtr_s):
    Xc = Xtr_s - Xtr_s.mean(axis=0)
    _, svals, Vt = np.linalg.svd(Xc, full_matrices=False)
    out = {}
    for k, name in enumerate(["PC1", "PC2", "PC3"]):
        v = Vt[k].copy()
        v *= np.sign(v[np.argmax(np.abs(v))])     # deterministic sign
        out[name] = v
    d = Xtr_s.shape[1]
    out["Diag"] = np.ones(d) / np.sqrt(d)
    ref = svals[0] / np.sqrt(len(Xtr_s) - 1)       # std along PC1
    return out, ref


def continuous_mask(Xtr_raw, min_unique=11):
    """Features that are neither binary nor small-cardinality integers."""
    mask = np.zeros(Xtr_raw.shape[1], dtype=bool)
    for j in range(Xtr_raw.shape[1]):
        col = Xtr_raw[:, j]
        integer = np.allclose(col, np.round(col))
        mask[j] = len(np.unique(col)) >= min_unique and not integer
    return mask


def shift(X_raw, scaler, v, mag, ref, feature_mask=None):
    """Move points by mag * ref along unit direction v (standardized space).
    With a feature mask, v is restricted to the allowed features and
    re-normalized so the displacement length is unchanged."""
    v = v.copy()
    if feature_mask is not None:
        v[~feature_mask] = 0.0
        nv = np.linalg.norm(v)
        if nv == 0:
            return None
        v /= nv
    Xs = scaler.transform(X_raw) + mag * ref * v[None, :]
    return scaler.inverse_transform(Xs)
