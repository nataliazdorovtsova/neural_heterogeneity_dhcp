"""Regional measures and fold-wise standardisation (DECISIONS.md D5, D6).

Definitions are pinned here, not inherited from ab15's earlier topology run,
which she has confirmed is not authoritative.
"""

import numpy as np
from scipy.linalg import expm

# Transform choices are empirical, not inherited from the methods supplement.
# See DECISIONS.md D10: after degree normalisation, communicability is already
# symmetric and the log makes it WORSE; clustering is the strongly skewed one.
MEASURES = ("strength", "log_clustering", "communicability")

CLUSTERING_EPS = 1e-4   # below the smallest observed nonzero (4.6e-4)


def strength(W: np.ndarray) -> np.ndarray:
    """Weighted row sum. On a binarised matrix this is the degree."""
    return W.sum(1)


def clustering_onnela(W: np.ndarray) -> np.ndarray:
    """Onnela weighted clustering, normalised by the PER-SUBJECT maximum.

    c_i = (W13 @ W13 @ W13)_ii / (k_i (k_i - 1)),  W13 = (W / max W) ** (1/3)

    Onnela is used because it is scale-free; Barrat is not. Matches
    networkx.clustering(..., weight=...), which is verified in the extraction
    script rather than assumed.
    """
    w = W / W.max()
    w13 = np.cbrt(w)
    np.fill_diagonal(w13, 0.0)
    tri = np.diagonal(w13 @ w13 @ w13)
    k = (W > 0).sum(1).astype(float)
    denom = k * (k - 1)
    return np.divide(tri, denom, out=np.zeros_like(tri), where=denom > 0)


def communicability(W: np.ndarray) -> np.ndarray:
    """Mean off-diagonal communicability per region.

    Degree-normalised (Crofts & Higham 2009) before the matrix exponential:
    expm on raw mu weights is dominated by exp(lambda_max) and spans four orders
    of magnitude, which makes it useless as a per-subject feature. The diagonal
    (self-communicability) is excluded because it dominates the row mean.
    """
    s = W.sum(1)
    d = np.sqrt(np.where(s > 0, s, 1.0))
    C = expm(W / np.outer(d, d))
    np.fill_diagonal(C, np.nan)
    return np.nanmean(C, axis=1)


def regional_features(W: np.ndarray, log_clustering: bool = True,
                      log_communicability: bool = False) -> np.ndarray:
    """(90, 3) table for one connectome, columns ordered as MEASURES.

    Defaults follow DECISIONS.md D10. Both transforms stay switchable so the
    supplement's original choice can be reproduced by flipping them.
    """
    c = clustering_onnela(W)
    c = np.log(c + CLUSTERING_EPS) if log_clustering else c
    q = communicability(W)
    q = np.log(q) if log_communicability else q
    return np.column_stack([strength(W), c, q])


def standardise(X_fit: np.ndarray, *others: np.ndarray):
    """z-score coordinates using X_fit's moments only (DECISIONS.md D6).

    X arrays are (n, 270). The moments come from the fitting fold alone, so no
    information leaks from held-out infants. Returns the standardised arrays in
    the order given, followed by (mean, sd).
    """
    mu = X_fit.mean(0)
    sd = X_fit.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    out = [(X_fit - mu) / sd] + [(X - mu) / sd for X in others]
    return (*out, mu, sd)
