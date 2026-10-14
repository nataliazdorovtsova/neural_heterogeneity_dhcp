"""Cubic B-spline basis and the second-difference penalty (SUPPLEMENT.md S8)."""

import numpy as np
from scipy.interpolate import BSpline

LO, HI = 26.5, 45.5      # observed PMA is 26.71-45.14; 26-45 would extrapolate
DEGREE = 3
M_DEFAULT = 20


def basis(t, M: int = M_DEFAULT, lo: float = LO, hi: float = HI,
          degree: int = DEGREE) -> np.ndarray:
    """(n, M) design matrix of cubic B-splines with equally spaced knots.

    M basis functions need M - degree + 1 breakpoints, with the boundary knots
    repeated degree + 1 times.
    """
    if M <= degree:
        raise ValueError(f"M must exceed the spline degree ({degree})")
    interior = np.linspace(lo, hi, M - degree + 1)[1:-1]
    knots = np.r_[[lo] * (degree + 1), interior, [hi] * (degree + 1)]
    t = np.clip(np.asarray(t, dtype=float), lo, hi)
    B = np.asarray(BSpline.design_matrix(t, knots, degree).todense())
    assert B.shape[1] == M, f"expected {M} basis functions, built {B.shape[1]}"
    return B


def linear_basis(t, lo: float = LO, hi: float = HI) -> np.ndarray:
    """(n, 2) intercept-and-slope design for the 'linear age effect' variant.

    A review comment: the linear variant needs a genuine two-term linear
    basis, not two cubic B-splines.
    """
    t = np.asarray(t, dtype=float)
    return np.column_stack([np.ones_like(t), (t - lo) / (hi - lo)])


def difference_penalty(M: int, order: int = 2) -> np.ndarray:
    """P = D' D for the order-th difference operator.

    Rank is M - order, so the prior is improper: it leaves the overall level and
    slope unpenalised. The smoothness update uses that rank, not M.
    """
    D = np.diff(np.eye(M), n=order, axis=0)
    return D.T @ D


def penalty_rank(M: int, order: int = 2) -> int:
    return max(M - order, 0)
