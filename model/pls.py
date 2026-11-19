"""Partial least squares correlation (PLS-C; McIntosh & Lobaugh 2004;
Krishnan et al. 2011) with family-aware resampling.

Two blocks X (n, p) and Y (n, k), each column z-scored. The cross-correlation
matrix R = X'Y / (n - 1) is decomposed by SVD: R = U S V'. Each latent
variable (LV) l has saliences U[:, l] (X weights) and V[:, l] (Y weights),
and latent scores X U[:, l], Y V[:, l]. The share of the cross-block
covariance for LV l is s_l^2 / sum(s^2).

Inference:
- Permutation (significance of each LV): Y rows are permuted between whole
  families of the same size (twins move as a pair, singletons as singletons).
  LV l is compared with the largest singular value of the permuted R after
  the observed LVs 1..l-1 are projected out (deflation).
  p_l = (1 + #{s*_l >= s_l}) / (1 + B).
- Bootstrap (stability of each salience): whole families are resampled with
  replacement. Each bootstrap LV is sign-matched to the original (default,
  align_method="sign"). Procrustes rotation (Milan & Whittaker 1995) is
  available but mis-calibrated here: fitted to V alone it is exact when V is
  square (bootstrap SD 0), and fitted to U and V stacked it makes the
  bootstrap SD about 40% too small; sign matching gives bootstrap SD / true
  sampling SD = 1.01-1.09 (scripts/check_pls_bootstrap.py, 2026-11-19). Bootstrap ratio = salience /
  bootstrap SD (read like a z-score). Structure correlations (corr of each
  variable with its block's latent score) get bootstrap 95% percentile CIs.
"""

import numpy as np


def zscore(A):
    A = np.asarray(A, float)
    return (A - A.mean(0)) / A.std(0, ddof=1)


def svd_pls(X, Y):
    """X, Y already z-scored. Returns U (p, L), s (L,), V (k, L), signs fixed so
    the largest |salience| in U is positive."""
    R = X.T @ Y / (len(X) - 1)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    V = Vt.T
    sign = np.sign(U[np.abs(U).argmax(0), np.arange(U.shape[1])])
    sign[sign == 0] = 1
    return U * sign, s, V * sign


def family_blocks(families):
    """Row indices of each family, grouped by family size: {size: [idx arrays]}."""
    fam = np.asarray(families)
    _, inv = np.unique(fam, return_inverse=True)
    blocks = {}
    for f in range(inv.max() + 1):
        idx = np.where(inv == f)[0]
        blocks.setdefault(len(idx), []).append(idx)
    return blocks


def permute_rows(blocks, n, rng):
    """A permutation of 0..n-1 that swaps whole families of the same size (and
    shuffles members within each family)."""
    perm = np.empty(n, int)
    for size, fams in blocks.items():
        order = rng.permutation(len(fams))
        for src, dst in zip(fams, (fams[o] for o in order)):
            perm[src] = rng.permutation(dst)
    return perm


def bootstrap_rows(blocks, rng):
    fams = [f for fs in blocks.values() for f in fs]
    pick = rng.integers(0, len(fams), len(fams))
    return np.concatenate([fams[i] for i in pick])


def procrustes(A, B):
    """Orthogonal Q minimising ||A Q - B||."""
    W, _, Zt = np.linalg.svd(A.T @ B)
    return W @ Zt


def structure(X, Y, U, V):
    """Corr of each X column with X U, and of each Y column with Y V."""
    lx, ly = X @ U, Y @ V
    rx = zscore(X).T @ zscore(lx) / (len(X) - 1)
    ry = zscore(Y).T @ zscore(ly) / (len(Y) - 1)
    return rx, ry


def align(u, v, U, V, method):
    """Match a bootstrap solution (u, v) to the original (U, V)."""
    if method == "sign":
        sg = np.sign((u * U).sum(0) + (v * V).sum(0))
        sg[sg == 0] = 1
        return u * sg, v * sg
    if method == "stacked":
        Q = procrustes(np.vstack([u, v]), np.vstack([U, V]))
    elif method == "v":
        Q = procrustes(v, V)
    else:
        raise ValueError(method)
    return u @ Q, v @ Q


def pls(X, Y, families, n_perm=10_000, n_boot=10_000, seed=0, align_method="sign"):
    X, Y = zscore(X), zscore(Y)
    n = len(X)
    rng = np.random.default_rng(seed)
    U, s, V = svd_pls(X, Y)
    L = len(s)
    blocks = family_blocks(families)

    # Deflated permutation test: LV l is tested against the largest singular
    # value of the permuted R after the observed LVs 1..l-1 are projected out of
    # both sides. Comparing s_l with the l-th permuted singular value instead
    # makes LVs after a real LV1 anti-conservative (the structured noise that
    # is LV1 under the null moves to LV2 in the data; scripts/check_pls.py).
    Px = [np.eye(U.shape[0]) - U[:, :l] @ U[:, :l].T for l in range(L)]
    Py = [np.eye(V.shape[0]) - V[:, :l] @ V[:, :l].T for l in range(L)]
    s_perm = np.empty((n_perm, L))
    for b in range(n_perm):
        Rp = X.T @ Y[permute_rows(blocks, n, rng)] / (n - 1)
        s_perm[b] = [np.linalg.norm(Px[l] @ Rp @ Py[l], 2) for l in range(L)]
    p_perm = (1 + (s_perm >= s).sum(0)) / (1 + n_perm)

    Ub = np.empty((n_boot,) + U.shape)
    Vb = np.empty((n_boot,) + V.shape)
    rxb = np.empty((n_boot,) + U.shape)
    ryb = np.empty((n_boot,) + V.shape)
    for b in range(n_boot):
        idx = bootstrap_rows(blocks, rng)
        Xb, Yb = zscore(X[idx]), zscore(Y[idx])
        u, _, v = svd_pls(Xb, Yb)
        # "v": Procrustes on V alone; degenerate when V is square (e.g. 10
        # behaviour scales, 10 LVs): it maps any bootstrap solution exactly onto
        # V, so the bootstrap SD is 0. "stacked": one rotation for U and V.
        # "sign": no rotation, only the sign of each LV. scripts/check_pls.py
        # (checks 4-5) compares them with the true sampling SD.
        u, v = align(u, v, U, V, align_method)
        Ub[b], Vb[b] = u, v
        rxb[b], ryb[b] = structure(Xb, Yb, u, v)
    rx, ry = structure(X, Y, U, V)
    return dict(U=U, s=s, V=V, cov_share=s ** 2 / (s ** 2).sum(), p_perm=p_perm,
                s_perm_95=np.percentile(s_perm, 95, axis=0),
                bsr_x=U / Ub.std(0, ddof=1), bsr_y=V / Vb.std(0, ddof=1),
                rx=rx, ry=ry,
                rx_ci=np.percentile(rxb, [2.5, 97.5], axis=0), ry_ci=np.percentile(ryb, [2.5, 97.5], axis=0),
                n=n, n_families=sum(len(f) for f in blocks.values()))


def replicate(Xtr, Ytr, Xte, Yte, fam_te, n_perm=10_000, seed=1):
    """Fit saliences in the training infants, apply them to the validation
    infants (z-scored with training means/SDs), and test corr(X U, Y V) per LV
    with a family-block permutation in the validation infants."""
    mx, sx = Xtr.mean(0), Xtr.std(0, ddof=1)
    my, sy = Ytr.mean(0), Ytr.std(0, ddof=1)
    U, s, V = svd_pls((Xtr - mx) / sx, (Ytr - my) / sy)
    lx, ly = ((Xte - mx) / sx) @ U, ((Yte - my) / sy) @ V
    r = np.array([np.corrcoef(lx[:, l], ly[:, l])[0, 1] for l in range(len(s))])
    rng = np.random.default_rng(seed)
    blocks = family_blocks(fam_te)
    null = np.empty((n_perm, len(s)))
    for b in range(n_perm):
        p = permute_rows(blocks, len(lx), rng)
        null[b] = [np.corrcoef(lx[:, l], ly[p, l])[0, 1] for l in range(len(s))]
    return dict(r=r, p=(1 + (null >= r).sum(0)) / (1 + n_perm), U=U, V=V, n=len(lx))
