"""Mixture of factor analysers with P-spline age-varying centres.

    z_i ~ Categorical(pi)
    theta_i ~ N(0, I_q)
    x_i | z_i = k, theta_i ~ N(mu_k(t_i) + W_k theta_i, Psi_k),  mu_k(t) = B(t) beta_k

Integrating out the scores gives the mixture actually fitted:

    p(x_i | t_i) = sum_k pi_k N(x_i; B(t_i) beta_k, W_k W_k' + Psi_k)

Fitted by ECM. The Woodbury identity and matrix determinant lemma keep every
covariance operation at q x q, so cost is O(N K D q) rather than O(D^3).

Shapes: X (N, D), B (N, M), pi (K,), beta (K, M, D), W (K, D, q), psi (K, D),
lam (K, D). Currently group-specific W and Psi; the shared variants are the
next increment.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from jax.scipy.special import logsumexp

LOG2PI = float(jnp.log(2.0 * jnp.pi))


class Params(NamedTuple):
    log_pi: jnp.ndarray   # (K,)
    beta: jnp.ndarray     # (K, M, D)
    W: jnp.ndarray        # (K, D, q)
    psi: jnp.ndarray      # (K, D)
    lam: jnp.ndarray      # (K, D)


class Hyper(NamedTuple):
    P: jnp.ndarray        # (M, M) difference penalty
    rank: int             # rank of P, i.e. M - order
    a0: float = 1e-3      # Gamma shape on lambda
    b0: float = 1e-3      # Gamma rate on lambda
    psi_floor: float = 1e-6   # times the marginal variance of each coordinate


# ---------------------------------------------------------------- E step ----

def _woodbury(W, psi):
    """Pieces shared by the density and the score moments.

    A = I_q + W' Psi^-1 W      (K, q, q)
    G = A^-1                   = Cov[theta | x, k], independent of i
    logdet C = sum(log psi) + logdet A
    """
    K, D, q = W.shape
    WtPinv = jnp.swapaxes(W, 1, 2) / psi[:, None, :]          # (K, q, D)
    A = jnp.eye(q)[None] + WtPinv @ W                          # (K, q, q)
    L = jnp.linalg.cholesky(A)                                 # batched, lower
    # `True` stays a static python bool here; passing cho_factor's tuple through
    # vmap would turn the `lower` flag into a tracer.
    G = jax.vmap(lambda l: cho_solve((l, True), jnp.eye(q)))(L)
    logdet_A = 2.0 * jnp.log(jnp.diagonal(L, axis1=1, axis2=2)).sum(-1)
    logdet_C = jnp.log(psi).sum(-1) + logdet_A                 # (K,)
    return WtPinv, G, logdet_C


def e_step(X, B, p: Params):
    """Responsibilities, conditional score means, and the observed log likelihood."""
    K, M, D = p.beta.shape
    q = p.W.shape[2]

    mu = jnp.einsum("nm,kmd->knd", B, p.beta)                  # (K, N, D)
    e = X[None] - mu                                           # (K, N, D)

    WtPinv, G, logdet_C = _woodbury(p.W, p.psi)
    y = jnp.einsum("kqd,knd->knq", WtPinv, e)                  # (K, N, q)
    m = jnp.einsum("kqp,knp->knq", G, y)                       # E[theta | x, k]

    quad = (e ** 2 / p.psi[:, None, :]).sum(-1) - (y * m).sum(-1)   # (K, N)
    log_comp = -0.5 * (D * LOG2PI + logdet_C[:, None] + quad) + p.log_pi[:, None]
    ll_i = logsumexp(log_comp, axis=0)                         # (N,)
    r = jnp.exp(log_comp - ll_i[None, :])                      # (K, N)
    return r, m, G, ll_i.sum()


# ---------------------------------------------------------------- M step ----

def m_step(X, B, p: Params, h: Hyper, r, m, G, var_floor):
    """One conditional-maximisation sweep: (beta, W) jointly, then psi, then lam."""
    K, M, D = p.beta.shape
    q = p.W.shape[2]
    Mq = M + q

    Nk = r.sum(1)                                              # (K,)
    log_pi = jnp.log(Nk / r.shape[1])

    # Augmented design cross-products. The score block uses E[theta theta'],
    # not the outer product of the means, so posterior score uncertainty is
    # retained (Rubin & Thayer 1982).
    rB = r[:, :, None] * B[None]                               # (K, N, M)
    S_bb = jnp.einsum("knm,nl->kml", rB, B)                    # (K, M, M)
    S_bm = jnp.einsum("knm,knq->kmq", rB, m)                   # (K, M, q)
    S_mm = jnp.einsum("kn,knq,knp->kqp", r, m, m) + Nk[:, None, None] * G
    S = jnp.concatenate([jnp.concatenate([S_bb, S_bm], axis=2),
                         jnp.concatenate([jnp.swapaxes(S_bm, 1, 2), S_mm], axis=2)],
                        axis=1)                                # (K, M+q, M+q)

    c = jnp.concatenate([jnp.einsum("knm,nd->kmd", rB, X),
                         jnp.einsum("kn,knq,nd->kqd", r, m, X)], axis=1)  # (K, M+q, D)

    # Penalty on the spline block only, carrying the psi factor (raised in review).
    P_aug = jnp.zeros((Mq, Mq)).at[:M, :M].set(h.P)
    scale = (p.psi * p.lam)                                    # (K, D)

    def solve_kd(S_k, c_kd, s_kd):
        return jnp.linalg.solve(S_k + s_kd * P_aug, c_kd)
    theta = jax.vmap(jax.vmap(solve_kd, in_axes=(None, 1, 0), out_axes=1),
                     in_axes=(0, 0, 0), out_axes=0)(S, c, scale)   # (K, M+q, D)

    beta = theta[:, :M, :]                                     # (K, M, D)
    W = jnp.swapaxes(theta[:, M:, :], 1, 2)                    # (K, D, q)

    # psi: responsibility-weighted mean squared residual plus the score-uncertainty
    # trace term, floored so variances cannot collapse.
    mu = jnp.einsum("nm,kmd->knd", B, beta)
    pred = mu + jnp.einsum("knq,kdq->knd", m, W)
    resid2 = jnp.einsum("kn,knd->kd", r, (X[None] - pred) ** 2)
    trace = jnp.einsum("kdq,kqp,kdp->kd", W, G, W) * Nk[:, None]
    psi = jnp.maximum((resid2 + trace) / Nk[:, None], h.psi_floor * var_floor[None, :])

    # lambda: posterior mode under Gamma(a0, b0), using rank M - order.
    quad_pen = jnp.einsum("kmd,ml,kld->kd", beta, h.P, beta)
    lam = (0.5 * h.rank + h.a0 - 1.0) / (0.5 * quad_pen + h.b0)
    lam = jnp.maximum(lam, 1e-12)

    return Params(log_pi=log_pi, beta=beta, W=W, psi=psi, lam=lam)


# ------------------------------------------------------------- objective ----

def penalised_loglik(ll, p: Params, h: Hyper):
    """The quantity ECM must increase monotonically."""
    quad_pen = jnp.einsum("kmd,ml,kld->kd", p.beta, h.P, p.beta)
    log_lam = jnp.log(p.lam)
    prior = (0.5 * h.rank * log_lam - 0.5 * p.lam * quad_pen
             + (h.a0 - 1.0) * log_lam - h.b0 * p.lam).sum()
    return ll + prior


@jax.jit
def _step(carry, _, X, B, h, var_floor):
    p = carry
    r, m, G, ll = e_step(X, B, p)
    p_new = m_step(X, B, p, h, r, m, G, var_floor)
    return p_new, penalised_loglik(ll, p, h)


def fit(X, B, p0: Params, h: Hyper, n_iter: int = 200):
    """Run n_iter ECM sweeps. Returns final parameters and the objective trace."""
    X = jnp.asarray(X); B = jnp.asarray(B)
    var_floor = jnp.var(X, axis=0)

    def body(p, _):
        r, m, G, ll = e_step(X, B, p)
        obj = penalised_loglik(ll, p, h)
        return m_step(X, B, p, h, r, m, G, var_floor), obj

    p_final, trace = jax.lax.scan(body, p0, None, length=n_iter)
    return p_final, trace


# ------------------------------------------------------------ prediction ----

def log_predictive_density(X, B, p: Params):
    """Sum of log mixture densities — the held-out model-selection score."""
    _, _, _, ll = e_step(jnp.asarray(X), jnp.asarray(B), p)
    return ll


def responsibilities(X, B, p: Params):
    """(N, K) posterior probability of group membership per infant."""
    r, _, _, _ = e_step(jnp.asarray(X), jnp.asarray(B), p)
    return jnp.swapaxes(r, 0, 1)


# --------------------------------------------------- init and simulation ----

def init_params(X, B, K, q, rng, h: Hyper, jitter: float = 0.1) -> Params:
    """Initialisation per the methods supplement.

    A linear age effect is removed before k-means so the initial grouping is not
    determined solely by age; loadings start from the leading principal
    components of within-group residuals and uniquenesses from residual
    variances. `jitter` perturbs the k-means seeding so repeated calls with
    different rngs give genuinely different starts.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.asarray(X); B = np.asarray(B)
    N, D = X.shape
    M = B.shape[1]

    # 1. de-trend on the spline basis, then cluster the residuals
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    resid = X - B @ coef
    if K == 1:
        labels = np.zeros(N, dtype=int)
    else:
        seed = int(rng.integers(0, 2**31 - 1))
        labels = KMeans(K, n_init=5, random_state=seed).fit_predict(
            resid + jitter * resid.std() * rng.standard_normal(resid.shape))

    beta = np.zeros((K, M, D)); W = np.zeros((K, D, q)); psi = np.zeros((K, D))
    P = np.asarray(h.P)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < M + 2:                      # degenerate start, fall back
            idx = np.arange(N)
        Bk, Xk = B[idx], X[idx]
        beta[k] = np.linalg.solve(Bk.T @ Bk + 1e-2 * P, Bk.T @ Xk)
        rk = Xk - Bk @ beta[k]
        if q > 0:
            U, S, Vt = np.linalg.svd(rk - rk.mean(0), full_matrices=False)
            W[k] = (Vt[:q].T * (S[:q] / np.sqrt(max(len(idx) - 1, 1))))
        psi[k] = np.maximum(rk.var(0), 1e-6 * X.var(0))

    counts = np.bincount(labels, minlength=K).astype(float) + 1.0
    log_pi = np.log(counts / counts.sum())
    quad = np.einsum("kmd,ml,kld->kd", beta, P, beta)
    lam = (0.5 * h.rank + h.a0 - 1.0) / (0.5 * quad + h.b0)

    return Params(log_pi=jnp.asarray(log_pi), beta=jnp.asarray(beta),
                  W=jnp.asarray(W), psi=jnp.asarray(psi),
                  lam=jnp.asarray(np.maximum(lam, 1e-12)))


def simulate(rng, B, p: Params):
    """Draw a dataset from the generative model — used to validate recovery."""
    import numpy as np
    B = np.asarray(B); N, M = B.shape
    pi = np.asarray(jnp.exp(p.log_pi)); beta = np.asarray(p.beta)
    W = np.asarray(p.W); psi = np.asarray(p.psi)
    K, D, q = W.shape
    z = rng.choice(K, size=N, p=pi / pi.sum())
    theta = rng.standard_normal((N, q))
    X = np.empty((N, D))
    for i in range(N):
        k = z[i]
        mean = B[i] @ beta[k] + (W[k] @ theta[i] if q else 0.0)
        X[i] = mean + rng.standard_normal(D) * np.sqrt(psi[k])
    return X, z
