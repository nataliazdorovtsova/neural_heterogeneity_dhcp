"""One-group factor model whose spread changes with age (RESEARCH_PLAN_BIN10 step 3b).

    theta_i ~ N(0, I_q),  eps_i ~ N(0, Psi)
    x_i = B(t_i) beta + W S_i theta_i + s0_i eps_i
    S_i = diag(s_i1, ..., s_iq),  log s_ik = Z_i g_k,  log s0_i = Z_i g_0

so that the covariance at age t is C(t) = W S(t)^2 W' + s0(t)^2 Psi.

Layers (RESEARCH_PLAN_BIN10 section 4b):
    0  no age scaling (g = 0): the step 2 model, C = W W' + Psi
    1  one scale curve shared by every axis and by Psi: C(t) = s(t)^2 (W W' + Psi)
    2  one curve for Psi and one per axis (q + 1 curves)

Z holds the scale design, by default [u, u^2] with u = (t - 40) / 5, so every
curve is 0 at 40 weeks and W, Psi describe the spread at 40 weeks. Extra
columns (e.g. log motion) can be appended for checks.

Fitting is ECM, as model/mfa.py. Dividing infant i's data and design row by
s0_i gives an ordinary factor model whose loadings are W R_i, with
R_i = S_i / s0_i known in the E step. So the (beta, W) and psi steps keep
their form, with each infant's posterior score covariance G_i now its own
q x q matrix. The scale coefficients g are then updated by damped Newton
steps on the observed log likelihood plus a N(0, tau^2) prior (an ECME step):
every candidate step size is evaluated and the best one kept, including a
zero step, so the objective never decreases.

Shapes: X (N, D), B (N, M), Z (N, J), beta (M, D), W (D, q), psi (D,),
lam (D,), g (L, J) with L = 1 (layers 0, 1) or q + 1 (layer 2).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from model import mfa

jax.config.update("jax_enable_x64", True)
LOG2PI = float(np.log(2.0 * np.pi))
T_REF, T_UNIT = 40.0, 5.0
STEPS = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.0])   # Newton step sizes tried
N_NEWTON = 2                                             # Newton steps per sweep


class Params(NamedTuple):
    beta: jnp.ndarray     # (M, D)
    W: jnp.ndarray        # (D, q)
    psi: jnp.ndarray      # (D,)
    lam: jnp.ndarray      # (D,)
    g: jnp.ndarray        # (L, J)


class Hyper(NamedTuple):
    P: jnp.ndarray            # (M, M) difference penalty
    rank: int
    a0: float = 1e-3
    b0: float = 1e-3
    psi_floor: float = 1e-6
    tau: float = 1.0          # prior SD of each scale coefficient


HINGES = (36.0, 32.0)     # weeks; the "hinge" curve may bend at each


def scale_design(t, extra=None, curve="quad"):
    """(N, J) scale design, every column 0 at 40 weeks, plus any extra columns.

    quad   [u, u^2] with u = (t - 40) / 5
    hinge  [u, (36 - t)_+ / 5, (32 - t)_+ / 5]: log-linear from 36 weeks up,
           with its own slope in 32-36 and below 32 weeks
    """
    t = np.asarray(t, float)
    u = (t - T_REF) / T_UNIT
    if curve == "quad":
        Z = np.column_stack([u, u ** 2])
    elif curve == "hinge":
        Z = np.column_stack([u] + [np.maximum(0.0, (k - t) / T_UNIT) for k in HINGES])
    else:
        raise ValueError(curve)
    return Z if extra is None else np.column_stack([Z, extra])


def curve_of(g):
    """The curve of a fitted g without extra columns: 2 columns quad, 3 hinge."""
    return "quad" if np.shape(g)[1] == 2 else "hinge"


def n_curves(layer, q):
    return q + 1 if layer == 2 else 1


def log_scales(g, Z, layer, q):
    """log s0 (N,) and log s (N, q) for each infant."""
    ls0 = Z @ g[0]
    if layer == 0:
        ls0 = jnp.zeros_like(ls0)
        return ls0, jnp.zeros((Z.shape[0], q))
    if layer == 1:
        return ls0, jnp.repeat(ls0[:, None], q, axis=1)
    return ls0, Z @ g[1:].T


# ---------------------------------------------------------------- E step ----

def _posterior(X, B, Z, p: Params, layer, q):
    """Per-infant log density, posterior score moments and pieces for the M step."""
    D = X.shape[1]
    ls0, ls = log_scales(p.g, Z, layer, q)
    s0 = jnp.exp(ls0)
    Xt, Bt = X / s0[:, None], B / s0[:, None]
    e = Xt - Bt @ p.beta                                       # (N, D)
    quad0 = (e ** 2 / p.psi).sum(1)
    if q == 0:
        ll = -0.5 * (D * LOG2PI + jnp.log(p.psi).sum() + quad0) - D * ls0
        return ll, dict(Xt=Xt, Bt=Bt, e=e)
    R = jnp.exp(ls - ls0[:, None])                             # (N, q)
    WtPinv = p.W.T / p.psi                                     # (q, D)
    M0 = WtPinv @ p.W                                          # (q, q)
    A = jnp.eye(q)[None] + R[:, :, None] * M0[None] * R[:, None, :]
    L = jnp.linalg.cholesky(A)
    G = jax.vmap(lambda l: jax.scipy.linalg.cho_solve((l, True), jnp.eye(q)))(L)
    logdet_A = 2.0 * jnp.log(jnp.diagonal(L, axis1=1, axis2=2)).sum(-1)
    y = R * (e @ WtPinv.T)                                     # (N, q)
    m = jnp.einsum("nqp,np->nq", G, y)                         # E[theta | x]
    quad = quad0 - (y * m).sum(1)
    ll = -0.5 * (D * LOG2PI + jnp.log(p.psi).sum() + logdet_A + quad) - D * ls0
    Ev = R * m                                                 # E[S theta / s0]
    Cv = R[:, :, None] * G * R[:, None, :]                     # Cov[S theta / s0]
    return ll, dict(Xt=Xt, Bt=Bt, e=e, Ev=Ev, Cv=Cv)


def pointwise_log_density(X, B, Z, p: Params, layer, q):
    ll, _ = _posterior(jnp.asarray(X), jnp.asarray(B), jnp.asarray(Z), p, layer, q)
    return ll


# ---------------------------------------------------------------- M step ----

def _m_beta_w_psi(p: Params, h: Hyper, post, q, var_floor):
    Xt, Bt = post["Xt"], post["Bt"]
    M = Bt.shape[1]
    N = Xt.shape[0]
    if q == 0:
        S, c = Bt.T @ Bt, Bt.T @ Xt
        P_aug = h.P
    else:
        Ev, Cv = post["Ev"], post["Cv"]
        S_bv = Bt.T @ Ev
        S_vv = Ev.T @ Ev + Cv.sum(0)
        S = jnp.block([[Bt.T @ Bt, S_bv], [S_bv.T, S_vv]])
        c = jnp.concatenate([Bt.T @ Xt, Ev.T @ Xt], axis=0)    # (M + q, D)
        P_aug = jnp.zeros((M + q, M + q)).at[:M, :M].set(h.P)
    scale = p.psi * p.lam
    theta = jax.vmap(lambda c_d, s_d: jnp.linalg.solve(S + s_d * P_aug, c_d),
                     in_axes=(1, 0), out_axes=1)(c, scale)     # (M + q, D)
    beta = theta[:M]
    if q == 0:
        W = p.W
        resid2 = ((Xt - Bt @ beta) ** 2).sum(0)
        trace = 0.0
    else:
        W = theta[M:].T                                        # (D, q)
        resid2 = ((Xt - Bt @ beta - Ev @ W.T) ** 2).sum(0)
        trace = jnp.einsum("dq,nqp,dp->d", W, Cv, W)
    psi = jnp.maximum((resid2 + trace) / N, h.psi_floor * var_floor)
    quad_pen = jnp.einsum("md,ml,ld->d", beta, h.P, beta)
    lam = jnp.maximum((0.5 * h.rank + h.a0 - 1.0) / (0.5 * quad_pen + h.b0), 1e-12)
    return p._replace(beta=beta, W=W, psi=psi, lam=lam)


def _prior(p: Params, h: Hyper):
    quad_pen = jnp.einsum("md,ml,ld->d", p.beta, h.P, p.beta)
    log_lam = jnp.log(p.lam)
    lam_prior = (0.5 * h.rank * log_lam - 0.5 * p.lam * quad_pen
                 + (h.a0 - 1.0) * log_lam - h.b0 * p.lam).sum()
    return lam_prior - 0.5 * (p.g ** 2).sum() / h.tau ** 2


def objective(X, B, Z, p: Params, h: Hyper, layer, q):
    ll, _ = _posterior(X, B, Z, p, layer, q)
    return ll.sum() + _prior(p, h)


def _m_g(X, B, Z, p: Params, h: Hyper, layer, q):
    """Damped Newton on g; the best of STEPS (including 0) is kept."""
    shape = p.g.shape

    def f(gv):
        return objective(X, B, Z, p._replace(g=gv.reshape(shape)), h, layer, q)

    gv = p.g.ravel()
    for _ in range(N_NEWTON):
        grad = jax.grad(f)(gv)
        H = jax.hessian(f)(gv)
        negH = -(H + H.T) / 2
        shift = jnp.maximum(0.0, -jnp.linalg.eigvalsh(negH).min()) + 1e-8
        step = jnp.linalg.solve(negH + shift * jnp.eye(len(gv)), grad)
        cand = gv[None] + STEPS[:, None] * step[None]
        vals = jax.vmap(f)(cand)
        vals = jnp.where(jnp.isfinite(vals), vals, -jnp.inf)
        gv = cand[jnp.argmax(vals)]
    return p._replace(g=gv.reshape(shape))


def _sweep(X, B, Z, p, h, layer, q, var_floor):
    _, post = _posterior(X, B, Z, p, layer, q)
    obj = objective(X, B, Z, p, h, layer, q)
    p = _m_beta_w_psi(p, h, post, q, var_floor)
    if layer > 0:
        p = _m_g(X, B, Z, p, h, layer, q)
    return p, obj


@partial(jax.jit, static_argnames=("layer", "q", "n_iter"))
def fit(X, B, Z, p0: Params, h: Hyper, layer: int, q: int, n_iter: int = 1000):
    """n_iter ECM sweeps; returns the final parameters and the objective trace."""
    X, B, Z = jnp.asarray(X), jnp.asarray(B), jnp.asarray(Z)
    var_floor = jnp.var(X, axis=0)
    return jax.lax.scan(lambda p, _: _sweep(X, B, Z, p, h, layer, q, var_floor),
                        p0, None, length=n_iter)


# --------------------------------------------------------------- helpers ----

def init_params(X, B, Z, q, layer, rng, h: Hyper, g_sd: float = 0.0) -> Params:
    """The step 2 initialisation (model/mfa.py, K = 1), with g = 0 or random."""
    h1 = mfa.Hyper(P=h.P, rank=h.rank, a0=h.a0, b0=h.b0, psi_floor=h.psi_floor)
    p1 = mfa.init_params(np.asarray(X), np.asarray(B), 1, q, rng, h1)
    L = n_curves(layer, q)
    g = g_sd * rng.standard_normal((L, Z.shape[1])) if layer > 0 else np.zeros((L, Z.shape[1]))
    return Params(beta=jnp.asarray(p1.beta[0]), W=jnp.asarray(p1.W[0]),
                  psi=jnp.asarray(p1.psi[0]), lam=jnp.asarray(p1.lam[0]),
                  g=jnp.asarray(g))


def covariance(p: Params, z_row, layer, q):
    """C at one scale-design row (e.g. scale_design([t]))."""
    ls0, ls = log_scales(p.g, jnp.atleast_2d(jnp.asarray(z_row)), layer, q)
    s0, s = jnp.exp(ls0[0]), jnp.exp(ls[0])
    return (p.W * s ** 2) @ p.W.T + s0 ** 2 * jnp.diag(p.psi)


def n_params(D, q, layer, J=2, M=20):
    base = M * D + D * q - q * (q - 1) // 2 + D
    return base + (0 if layer == 0 else n_curves(layer, q) * J)


def simulate(rng, B, Z, p: Params, layer, q):
    B, Z = np.asarray(B), np.asarray(Z)
    ls0, ls = (np.asarray(a) for a in log_scales(p.g, jnp.asarray(Z), layer, q))
    N, D = B.shape[0], p.beta.shape[1]
    theta = rng.standard_normal((N, q)) * np.exp(ls)
    eps = rng.standard_normal((N, D)) * np.sqrt(np.asarray(p.psi)) * np.exp(ls0)[:, None]
    return B @ np.asarray(p.beta) + theta @ np.asarray(p.W).T + eps
