"""Fit one specification on all training infants and save the fitted model.

    python scripts/fit_final.py --K 1 --q 15

The sweep (run_spec.py) keeps only held-out scores. This keeps the parameters,
so the axes can be inspected: loadings W, unique variances psi, age
trajectories beta, and each infant's posterior axis scores. The 101
validation infants are not touched.

Writes results/final/K{K}_q{q}.npz and a provenance record.
"""

import argparse
import time

import numpy as np
import jax.numpy as jnp

import paths
import provenance as prov
from features import standardise
from model import mfa, splines
from scripts.run_spec import load, VARIANT, CLOCK, age_for

OUT = paths.RESULTS / "final"
SEED_BASE = 900_000          # distinct from the CV seeds (10000*fold + restart)


def axis_scores(X, B, p):
    """Posterior mean axis scores E[theta | x, z=k], shape (N, K, q)."""
    X, B = np.asarray(X), np.asarray(B)
    W, psi, beta = np.asarray(p.W), np.asarray(p.psi), np.asarray(p.beta)
    K, D, q = W.shape
    out = np.zeros((len(X), K, q))
    for k in range(K):
        Wp = W[k] / psi[k][:, None]                          # Psi^-1 W, (D, q)
        Minv = np.linalg.inv(np.eye(q) + W[k].T @ Wp)
        out[:, k] = (X - B @ beta[k]) @ Wp @ Minv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--M", type=int, default=splines.M_DEFAULT)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=1000)
    a = ap.parse_args()

    coh, Xraw = load()
    X, mu, sd = standardise(Xraw)
    t = age_for(coh, CLOCK)
    B = splines.basis(t, M=a.M)
    h = mfa.Hyper(P=jnp.asarray(splines.difference_penalty(a.M)),
                  rank=splines.penalty_rank(a.M))

    t0 = time.time()
    inits = [mfa.init_params(X, B, a.K, a.q, np.random.default_rng(SEED_BASE + s), h)
             for s in range(a.n_init)]
    p_all, traces = mfa.fit_restarts(X, B, mfa.stack_params(inits), h, n_iter=a.n_iter)
    traces = np.asarray(traces)
    obj = traces[:, -1]
    finite = np.isfinite(obj)
    if not finite.any():
        raise SystemExit("no initialisation produced a finite fit")
    b = int(np.where(finite, obj, -np.inf).argmax())
    p = mfa.unstack_params(p_all, b)
    tr = traces[b]
    converged = bool(abs(tr[-1] - tr[-2]) / max(abs(tr[-1]), 1.0) < 1e-6)
    # How much do the restarts agree? A wide spread means a rough likelihood.
    spread = float(np.sort(obj[finite])[-1] - np.sort(obj[finite])[0])

    scores = axis_scores(X, B, p)
    resp = np.asarray(mfa.responsibilities(X, B, p))

    OUT.mkdir(parents=True, exist_ok=True)
    name = f"K{a.K}_q{a.q}"
    np.savez(OUT / f"{name}.npz",
             log_pi=np.asarray(p.log_pi), beta=np.asarray(p.beta), W=np.asarray(p.W),
             psi=np.asarray(p.psi), lam=np.asarray(p.lam),
             mu=mu, sd=sd, scores=scores, resp=resp, trace=tr,
             dhcp_id=coh.dhcp_id.values.astype(str), age=t)
    secs = time.time() - t0
    print(f"{name}: penalised loglik {tr[-1]:.1f}, converged={converged}, "
          f"restart spread {spread:.1f}, {secs:.0f}s -> {OUT / name}.npz")

    prov.snapshot(
        name=f"final_{name}",
        config=dict(K=a.K, q=a.q, M=a.M, clock=CLOCK, variant=VARIANT,
                    covariance_sharing="group_specific", trajectory="group_smooth",
                    n_init=a.n_init, n_iter=a.n_iter, seed_rule=f"{SEED_BASE} + restart",
                    fitted_on="all training infants (validation untouched)"),
        results=dict(n=len(X), penalised_loglik=float(tr[-1]), converged=converged,
                     restart_spread=spread, best_restart=b,
                     successful_restarts=int(finite.sum()), seconds=secs),
        thresholds=dict(convergence=prov.threshold(1e-6, "relative change in penalised log lik at final sweep")),
        inputs=[paths.RESULTS / "features" / f"{VARIANT}.npz",
                paths.RESULTS / "cohort" / "cohort.csv"])


if __name__ == "__main__":
    main()
