"""Fit one specification on all training infants and save the fitted model.

    python scripts/fit_final.py --K 1 --q 15

The sweep (run_spec.py) keeps only held-out scores. This keeps the parameters,
so the axes can be inspected: loadings W, unique variances psi, age
trajectories beta, and each infant's posterior axis scores. The 101
validation infants are not touched.

Writes results/final/K{K}_q{q}.npz and a provenance record.
"""

import argparse
import json
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


def fit_best(X, B, K, q, M=splines.M_DEFAULT, n_init=20, n_iter=1000, seed_base=SEED_BASE,
             extra_inits=()):
    """Fit n_init random starts (plus any given `extra_inits`) in one vectorised
    pass and keep the best. Returns (params, trace, converged, restart spread)."""
    h = mfa.Hyper(P=jnp.asarray(splines.difference_penalty(M)), rank=splines.penalty_rank(M))
    inits = list(extra_inits) + [mfa.init_params(X, B, K, q, np.random.default_rng(seed_base + s), h)
                                 for s in range(n_init)]
    p_all, traces = mfa.fit_restarts(X, B, mfa.stack_params(inits), h, n_iter=n_iter)
    traces = np.asarray(traces)
    obj = traces[:, -1]
    finite = np.isfinite(obj)
    if not finite.any():
        raise RuntimeError("no initialisation produced a finite fit")
    b = int(np.where(finite, obj, -np.inf).argmax())
    tr = traces[b]
    converged = bool(abs(tr[-1] - tr[-2]) / max(abs(tr[-1]), 1.0) < 1e-6)
    spread = float(obj[finite].max() - obj[finite].min())
    return mfa.unstack_params(p_all, b), tr, converged, spread


def principal(W, psi):
    """Principal axes of a K = 1 loading matrix: loadings U S (largest first),
    each signed so its loadings sum positive, their variances S^2, the rotation
    Vt (scores rotate as theta @ Vt.T), and the signs applied."""
    U, S, Vt = np.linalg.svd(np.asarray(W), full_matrices=False)
    L = U * S
    sign = np.where(L.sum(0) < 0, -1.0, 1.0)
    return L * sign, S ** 2, Vt, sign


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
    ap.add_argument("--K", type=int)
    ap.add_argument("--q", type=int)
    ap.add_argument("--from-selection", choices=("selected", "runner_up"),
                    help="take K and q from results/selection.json (scripts/compare_models.py)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do nothing if this model is already fitted (same seeds give the same fit)")
    ap.add_argument("--M", type=int, default=splines.M_DEFAULT)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--n-iter", type=int, default=1000)
    a = ap.parse_args()
    if a.from_selection:
        pick = json.loads((paths.RESULTS / "selection.json").read_text())[a.from_selection]
        if pick is None:
            raise SystemExit(f"selection.json has no {a.from_selection}")
        a.K, a.q = pick["K"], pick["q"]
    if a.K is None or a.q is None:
        ap.error("give --K and --q, or --from-selection")
    name = f"K{a.K}_q{a.q}"
    if a.skip_existing and (OUT / f"{name}.npz").exists():
        print(f"{name} already fitted -> {OUT / name}.npz"); return

    coh, Xraw = load()
    X, mu, sd = standardise(Xraw)
    t = age_for(coh, CLOCK)
    B = splines.basis(t, M=a.M)

    t0 = time.time()
    p, tr, converged, spread = fit_best(X, B, a.K, a.q, M=a.M, n_init=a.n_init,
                                        n_iter=a.n_iter)
    scores = axis_scores(X, B, p)
    resp = np.asarray(mfa.responsibilities(X, B, p))

    OUT.mkdir(parents=True, exist_ok=True)
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
                     restart_spread=spread, seconds=secs),
        thresholds=dict(convergence=prov.threshold(1e-6, "relative change in penalised log lik at final sweep")),
        inputs=[paths.RESULTS / "features" / f"{VARIANT}.npz",
                paths.RESULTS / "cohort" / "cohort.csv"])


if __name__ == "__main__":
    main()
