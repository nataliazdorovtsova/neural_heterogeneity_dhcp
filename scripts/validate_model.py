"""Stage 4 gate: the model must be correct before it touches real infants.

    python scripts/validate_model.py

Three checks:
  1. ECM increases the penalised log likelihood monotonically.
  2. Parameters are recovered from data simulated at known K and q.
  3. Held-out predictive density prefers the generating K over the alternatives.
"""

import numpy as np
import jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score

from model import mfa, splines

RNG = np.random.default_rng(0)
N, D, M = 500, 40, 12


def make_truth(K, q, rng):
    h = mfa.Hyper(P=jnp.asarray(splines.difference_penalty(M)),
                  rank=splines.penalty_rank(M))
    t = rng.uniform(splines.LO, splines.HI, N)
    B = splines.basis(np.sort(t), M=M)
    beta = np.zeros((K, M, D))
    for k in range(K):
        trend = np.linspace(-1, 1, M)[:, None] * rng.standard_normal(D)
        beta[k] = trend + 2.0 * k * rng.standard_normal(D)       # separated groups
    W = rng.standard_normal((K, D, q)) * 0.6 if q else np.zeros((K, D, 0))
    psi = np.full((K, D), 0.25)
    pi = np.full(K, 1.0 / K)
    truth = mfa.Params(log_pi=jnp.log(jnp.asarray(pi)), beta=jnp.asarray(beta),
                       W=jnp.asarray(W), psi=jnp.asarray(psi),
                       lam=jnp.ones((K, D)))
    return B, h, truth


def check_monotonic():
    print("1. ECM monotonicity")
    B, h, truth = make_truth(K=2, q=3, rng=np.random.default_rng(1))
    X, _ = mfa.simulate(np.random.default_rng(2), B, truth)
    p0 = mfa.init_params(X, B, K=2, q=3, rng=np.random.default_rng(3), h=h)
    _, trace = mfa.fit(X, B, p0, h, n_iter=120)
    tr = np.asarray(trace)
    drops = np.diff(tr)
    worst = drops.min()
    tol = 1e-6 * max(abs(tr[-1]), 1.0)
    print(f"   objective {tr[0]:.1f} -> {tr[-1]:.1f} over {len(tr)} sweeps")
    print(f"   largest single-step decrease: {worst:+.3e}  (tolerance {tol:.1e})")
    assert worst > -tol, "ECM decreased the penalised log likelihood"
    print("   PASS\n")
    return tr


def check_recovery():
    print("2. parameter recovery at known K=3, q=3")
    rng = np.random.default_rng(10)
    B, h, truth = make_truth(K=3, q=3, rng=rng)
    X, z = mfa.simulate(np.random.default_rng(11), B, truth)

    best, best_obj = None, -np.inf
    for s in range(8):                                   # multiple starts
        p0 = mfa.init_params(X, B, K=3, q=3, rng=np.random.default_rng(100 + s), h=h)
        p, trace = mfa.fit(X, B, p0, h, n_iter=200)
        if float(trace[-1]) > best_obj:
            best_obj, best = float(trace[-1]), p

    r = np.asarray(mfa.responsibilities(X, B, best))
    ari = adjusted_rand_score(z, r.argmax(1))
    print(f"   adjusted Rand index vs true labels: {ari:.3f}")

    # centres, matched to truth by label permutation
    from itertools import permutations
    mu_true = np.einsum("nm,kmd->knd", np.asarray(B), np.asarray(truth.beta))
    mu_fit = np.einsum("nm,kmd->knd", np.asarray(B), np.asarray(best.beta))
    err = min(np.abs(mu_true - mu_fit[list(perm)]).mean()
              for perm in permutations(range(3)))
    scale = np.abs(mu_true).mean()
    print(f"   mean |error| in group centres: {err:.4f}  ({100*err/scale:.1f}% of signal)")

    # covariance, which is what the likelihood identifies (W itself is rotation-free)
    C_true = np.einsum("kdq,keq->kde", np.asarray(truth.W), np.asarray(truth.W)) \
             + np.asarray(truth.psi)[:, :, None] * np.eye(D)[None]
    C_fit = np.einsum("kdq,keq->kde", np.asarray(best.W), np.asarray(best.W)) \
            + np.asarray(best.psi)[:, :, None] * np.eye(D)[None]
    cerr = min(np.abs(C_true - C_fit[list(perm)]).mean()
               for perm in permutations(range(3)))
    print(f"   mean |error| in within-group covariance C: {cerr:.4f} "
          f"({100*cerr/np.abs(C_true).mean():.1f}% of scale)")
    assert ari > 0.9, f"failed to recover the generating partition (ARI {ari:.3f})"
    assert err / scale < 0.10, "group centres not recovered"
    print("   PASS\n")


def check_selection():
    print("3. held-out predictive density prefers the generating K")
    rng = np.random.default_rng(20)
    B, h, truth = make_truth(K=2, q=2, rng=rng)
    X, _ = mfa.simulate(np.random.default_rng(21), B, truth)
    n_tr = int(0.8 * N)
    Xtr, Btr, Xte, Bte = X[:n_tr], B[:n_tr], X[n_tr:], B[n_tr:]

    scores = {}
    for K in (1, 2, 3):
        best, best_obj = None, -np.inf
        for s in range(6):
            p0 = mfa.init_params(Xtr, Btr, K=K, q=2, rng=np.random.default_rng(300 + s), h=h)
            p, trace = mfa.fit(Xtr, Btr, p0, h, n_iter=200)
            if float(trace[-1]) > best_obj:
                best_obj, best = float(trace[-1]), p
        scores[K] = float(mfa.log_predictive_density(Xte, Bte, best))
        print(f"   K={K}: held-out log predictive density {scores[K]:11.1f}")
    chosen = max(scores, key=scores.get)
    print(f"   selected K = {chosen} (generated with K = 2)")
    assert chosen == 2, "selection did not recover the generating K"
    print("   PASS\n")


if __name__ == "__main__":
    check_monotonic()
    check_recovery()
    check_selection()
    print("all Stage 4 validation checks passed")
