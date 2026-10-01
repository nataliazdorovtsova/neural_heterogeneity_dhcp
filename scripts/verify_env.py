"""Smoke test: prove the stack does what this project actually needs.

    python scripts/verify_env.py

Checks float64 is really on, that a discrete mixture can be enumerated out
under NUTS, that the low-rank LambdaLambda' + Psi covariance works, and that
the three graph measures compute on a real connectome.
"""

import os
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import paths

print(f"jax {jax.__version__} | numpyro {numpyro.__version__} | devices {jax.devices()}")

# --- float64 actually on? ----------------------------------------------------
assert os.environ.get("JAX_ENABLE_X64") == "1", "JAX_ENABLE_X64 unset — source activate.sh"
assert jnp.zeros(1).dtype == jnp.float64, f"not float64: {jnp.zeros(1).dtype}"
print("float64: on")

# --- real data loads and lines up -------------------------------------------
A = np.load(paths.STACK, mmap_mode="r")
part = pd.read_csv(paths.PARTICIPANTS)
assert A.shape == (90, 90, 753)
assert len(part) == A.shape[2]
print(f"connectomes: {A.shape} | participants: {part.shape}")

# --- the three regional measures, on one real infant ------------------------
import networkx as nx
from scipy.linalg import expm

W = np.asarray(A[:, :, 0], dtype=float)
strength = W.sum(1)
clustering = np.array(list(nx.clustering(nx.from_numpy_array(W), weight="weight").values()))
# communicability on the degree-normalised weighted matrix (Crofts & Higham)
d = np.sqrt(strength)
comm = expm(W / np.outer(d, d)).sum(1)
feats = np.column_stack([strength, clustering, comm])
assert feats.shape == (90, 3) and np.isfinite(feats).all()
print(f"features per infant: {feats.shape} -> flattened dim {feats.size}")

# --- the modelling pattern: marginalised mixture, low-rank covariance -------
# The cluster label is summed out explicitly with logsumexp rather than via
# numpyro's enumeration: funsor (which enumeration requires) is incompatible
# with this jax, and explicit marginalisation is faster anyway. Per-infant
# responsibilities come back post hoc from the same per-component densities.
from jax.scipy.special import logsumexp

def component_logp(y, pi, mu, lam, psi):
    """(K, N) log pi_k + log N(y_i | mu_k, lam lam' + psi)."""
    comp = dist.LowRankMultivariateNormal(mu[:, None, :], lam, psi)
    return comp.log_prob(y) + jnp.log(pi)[:, None]

def model(y, K=2, q=2):
    N, D = y.shape
    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(K)))
    mu = numpyro.sample("mu", dist.Normal(0, 5).expand([K, D]).to_event(2))
    lam = numpyro.sample("lam", dist.Normal(0, 1).expand([D, q]).to_event(2))
    psi = numpyro.sample("psi", dist.HalfNormal(1).expand([D]).to_event(1))
    lp = component_logp(y, pi, mu, lam, psi)
    numpyro.factor("lik", logsumexp(lp, axis=0).sum())

rng = np.random.default_rng(0)
y = jnp.asarray(np.vstack([rng.normal(-2, 1, (60, 5)), rng.normal(2, 1, (60, 5))]))

mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=300, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0), y)
post = mcmc.get_samples()
means = np.asarray(post["mu"]).mean(0)
sep = float(np.abs(means[0] - means[1]).mean())
assert sep > 2.0, f"mixture failed to separate two obvious clusters (sep={sep:.2f})"
print(f"marginalised mixture + low-rank MVN: ok (recovered separation {sep:.2f})")

# per-infant responsibilities, at the posterior mean
lp = component_logp(y, post["pi"].mean(0), post["mu"].mean(0),
                    post["lam"].mean(0), post["psi"].mean(0))
resp = np.asarray(jnp.exp(lp - logsumexp(lp, axis=0)))
acc = max((resp[0, :60] > 0.5).mean(), (resp[1, :60] > 0.5).mean())
assert acc > 0.9, f"responsibilities did not recover known labels ({acc:.2f})"
print(f"per-infant responsibilities: ok ({acc:.0%} of known group-1 assigned correctly)")

# --- model comparison machinery ---------------------------------------------
import arviz as az
idata = az.from_numpyro(mcmc)
print(f"arviz: ok ({len(idata.posterior.data_vars)} posterior vars, loo available)")

import nilearn, sklearn, matplotlib
print(f"nilearn {nilearn.__version__} | sklearn {sklearn.__version__} | mpl {matplotlib.__version__}")
print("\nall checks passed")
