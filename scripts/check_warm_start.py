"""Gate: do warm-start-only bootstrap refits reproduce the original replicates?

    python scripts/check_warm_start.py

The first 200 bootstrap replicates (job 4199434) were refitted from the fitted
parameters plus two random starts. Every one-group fit so far has reached a
single optimum from all starts, so the random starts should add cost but not
change the answer. This refits the first five original replicates (identical
simulated data: same seeds) from the warm start alone and compares the aligned
loadings and axis variances with the stored ones.

Exits non-zero if they differ, so a run chained on it (afterok) never starts.
"""

import sys

import numpy as np

import paths
from model import mfa, splines
from scripts.run_spec import load, age_for, CLOCK
from scripts.fit_final import fit_best, principal
from scripts.bootstrap import selected_model, procrustes, SEED

N_CHECK = 5
TOL_LOADING = 1e-3          # loadings are O(0.1-0.8); convergence tolerance is 1e-6 relative
TOL_VAR = 1e-3              # relative, on axis variances


def main():
    orig = np.load(paths.RESULTS / "significance" / "bootstrap" / "chunk0.npz")
    q, p_hat = selected_model(int(orig["q"]))
    coh, _ = load()
    B = splines.basis(age_for(coh, CLOCK))
    L_hat, _, _, _ = principal(p_hat.W[0], p_hat.psi[0])
    p_hat = mfa.Params(*(np.asarray(x) for x in p_hat))

    worst_L, worst_V = 0.0, 0.0
    for b in range(N_CHECK):
        Xb, _ = mfa.simulate(np.random.default_rng(SEED + b), B, p_hat)
        p, _, c, _ = fit_best(Xb, B, 1, q, n_init=0, seed_base=SEED * 10 + 10 * b,
                              extra_inits=[p_hat])
        W = np.asarray(p.W[0])
        aligned, s2 = W @ procrustes(W, L_hat), principal(W, p.psi[0])[1]
        dL = float(np.abs(aligned - orig["loadings"][b]).max())
        dV = float(np.max(np.abs(s2 - orig["S2"][b]) / orig["S2"][b]))
        worst_L, worst_V = max(worst_L, dL), max(worst_V, dV)
        print(f"replicate {b}: converged={c}, max |dloading| {dL:.2e}, max rel |dvariance| {dV:.2e}")

    ok = worst_L < TOL_LOADING and worst_V < TOL_VAR
    print(f"{'PASS' if ok else 'FAIL'}: worst loading difference {worst_L:.2e} (tol {TOL_LOADING}), "
          f"worst relative variance difference {worst_V:.2e} (tol {TOL_VAR})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
