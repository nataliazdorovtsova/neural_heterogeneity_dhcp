"""Permutation null for the axes (parallel analysis).

    python scripts/permutation_null.py --chunk 0 --n-perm 10     # array task

Asks how large the axes would be if there were no shared structure at all.
Starting from the standardised training data, the age trajectory of the fitted
one-group model is removed, then each of the 270 coordinates' residuals is
shuffled independently across infants. That destroys every correlation between
coordinates (which is all the axes describe) while keeping each coordinate's
distribution and its age trend. The one-group model is refitted to each
shuffled dataset, and the sizes of its principal axes form the null.

A real axis is "bigger than noise" if its variance exceeds the 95th percentile
of the null variances at the same rank (Horn's parallel analysis).

Writes results/significance/permutation/chunk{c}.npz.
"""

import argparse
import time

import numpy as np

import paths
from features import standardise
from model import splines
from scripts.run_spec import load, age_for, CLOCK
from scripts.fit_final import fit_best, principal

OUT = paths.RESULTS / "significance" / "permutation"
Q = 15
SEED = 47_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--n-init", type=int, default=5)
    a = ap.parse_args()

    coh, Xraw = load()
    X, *_ = standardise(Xraw)
    B = splines.basis(age_for(coh, CLOCK))
    full = np.load(paths.RESULTS / "final" / f"K1_q{Q}.npz")
    mean = B @ full["beta"][0]                       # the fitted age trajectory
    resid = X - mean

    var, conv, secs = [], [], []
    for i in range(a.n_perm):
        t0 = time.time()
        perm = a.chunk * a.n_perm + i
        rng = np.random.default_rng(SEED + perm)
        X0 = mean + rng.permuted(resid, axis=0)      # each column shuffled on its own
        p, tr, c, _ = fit_best(X0, B, 1, Q, n_init=a.n_init, seed_base=SEED * 10 + 100 * perm)
        _, S2, _, _ = principal(p.W[0], p.psi[0])
        var.append(S2); conv.append(c); secs.append(time.time() - t0)
        print(f"perm {perm}: largest null axis variance {S2[0]:.2f}, converged={c}, "
              f"{secs[-1]:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"chunk{a.chunk}.npz", var=np.array(var), converged=np.array(conv),
             seconds=np.array(secs), first_perm=a.chunk * a.n_perm)


if __name__ == "__main__":
    main()
