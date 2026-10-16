"""Split-half replication of the within-population axes.

    python scripts/split_half.py --split 0          # one random split (array task)

Splits the training infants into two halves, stratified by age quartile and
preterm status, fits the one-group model to each half separately, and asks
whether the same axes come out. Each full-data axis is matched to one axis in
each half (Hungarian assignment on |cosine|, so no half-axis is used twice),
and the two halves' matched axes are compared with each other.

Standardisation uses the full training set's moments, so the halves are on
one scale and their loadings are directly comparable. No held-out scoring is
involved, so this leaks nothing.

Writes results/significance/split_half/split{s}.npz.
"""

import argparse
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import paths
from features import standardise
from model import splines
from scripts.run_spec import load, age_for, CLOCK
from scripts.fit_final import fit_best, principal

OUT = paths.RESULTS / "significance" / "split_half"
Q = 15
N_MATCH = 8                         # full-data axes followed into the halves
SEED = 31_000


def halves(coh, rng):
    band = np.digitize(coh.scan_age.values, np.quantile(coh.scan_age.values, [0.25, 0.5, 0.75]))
    stratum = band * 2 + coh.preterm.values.astype(int)
    a = np.zeros(len(coh), bool)
    for s in np.unique(stratum):
        idx = rng.permutation(np.where(stratum == s)[0])
        a[idx[: len(idx) // 2 + rng.integers(0, 2) * (len(idx) % 2)]] = True
    return a


def match(L_ref, L_other, n):
    """For the first n reference axes: the matched axis index in L_other and |cos|."""
    A = L_ref[:, :n] / np.linalg.norm(L_ref[:, :n], axis=0)
    Bm = L_other / np.linalg.norm(L_other, axis=0)
    C = np.abs(A.T @ Bm)
    rows, cols = linear_sum_assignment(-C)
    return cols[np.argsort(rows)], C[rows, cols][np.argsort(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--n-init", type=int, default=5)
    a = ap.parse_args()

    t0 = time.time()
    coh, Xraw = load()
    X, *_ = standardise(Xraw)
    B = splines.basis(age_for(coh, CLOCK))
    full = np.load(paths.RESULTS / "final" / f"K1_q{Q}.npz")
    L_full, *_ = principal(full["W"][0], full["psi"][0])

    rng = np.random.default_rng(SEED + a.split)
    in_a = halves(coh, rng)
    L, conv = {}, {}
    for name, m in (("A", in_a), ("B", ~in_a)):
        p, tr, conv[name], _ = fit_best(X[m], B[m], 1, Q, n_init=a.n_init,
                                        seed_base=SEED + 100 * a.split + (name == "B") * 50)
        L[name], *_ = principal(p.W[0], p.psi[0])

    ia, cos_full_a = match(L_full, L["A"], N_MATCH)
    ib, cos_full_b = match(L_full, L["B"], N_MATCH)
    LA, LB = L["A"][:, ia], L["B"][:, ib]
    cos_ab = np.abs((LA / np.linalg.norm(LA, axis=0) * (LB / np.linalg.norm(LB, axis=0))).sum(0))

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"split{a.split}.npz", LA=LA, LB=LB, cos_ab=cos_ab, cos_full_a=cos_full_a,
             cos_full_b=cos_full_b, n_a=int(in_a.sum()), n_b=int((~in_a).sum()),
             converged=np.array([conv["A"], conv["B"]]))
    print(f"split {a.split}: halves {in_a.sum()}/{(~in_a).sum()}, converged {conv}, "
          f"{time.time() - t0:.0f}s")
    print("   |cos| half A vs half B, axes 1-8:", np.round(cos_ab, 2))


if __name__ == "__main__":
    main()
