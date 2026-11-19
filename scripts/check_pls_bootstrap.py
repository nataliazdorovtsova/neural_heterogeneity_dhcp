"""Is the PLS bootstrap standard error right? (model/pls.py, align_method)

    python scripts/check_pls_bootstrap.py

Generator as scripts/check_pls.py (one planted LV, twin families), with the
two real shapes: 12 x 12 (one score per condition) and 12 x 10 (behaviour).
True sampling SD of each LV1 salience: 400 fresh datasets, each LV1 sign-
matched to the mean direction. Bootstrap SD: mean over 30 datasets of the
family-bootstrap SD (500 resamples) for each alignment method. A good method
has a ratio bootstrap SD / true SD near 1 (bootstrap ratios then read as z).
Also: effect 0 (no real LV), where LV1 saliences are noise, so |BSR| >= 2.58
should be rare for a well-calibrated method.
"""

import numpy as np

from model import pls as P
from scripts.check_pls import data

METHODS = ("sign", "stacked", "v")


def lv1(X, Y):
    U, s, V = P.svd_pls(P.zscore(X), P.zscore(Y))
    return U[:, 0], V[:, 0]


def true_sd(rng, k, effect, n_sim=400):
    us, vs = [], []
    for _ in range(n_sim):
        X, Y, _ = data(rng, k=k, effect=effect)
        u, v = lv1(X, Y)
        us.append(u), vs.append(v)
    us, vs = np.array(us), np.array(vs)
    ref = np.concatenate([us[0], vs[0]])
    for _ in range(3):  # sign-match to the running mean direction
        sg = np.sign(np.concatenate([us, vs], 1) @ ref)
        us, vs = us * sg[:, None], vs * sg[:, None]
        ref = np.concatenate([us, vs], 1).mean(0)
    return us.std(0, ddof=1), vs.std(0, ddof=1)


def main():
    rng = np.random.default_rng(3)
    for k in (12, 10):
        for effect in (0.6, 0.0):
            print(f"\n--- 12 x {k}, effect {effect} ---")
            if effect:
                tu, tv = true_sd(rng, k, effect)
            for m in METHODS:
                su, sv, frac = [], [], []
                for _ in range(30):
                    X, Y, fam = data(rng, k=k, effect=effect)
                    r = P.pls(X, Y, fam, n_perm=1, n_boot=500, align_method=m)
                    su.append(r["U"][:, 0] / r["bsr_x"][:, 0])
                    sv.append(r["V"][:, 0] / r["bsr_y"][:, 0])
                    frac.append(np.mean(np.abs(np.concatenate([r["bsr_x"][:, 0], r["bsr_y"][:, 0]])) >= 2.58))
                su, sv = np.mean(su, 0), np.mean(sv, 0)
                msg = f"{m:8s} share |BSR| >= 2.58 on LV1: {np.mean(frac):.3f}"
                if effect:
                    msg += (f" | boot/true SD, lobes: median {np.median(su / tu):.2f}"
                            f" | Y: median {np.median(sv / tv):.2f}")
                print(msg, flush=True)


if __name__ == "__main__":
    main()
