"""Simulation checks for model/pls.py.

    python scripts/check_pls.py

1. Planted signal: one latent variable links lobes 0-2 with Y columns 0-3
   (n = 400, 12 x 27, with twin families; effect 0.6 gives a correlation of
   about 0.2 for each planted pair). LV1 must be significant and its
   stable saliences must be the planted ones.
2. Null calibration: 200 null datasets (with twin families), 500
   permutations each. The share of LV1 p < 0.05 must be near 0.05.
3. With one real LV, the share of LV2 p < 0.05 must stay near or below 0.05
   (tests the deflated permutation test).
4. A Y block with fewer columns than X (12 x 10, like the behaviour PLS):
   bootstrap ratios must be finite and of normal size.
"""

import numpy as np

from model import pls as P


def data(rng, n=400, p=12, k=27, effect=0.0, twin_share=0.15):
    n_tw = int(n * twin_share / 2)
    fam = np.concatenate([np.repeat(np.arange(n_tw), 2), np.arange(n_tw, n_tw + n - 2 * n_tw)])
    shared = rng.normal(size=fam.max() + 1)[fam]          # twins are correlated
    X = rng.normal(size=(n, p)) + 0.5 * shared[:, None]
    Y = rng.normal(size=(n, k)) + 0.5 * rng.normal(size=fam.max() + 1)[fam][:, None]  # own family effect
    if effect:
        z = rng.normal(size=n)
        X[:, :3] += effect * z[:, None]
        Y[:, :4] += effect * z[:, None]
    return X, Y, fam


def main():
    rng = np.random.default_rng(42)
    X, Y, fam = data(rng, effect=0.6)
    r = P.pls(X, Y, fam, n_perm=1000, n_boot=1000)
    top_x = set(np.argsort(-np.abs(r["bsr_x"][:, 0]))[:3])
    top_y = set(np.argsort(-np.abs(r["bsr_y"][:, 0]))[:4])
    print("planted: p_perm LV1-3", r["p_perm"][:3].round(4), "| top lobes", sorted(top_x),
          "| top Y", sorted(top_y), "| BSR lobes 0-2", r["bsr_x"][:3, 0].round(1))
    assert r["p_perm"][0] < 0.05 and top_x == {0, 1, 2} and top_y == {0, 1, 2, 3}

    ps = []
    for _ in range(200):
        X, Y, fam = data(rng)
        ps.append(P.pls(X, Y, fam, n_perm=500, n_boot=2)["p_perm"][0])
    ps = np.array(ps)
    print(f"null: share LV1 p < 0.05 = {(ps < 0.05).mean():.3f} (expect 0.05; 95% band 0.02-0.08)")
    assert 0.015 <= (ps < 0.05).mean() <= 0.09

    # One real LV: LV2 must still be calibrated (the undeflated test fails this).
    ps2 = []
    for _ in range(200):
        X, Y, fam = data(rng, effect=0.6)
        ps2.append(P.pls(X, Y, fam, n_perm=500, n_boot=2)["p_perm"][1])
    ps2 = np.array(ps2)
    print(f"one real LV: share LV2 p < 0.05 = {(ps2 < 0.05).mean():.3f} (expect <= 0.05)")
    assert (ps2 < 0.05).mean() <= 0.09

    X, Y, fam = data(rng, k=10, effect=0.6)
    r = P.pls(X, Y, fam, n_perm=200, n_boot=500)
    big = max(np.abs(r["bsr_x"]).max(), np.abs(r["bsr_y"]).max())
    print(f"12 x 10 block: largest |BSR| = {big:.1f}, LV1 BSR Y 0-3 {r['bsr_y'][:4, 0].round(1)}")
    assert np.isfinite(big) and big < 50

    rng = np.random.default_rng(7)
    X, Y, fam = data(rng, effect=0.6)
    tr = rng.random(len(X)) < 0.85
    rep = P.replicate(X[tr], Y[tr], X[~tr], Y[~tr], fam[~tr], n_perm=1000)
    print("replication planted: r LV1", round(rep["r"][0], 3), "p", rep["p"][0])
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
