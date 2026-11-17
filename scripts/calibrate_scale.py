"""Calibration of the layer-1 scale curve on simulated data (RESEARCH_PLAN_BIN10 4b).

    NH_RUN=bin10_no7 python scripts/calibrate_scale.py --chunk 0 --n 10

Truths are built from the constant-spread fit (layer 0, q = 6) to the real
training data, at the real ages (n = 562):
    null   no age scaling (s = 1 at every age)
    known  log s(t) = -0.15 u + 0.03 u^2, u = (t - 40) / 5 (s(30) = 1.52,
           close to the step 3 binned SDs)
Each replicate simulates both and fits layer 1 (q = 6). The fitted s(t) at
28, 30, 35 and 44 weeks shows any bias at the young end (few infants), and the
null spread shows how large a curve noise alone can produce.
Writes scale/calib/chunk<c>.npz.
"""

import argparse

import numpy as np
import jax.numpy as jnp

import paths
from features import standardise
from model import mfa_scale as ms, splines
from scripts.run_spec import load
from scripts.summarise_scale import hyper, OUT

Q = 6
T_EVAL = np.array([28.0, 30.0, 35.0, 44.0])
G_KNOWN = np.array([[-0.15, 0.03]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    coh, X = load()
    X, *_ = standardise(X)
    t = coh.scan_age.values
    B, Z = splines.basis(t), ms.scale_design(t)
    h = hyper()
    p0 = ms.init_params(X, B, Z, Q, 0, np.random.default_rng(0), h)
    base, _ = ms.fit(X, B, Z, p0, h, layer=0, q=Q, n_iter=1000)
    Ze = jnp.asarray(ms.scale_design(T_EVAL))
    out = {"null": [], "known": []}
    for r in range(a.n):
        for name, g in (("null", np.zeros((1, 2))), ("known", G_KNOWN)):
            truth = base._replace(g=jnp.asarray(g))
            Xs = ms.simulate(np.random.default_rng(100_000 * a.chunk + 2 * r + (name == "known")),
                             B, Z, truth, 1, Q)
            Xs, *_ = standardise(Xs)
            pf, _ = ms.fit(Xs, B, Z, ms.init_params(Xs, B, Z, Q, 1, np.random.default_rng(r), h),
                           h, layer=1, q=Q, n_iter=1000)
            ls0, _ = ms.log_scales(pf.g, Ze, 1, Q)
            out[name].append(np.exp(np.asarray(ls0)))
    (OUT / "calib").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "calib" / f"chunk{a.chunk}.npz", t_eval=T_EVAL,
             null=np.array(out["null"]), known=np.array(out["known"]),
             known_truth=np.exp(np.asarray(ms.log_scales(jnp.asarray(G_KNOWN), Ze, 1, Q)[0])))
    print("null s(t):", np.array(out["null"]).round(2))
    print("known s(t):", np.array(out["known"]).round(2))


if __name__ == "__main__":
    main()
