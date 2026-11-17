"""Parametric bootstrap of the selected spread-with-age model (RESEARCH_PLAN_BIN10 4b).

    NH_RUN=bin10_no7 python scripts/bootstrap_scale.py --chunk 0 --n 50

Each replicate simulates the training infants' lobe values from the fitted
model (scale/final.npz) at their own ages and refits the same model, starting
from the fitted parameters. Seeds are 1000 * chunk + replicate.
Writes scale/boot/chunk<c>.npz with every replicate's parameters.
"""

import argparse

import numpy as np
import jax.numpy as jnp

import paths
from features import standardise
from model import mfa_scale as ms, splines
from scripts.run_spec import load
from scripts.summarise_scale import load_params, hyper, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    layer, q, p = load_params(OUT / "final.npz")
    coh, _ = load()
    t = coh.scan_age.values
    B, Z = splines.basis(t), ms.scale_design(t, curve=ms.curve_of(p.g))
    h = hyper()
    keep = {k: [] for k in ms.Params._fields}
    ok = []
    for r in range(a.n):
        Xb = ms.simulate(np.random.default_rng(1000 * a.chunk + r), B, Z, p, layer, q)
        Xb, *_ = standardise(Xb)
        pb, tr = ms.fit(Xb, B, Z, p, h, layer=layer, q=q, n_iter=1000)
        tr = np.asarray(tr)
        ok.append(bool(np.isfinite(tr[-1]) and abs(tr[-1] - tr[-2]) / abs(tr[-1]) < 1e-6))
        for k in ms.Params._fields:
            keep[k].append(np.asarray(getattr(pb, k)))
    (OUT / "boot").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "boot" / f"chunk{a.chunk}.npz", ok=np.array(ok),
             **{k: np.stack(v) for k, v in keep.items()})
    print(f"chunk {a.chunk}: {sum(ok)}/{a.n} converged")


if __name__ == "__main__":
    main()
