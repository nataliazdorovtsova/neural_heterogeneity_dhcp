"""Parametric bootstrap of the selected one-group model (methods plan, section 4).

    python scripts/bootstrap.py --chunk 0 --n-boot 10       # array task

Each replicate simulates a new dataset of 569 infants from the fitted model, at
the observed ages exactly (so the age and prematurity distribution matches by
construction), and refits the same specification. The refit starts from the
fitted parameters plus two random starts, and keeps the best.

The axes of a refit are only defined up to rotation, so each replicate's
loadings are rotated onto the original principal loadings by orthogonal
Procrustes before anything is compared. Small axes can trade places between
replicates, which the Procrustes step absorbs into wider intervals; that is the
honest uncertainty for axes of similar size.

These intervals describe sampling variability given the selected model; they
do not include model-selection uncertainty.

Writes results/significance/bootstrap/chunk{c}.npz.
"""

import argparse
import json
import time

import numpy as np

import paths
from model import mfa, splines
from scripts.run_spec import load, age_for, CLOCK
from scripts.fit_final import fit_best, principal

OUT = paths.RESULTS / "significance" / "bootstrap"
SEED = 83_000


def selected_model():
    sel = json.loads((paths.RESULTS / "selection.json").read_text())["selected"]
    if sel["K"] != 1:
        raise SystemExit(f"bootstrap is written for K = 1; selection is K = {sel['K']}")
    d = np.load(paths.RESULTS / "final" / f"K1_q{sel['q']}.npz")
    p = mfa.Params(log_pi=d["log_pi"], beta=d["beta"], W=d["W"], psi=d["psi"], lam=d["lam"])
    return sel["q"], p


def procrustes(W, target):
    """Orthogonal R minimising ||W R - target||."""
    U, _, Vt = np.linalg.svd(W.T @ target)
    return U @ Vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--n-boot", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=1000)
    a = ap.parse_args()

    q, p_hat = selected_model()
    coh, _ = load()
    B = splines.basis(age_for(coh, CLOCK))
    L_hat, S2_hat, _, _ = principal(p_hat.W[0], p_hat.psi[0])
    p_hat = mfa.Params(*(np.asarray(x) for x in p_hat))

    loadings, S2, beta, conv, secs = [], [], [], [], []
    for i in range(a.n_boot):
        t0 = time.time()
        b = a.chunk * a.n_boot + i
        Xb, _ = mfa.simulate(np.random.default_rng(SEED + b), B, p_hat)
        p, tr, c, _ = fit_best(Xb, B, 1, q, n_init=2, n_iter=a.n_iter,
                               seed_base=SEED * 10 + 10 * b, extra_inits=[p_hat])
        W = np.asarray(p.W[0])
        loadings.append(W @ procrustes(W, L_hat))
        S2.append(principal(W, p.psi[0])[1])
        beta.append(np.asarray(p.beta[0]))
        conv.append(c); secs.append(time.time() - t0)
        print(f"boot {b}: converged={c}, {secs[-1]:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"chunk{a.chunk}.npz", q=q, loadings=np.array(loadings), S2=np.array(S2),
             beta=np.array(beta), converged=np.array(conv), seconds=np.array(secs),
             first_boot=a.chunk * a.n_boot)


if __name__ == "__main__":
    main()
