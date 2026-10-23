"""Fit one model specification across the frozen CV folds.

    python scripts/run_spec.py --index 0          # one spec
    python scripts/run_spec.py --list             # show the grid

Designed to be the body of a SLURM array task: one task per specification, each
doing its own 5 folds x N initialisations internally. See CLUSTER.md for why we
bundle this way rather than submitting one job per fit.
"""

import argparse
import time

import numpy as np
import pandas as pd
import jax.numpy as jnp

import paths
import provenance as prov
from features import standardise
from model import mfa, splines

K_GRID = (1, 2, 3, 4, 5)
Q_GRID = (0, 1, 2, 5, 10, 15, 25)
N_INIT = 20
N_ITER = 250
M = splines.M_DEFAULT
VARIANT = "perbin25_mst"
CLOCK = "pma"

# Appended 2026-10-16 (indices 35-37): at K = 1 the held-out score was still
# rising at q = 25 (SUPPLEMENT S15). Appended, not inserted, so that existing
# array indices keep pointing at the same specifications.
EXTRA_K1_Q = (35, 50, 70)

SPECS = [dict(K=K, q=q, M=M, clock=CLOCK, variant=VARIANT,
              covariance_sharing="group_specific", trajectory="group_smooth")
         for K in K_GRID for q in Q_GRID] + \
        [dict(K=1, q=q, M=M, clock=CLOCK, variant=VARIANT,
              covariance_sharing="group_specific", trajectory="group_smooth")
         for q in EXTRA_K1_Q]

PERINFANT = paths.RESULTS / "perinfant"


def spec_name(s):
    base = f"K{s['K']}_q{s['q']}_M{s['M']}_{s['clock']}_{s['covariance_sharing']}"
    # Other feature sets are named apart, so they can never be mistaken for
    # (or overwrite) the primary model-selection records.
    return base if s.get("variant", VARIANT) == VARIANT else f"{base}_{s['variant']}"


def load(variant=VARIANT):
    coh = pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv")
    feat = np.load(paths.RESULTS / "features" / f"{variant}.npz")
    X = feat["X"].reshape(len(feat["X"]), -1)           # (n, 270)
    prim = coh.in_primary.values & (coh.split == "train").values
    return coh[prim].reset_index(drop=True), X[prim]


def age_for(coh, clock):
    if clock == "pma":
        return coh.scan_age.values
    if clock == "pna":
        return coh.pna.values
    raise ValueError(f"clock {clock!r} needs the additive-basis variant")


def run(spec, verbose=True):
    coh, X = load(spec.get("variant", VARIANT))
    t = age_for(coh, spec["clock"])
    folds = coh.fold.values
    h = mfa.Hyper(P=jnp.asarray(splines.difference_penalty(spec["M"])),
                  rank=splines.penalty_rank(spec["M"]))

    per_fold, n_restarts_used = [], 0
    ids, fold_of, lpd = [], [], []
    t0 = time.time()
    for f in sorted(np.unique(folds)):
        fit_idx, out_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        Xf, Xo, *_ = standardise(X[fit_idx], X[out_idx])     # fit-fold moments only
        Bf = splines.basis(t[fit_idx], M=spec["M"])
        Bo = splines.basis(t[out_idx], M=spec["M"])

        # All restarts are fitted in one vectorised pass: one compile, and the
        # per-restart cost drops from ~6.9 s to ~5.0 s.
        inits = [mfa.init_params(Xf, Bf, spec["K"], spec["q"],
                                 np.random.default_rng(10_000 * f + s), h)
                 for s in range(N_INIT)]
        p_all, traces = mfa.fit_restarts(Xf, Bf, mfa.stack_params(inits), h, n_iter=N_ITER)
        obj = np.asarray(traces)[:, -1]                      # (N_INIT,)
        finite = np.isfinite(obj)
        if not finite.any():
            return None
        n_restarts_used += int(finite.sum())
        b = int(np.where(finite, obj, -np.inf).argmax())
        best = mfa.unstack_params(p_all, b)
        best_obj = float(obj[b])
        tr = np.asarray(traces)[b]
        converged = abs(tr[-1] - tr[-2]) / max(abs(best_obj), 1.0) < 1e-6
        lpd_i = np.asarray(mfa.pointwise_log_density(Xo, Bo, best))
        score = float(lpd_i.sum())
        ids.append(coh.dhcp_id.values[out_idx]); fold_of.append(np.full(len(out_idx), f))
        lpd.append(lpd_i)
        resp = np.asarray(mfa.responsibilities(Xo, Bo, best))
        per_fold.append(dict(fold=int(f), n_fit=len(fit_idx), n_out=len(out_idx),
                             penalised_loglik=best_obj, heldout_lpd=score,
                             converged=bool(converged),
                             min_group_share=float(resp.mean(0).min())))
        if verbose:
            print(f"   fold {f}: held-out lpd {score:11.1f}  "
                  f"converged={converged}  smallest group {per_fold[-1]['min_group_share']:.3f}")

    total = sum(d["heldout_lpd"] for d in per_fold)
    elapsed = time.time() - t0
    per_infant = dict(dhcp_id=np.concatenate(ids).astype(str),
                      fold=np.concatenate(fold_of), lpd=np.concatenate(lpd))
    return per_infant, dict(per_fold=per_fold, total_heldout_lpd=total,
                mean_heldout_lpd_per_infant=total / sum(d["n_out"] for d in per_fold),
                all_converged=all(d["converged"] for d in per_fold),
                successful_restarts=n_restarts_used,
                seconds=elapsed)


def main():
    global N_ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n-iter", type=int, default=None,
                    help=f"ECM sweeps per restart (default {N_ITER}); raise for specs that did not converge")
    ap.add_argument("--variant", default=VARIANT,
                    help="feature set in results/features/ (default: the primary network)")
    ap.add_argument("--K", type=int, help="with --q: fit this specification instead of --index")
    ap.add_argument("--q", type=int)
    a = ap.parse_args()
    if a.n_iter is not None:
        N_ITER = a.n_iter
    if a.list:
        for i, s in enumerate(SPECS):
            print(f"{i:3d}  {spec_name(s)}")
        return
    if a.K is not None and a.q is not None:
        spec = dict(K=a.K, q=a.q, M=M, clock=CLOCK, variant=a.variant,
                    covariance_sharing="group_specific", trajectory="group_smooth")
    else:
        spec = dict(SPECS[a.index], variant=a.variant)
    print(f"[{a.index if a.index is not None else 'K,q'}] {spec_name(spec)}")
    out = run(spec)
    if out is None:
        print("   FAILED: no initialisation produced a finite fit")
        return
    per_infant, res = out
    print(f"   total held-out lpd {res['total_heldout_lpd']:.1f}  "
          f"({res['seconds']:.0f}s, {res['successful_restarts']}/{N_INIT*5} restarts usable)")

    rec = prov.snapshot(
        name=spec_name(spec), config=dict(spec, n_init=N_INIT, n_iter=N_ITER,
                                          seed_rule="10000*fold + restart"),
        results=res,
        thresholds=dict(
            psi_floor=prov.threshold(1e-6, "fraction of each coordinate's marginal variance"),
            convergence=prov.threshold(1e-6, "relative change in penalised log lik at final sweep"),
            lam_prior=prov.threshold([1e-3, 1e-3], "Gamma(a0,b0) on smoothness precision"),
        ),
        inputs=[paths.RESULTS / "features" / f"{spec['variant']}.npz",
                paths.RESULTS / "cohort" / "cohort.csv"])
    # Each infant's held-out log density, for paired comparisons between models
    # (scripts/compare_models.py). Named after the provenance record it belongs to.
    PERINFANT.mkdir(parents=True, exist_ok=True)
    np.savez(PERINFANT / f"{rec.stem}.npz", **per_infant)
    print(f"   provenance -> {rec.name}; per-infant scores -> perinfant/{rec.stem}.npz")


if __name__ == "__main__":
    main()
