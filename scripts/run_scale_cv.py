"""Cross-validate the age-scaled spread models (RESEARCH_PLAN_BIN10 section 4b).

    NH_RUN=bin10_no7 python scripts/run_scale_cv.py --index 0
    NH_RUN=bin10_no7 python scripts/run_scale_cv.py --list

One task per specification: layer 0 (constant spread), layer 1 (one age scale),
layer 2 (a scale per axis and one for the lobe-specific part), and layer 1 with
log motion in the scale (a check, not a selection candidate). Each runs the
frozen 5 folds with N_START starts (start 0 with flat scale curves, the others
with random ones) and keeps the best penalised fit per fold.

Writes results/<run>/scale/perinfant/<name>.npz (held-out log density per
infant) and results/<run>/scale/cv/<name>.json.
"""

import argparse
import json
import time

import numpy as np
import jax.numpy as jnp

import paths
from features import standardise
from model import mfa_scale as ms, splines
from scripts.run_spec import load

OUT = paths.RESULTS / "scale"
N_START = 5
N_ITER = 1000
Q_ALL = range(0, 8)

SPECS = ([dict(layer=0, q=q, motion=False) for q in Q_ALL]
         + [dict(layer=1, q=q, motion=False) for q in Q_ALL]
         + [dict(layer=2, q=q, motion=False) for q in Q_ALL if q > 0]
         + [dict(layer=1, q=q, motion=True) for q in Q_ALL]
         + [dict(layer=1, q=q, motion=False, curve="hinge") for q in Q_ALL])   # added after the
# first 31 ran: the quadratic overshot the youngest bin (figure b7b panel B)


def spec_name(s):
    h = "h" if s.get("curve", "quad") == "hinge" else ""
    return f"L{s['layer']}{h}_q{s['q']}" + ("_motion" if s["motion"] else "")


def log_motion(coh):
    m = np.log(coh.qc_dmri_eddy_rel_motion.values.astype(float))
    return m


def run(spec):
    coh, X = load()
    t = coh.scan_age.values
    folds = coh.fold.values
    lm = log_motion(coh)
    h = ms.Hyper(P=jnp.asarray(splines.difference_penalty(splines.M_DEFAULT)),
                 rank=splines.penalty_rank(splines.M_DEFAULT))
    layer, q = spec["layer"], spec["q"]
    curve = spec.get("curve", "quad")
    ids, lpd, per_fold = [], [], []
    for f in sorted(np.unique(folds)):
        fi, oi = np.where(folds != f)[0], np.where(folds == f)[0]
        Xf, Xo, *_ = standardise(X[fi], X[oi])
        Bf, Bo = splines.basis(t[fi]), splines.basis(t[oi])
        if spec["motion"]:
            mu, sd = np.nanmean(lm[fi]), np.nanstd(lm[fi])
            zf = np.nan_to_num((lm[fi] - mu) / sd)      # missing motion -> the mean
            zo = np.nan_to_num((lm[oi] - mu) / sd)
            Zf, Zo = ms.scale_design(t[fi], zf, curve), ms.scale_design(t[oi], zo, curve)
        else:
            Zf, Zo = ms.scale_design(t[fi], curve=curve), ms.scale_design(t[oi], curve=curve)
        best, n_ok = None, 0
        for s in range(N_START if layer > 0 else 1):     # layer 0 is deterministic at K = 1
            p0 = ms.init_params(Xf, Bf, Zf, q, layer, np.random.default_rng(10_000 * f + s), h,
                                g_sd=0.0 if s == 0 else 0.2)
            p, tr = ms.fit(Xf, Bf, Zf, p0, h, layer=layer, q=q, n_iter=N_ITER)
            tr = np.asarray(tr)
            if not np.isfinite(tr[-1]):
                continue
            n_ok += 1
            if best is None or tr[-1] > best[1][-1]:
                best = (p, tr)
        p, tr = best
        conv = abs(tr[-1] - tr[-2]) / max(abs(tr[-1]), 1.0) < 1e-6
        worst = float(np.min(np.diff(tr) / np.maximum(np.abs(tr[1:]), 1.0)))
        l_i = np.asarray(ms.pointwise_log_density(Xo, Bo, Zo, p, layer, q))
        ids.append(coh.dhcp_id.astype(str).values[oi]); lpd.append(l_i)
        per_fold.append(dict(fold=int(f), heldout_lpd=float(l_i.sum()), objective=float(tr[-1]),
                             converged=bool(conv), worst_monotone=worst, starts_ok=n_ok,
                             g=np.asarray(p.g).round(4).tolist()))
        print(f"  fold {f}: held-out {l_i.sum():9.1f}  converged={conv}  "
              f"worst step {worst:.1e}  g0={np.asarray(p.g)[0].round(3)}")
    return (dict(dhcp_id=np.concatenate(ids), lpd=np.concatenate(lpd)),
            dict(spec=spec, name=spec_name(spec), per_fold=per_fold,
                 total_heldout_lpd=sum(d["heldout_lpd"] for d in per_fold),
                 all_converged=all(d["converged"] for d in per_fold),
                 n_params=ms.n_params(X.shape[1], q, layer, J=Zf.shape[1]),
                 n_start=N_START, n_iter=N_ITER))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, s in enumerate(SPECS):
            print(i, spec_name(s))
        return
    spec = SPECS[a.index]
    print(f"[{a.index}] {spec_name(spec)}  (run {paths.RUN})")
    t0 = time.time()
    per_infant, res = run(spec)
    res["seconds"] = time.time() - t0
    (OUT / "perinfant").mkdir(parents=True, exist_ok=True)
    (OUT / "cv").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "perinfant" / f"{res['name']}.npz", **per_infant)
    (OUT / "cv" / f"{res['name']}.json").write_text(json.dumps(res, indent=2))
    print(f"  total held-out {res['total_heldout_lpd']:.1f}  ({res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
