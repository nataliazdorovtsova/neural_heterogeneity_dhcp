"""Checks for model/mfa_scale.py before it is used (RESEARCH_PLAN_BIN10 section 4b).

    NH_RUN=bin10_no7 python scripts/test_mfa_scale.py

1. Layer 0 reproduces model/mfa.py (K = 1, q = 6) on the training infants:
   same penalised log likelihood and the same per-infant log densities.
2. Recovery: data simulated from a layer-2 truth (q = 3, the real ages and
   n = 562) with known scale curves; layers 0, 1 and 2 are fitted, and the
   recovered curves are compared with the truth over 28-44 weeks.
3. A null simulation (no age scaling): the fitted layer-1 curve stays near 1.
4. Monotonicity: no sweep decreases the objective by more than 1e-8 relative.

Writes results/<run>/scale/test_mfa_scale.json.
"""

import json
import time

import numpy as np
import jax.numpy as jnp

import paths
from features import standardise
from model import mfa, mfa_scale as ms, splines
from scripts.run_spec import load

OUT = paths.RESULTS / "scale"
N_ITER = 1000
T_GRID = np.linspace(28, 44, 33)


def hyper():
    return ms.Hyper(P=jnp.asarray(splines.difference_penalty(splines.M_DEFAULT)),
                    rank=splines.penalty_rank(splines.M_DEFAULT))


def monotone(trace):
    tr = np.asarray(trace)
    worst = np.min(np.diff(tr) / np.maximum(np.abs(tr[1:]), 1.0))
    return float(worst)


def curves(p, layer, q):
    ls0, ls = ms.log_scales(p.g, jnp.asarray(ms.scale_design(T_GRID)), layer, q)
    return np.exp(np.asarray(ls0)), np.exp(np.asarray(ls))


def fit_best(X, B, Z, q, layer, h, n_start=3):
    best = None
    for s in range(n_start):
        p0 = ms.init_params(X, B, Z, q, layer, np.random.default_rng(s), h,
                            g_sd=0.0 if s == 0 else 0.2)
        p, tr = ms.fit(X, B, Z, p0, h, layer=layer, q=q, n_iter=N_ITER)
        if best is None or float(tr[-1]) > float(best[1][-1]):
            best = (p, tr)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coh, X = load()
    X, *_ = standardise(X)
    t = coh.scan_age.values
    B = splines.basis(t)
    Z = ms.scale_design(t)
    h = hyper()
    rep = {}

    # 1. layer 0 vs model/mfa.py
    t0 = time.time()
    q = 6
    h1 = mfa.Hyper(P=h.P, rank=h.rank)
    p_old, tr_old = mfa.fit(X, B, mfa.init_params(X, B, 1, q, np.random.default_rng(0), h1),
                            h1, n_iter=N_ITER)
    p_new, tr_new = ms.fit(X, B, Z, ms.init_params(X, B, Z, q, 0, np.random.default_rng(0), h),
                           h, layer=0, q=q, n_iter=N_ITER)
    lpd_old = np.asarray(mfa.pointwise_log_density(X, B, p_old))
    lpd_new = np.asarray(ms.pointwise_log_density(X, B, Z, p_new, 0, q))
    rep["layer0_vs_mfa"] = dict(obj_old=float(tr_old[-1]), obj_new=float(tr_new[-1]),
                                max_abs_lpd_diff=float(np.abs(lpd_old - lpd_new).max()),
                                total_lpd_old=float(lpd_old.sum()), total_lpd_new=float(lpd_new.sum()),
                                worst_monotone=monotone(tr_new), seconds=time.time() - t0)
    print("layer 0 vs mfa:", rep["layer0_vs_mfa"])

    # 2. recovery from a layer-2 truth
    q = 3
    rng = np.random.default_rng(42)
    p_true = ms.init_params(X, B, Z, q, 2, np.random.default_rng(0), h)
    g_true = np.zeros((q + 1, 2))
    g_true[0] = [-0.10, 0.02]          # lobe-specific part: 1.3x wider at 30 weeks
    g_true[1] = [-0.30, 0.04]          # axis 1: about 2.5x wider at 30 weeks
    g_true[2] = [0.0, 0.0]             # axis 2: constant
    g_true[3] = [0.10, 0.0]            # axis 3: narrower when young
    p_true = p_true._replace(g=jnp.asarray(g_true))
    Xs = ms.simulate(rng, B, Z, p_true, 2, q)
    s0_true, s_true = curves(p_true, 2, q)
    rec = {}
    for layer in (0, 1, 2):
        t0 = time.time()
        p, tr = fit_best(Xs, B, Z, q, layer, h)
        s0, s = curves(p, layer, q)
        rec[f"layer{layer}"] = dict(
            loglik=float(np.asarray(ms.pointwise_log_density(Xs, B, Z, p, layer, q)).sum()),
            g=np.asarray(p.g).round(3).tolist(),
            s0_at_30=float(s0[np.searchsorted(T_GRID, 30)]),
            s_at_30=s[np.searchsorted(T_GRID, 30)].round(3).tolist(),
            max_log_err_s0=float(np.abs(np.log(s0) - np.log(s0_true)).max()),
            worst_monotone=monotone(tr), seconds=time.time() - t0)
        if layer == 2:
            # the axes of W are identified only up to order and sign: match by |cos|
            Wt, Wf = np.asarray(p_true.W), np.asarray(p.W)
            cos = np.abs((Wt / np.linalg.norm(Wt, axis=0)).T @ (Wf / np.linalg.norm(Wf, axis=0)))
            match = cos.argmax(1)
            rec["layer2"]["axis_match"] = match.tolist()
            rec["layer2"]["axis_cos"] = cos.max(1).round(3).tolist()
            rec["layer2"]["max_log_err_axes"] = float(
                np.abs(np.log(s[:, match]) - np.log(s_true)).max())
        print(f"recovery layer {layer}:", rec[f"layer{layer}"])
    rec["truth"] = dict(g=g_true.tolist(), s0_at_30=float(s0_true[np.searchsorted(T_GRID, 30)]),
                        s_at_30=s_true[np.searchsorted(T_GRID, 30)].round(3).tolist())
    rep["recovery"] = rec

    # 3. null simulation
    p_null = p_true._replace(g=jnp.zeros((q + 1, 2)))
    Xn = ms.simulate(np.random.default_rng(7), B, Z, p_null, 2, q)
    p, tr = fit_best(Xn, B, Z, q, 1, h)
    s0, _ = curves(p, 1, q)
    rep["null_layer1"] = dict(g=np.asarray(p.g).round(3).tolist(),
                              s_range=[float(s0.min()), float(s0.max())],
                              worst_monotone=monotone(tr))
    print("null layer 1:", rep["null_layer1"])

    (OUT / "test_mfa_scale.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
