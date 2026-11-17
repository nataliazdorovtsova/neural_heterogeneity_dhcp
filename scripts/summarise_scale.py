"""Step 3b: select the spread-with-age model, fit it, and report (RESEARCH_PLAN_BIN10 4b).

    NH_RUN=bin10_no7 python scripts/summarise_scale.py --stage select
    NH_RUN=bin10_no7 python scripts/bootstrap_scale.py --chunk 0 --n 50      # x 10
    NH_RUN=bin10_no7 python scripts/summarise_scale.py --stage report

select  paired per-infant comparison of every CV specification
        (scripts/run_scale_cv.py) against the best, the 1-SE rule over the
        layer 0/1/2 candidates (fewest parameters wins), the motion check, and
        the final fit of the selected model on the training infants.
        -> scale/selection.json, scale/final.npz
report  bootstrap bands for the scale curves and the axes of C(40); the
        shape check (young vs older axes against random splits); positions
        for steps 4-6; figure b7b.
        -> scale/summary.json, positions_scaled.csv, figures/<run>/b7b_*
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax.numpy as jnp

import paths
import figstyle as fs
from features import standardise
from model import mfa_scale as ms, splines
from scripts.run_spec import load, clean_covariates
from scripts.compare_models import baseline_pointwise
from scripts.build_bin10 import GROUP_NAMES

OUT = paths.RESULTS / "scale"
T_GRID = np.linspace(27, 45, 73)
YOUNG = 37.0                     # shape check: scanned before 37 weeks vs from 37 weeks
PMA_BINS = (0, 35, 38, 41, 99)


def hyper():
    return ms.Hyper(P=jnp.asarray(splines.difference_penalty(splines.M_DEFAULT)),
                    rank=splines.penalty_rank(splines.M_DEFAULT))


def eig(C):
    ev, V = np.linalg.eigh(np.asarray(C))
    ev, V = ev[::-1], V[:, ::-1]
    return ev, V * np.where(V[np.abs(V).argmax(0), np.arange(V.shape[1])] < 0, -1.0, 1.0)


def principal_angles(A, B):
    Qa, Qb = np.linalg.qr(A)[0], np.linalg.qr(B)[0]
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s))


# ------------------------------------------------------------------ select --

def stage_select():
    coh, X = load()
    ids = coh.dhcp_id.astype(str).values
    base = baseline_pointwise().loc[ids].values
    rows, lpd = [], {}
    for f in sorted(glob.glob(str(OUT / "cv" / "*.json"))):
        r = json.loads(open(f).read())
        d = np.load(OUT / "perinfant" / f"{r['name']}.npz", allow_pickle=True)
        s = pd.Series(d["lpd"], index=d["dhcp_id"].astype(str)).loc[ids].values
        lpd[r["name"]] = s
        rows.append(dict(name=r["name"], **r["spec"], n_params=r["n_params"],
                         total=r["total_heldout_lpd"], gain_per_infant=float((s - base).mean()),
                         all_converged=r["all_converged"],
                         worst_monotone=min(p["worst_monotone"] for p in r["per_fold"]),
                         g0_folds=[p["g"][0] for p in r["per_fold"]]))
    df = pd.DataFrame(rows)
    df["curve"] = df["curve"].fillna("quad") if "curve" in df else "quad"
    cand = df[~df.motion].copy()
    best = cand.loc[cand.total.idxmax(), "name"]
    n = len(ids)
    for i, r in df.iterrows():
        d = lpd[best] - lpd[r["name"]]
        df.loc[i, "diff_to_best"] = d.sum()
        df.loc[i, "se_to_best"] = np.sqrt(n) * d.std(ddof=1) if r["name"] != best else 0.0
        z = lpd[r["name"]] - lpd[f"L0_q{r['q']}"]           # gain over constant spread, same q
        df.loc[i, "gain_vs_L0"] = z.sum()
        df.loc[i, "se_vs_L0"] = np.sqrt(n) * z.std(ddof=1) if r["layer"] or r["motion"] else 0.0
    cand = df[~df.motion]
    within = cand[cand.diff_to_best <= cand.se_to_best]
    sel = within.sort_values(["n_params", "total"], ascending=[True, False]).iloc[0]
    df.drop(columns="g0_folds").to_csv(OUT / "comparison.csv", index=False)

    # motion check: the age part of the layer-1 curve with and without motion
    mot = {}
    for q in sorted(df.q.unique()):
        a = df[(df.layer == 1) & (df.q == q) & ~df.motion & (df.curve == "quad")].iloc[0]
        b = df[(df.layer == 1) & (df.q == q) & df.motion].iloc[0]
        u30 = (30 - ms.T_REF) / ms.T_UNIT
        s30 = lambda g: float(np.exp(g[0] * u30 + g[1] * u30 ** 2))
        mot[int(q)] = dict(s30_without=float(np.median([s30(g) for g in a.g0_folds])),
                           s30_with_motion=float(np.median([s30(g) for g in b.g0_folds])),
                           motion_coef=float(np.median([g[2] for g in b.g0_folds])),
                           lpd_gain_of_motion=float(b.total - a.total),
                           se=float(np.sqrt(n) * (lpd[b["name"]] - lpd[a["name"]]).std(ddof=1)))

    # final fit of the selected model
    layer, q, curve = int(sel.layer), int(sel.q), sel.curve
    Xs, *_ = standardise(X)
    t = coh.scan_age.values
    B, Z = splines.basis(t), ms.scale_design(t, curve=curve)
    h = hyper()
    best_fit = None
    for s in range(10):
        p0 = ms.init_params(Xs, B, Z, q, layer, np.random.default_rng(s), h,
                            g_sd=0.0 if s == 0 else 0.2)
        p, tr = ms.fit(Xs, B, Z, p0, h, layer=layer, q=q, n_iter=2000)
        if best_fit is None or float(tr[-1]) > float(best_fit[1][-1]):
            best_fit = (p, np.asarray(tr))
    p, tr = best_fit
    np.savez(OUT / "final.npz", layer=layer, q=q, **{k: np.asarray(v) for k, v in p._asdict().items()})
    # the constant-spread model at the same q, for the axis comparison
    Z = ms.scale_design(t)
    p0f, tr0 = ms.fit(Xs, B, Z, ms.init_params(Xs, B, Z, q, 0, np.random.default_rng(0), h),
                      h, layer=0, q=q, n_iter=2000)
    np.savez(OUT / "final_layer0.npz", layer=0, q=q, **{k: np.asarray(v) for k, v in p0f._asdict().items()})
    out = dict(best=best, selected=dict(name=sel["name"], layer=layer, q=q, curve=curve,
                                        total=float(sel.total), n_params=int(sel.n_params)),
               final=dict(objective=float(tr[-1]),
                          converged=bool(abs(tr[-1] - tr[-2]) / abs(tr[-1]) < 1e-6),
                          g=np.asarray(p.g).round(4).tolist()),
               within_1se=within.name.tolist(), motion_check=mot)
    (OUT / "selection.json").write_text(json.dumps(out, indent=2))
    print(df.drop(columns="g0_folds").round(2).to_string())
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ report --

def load_params(path):
    z = np.load(path)
    return int(z["layer"]), int(z["q"]), ms.Params(**{k: jnp.asarray(z[k]) for k in ms.Params._fields})


def curves(g, layer, q):
    Zg = jnp.asarray(ms.scale_design(T_GRID, curve=ms.curve_of(g)))
    ls0, ls = ms.log_scales(jnp.asarray(g), Zg, layer, q)
    return np.exp(np.asarray(ls0)), np.exp(np.asarray(ls))


def stage_report():
    sel = json.loads((OUT / "selection.json").read_text())
    layer, q, p = load_params(OUT / "final.npz")
    _, _, p0 = load_params(OUT / "final_layer0.npz")
    coh, X = load()
    Xs, mu_x, sd_x = standardise(X)[0], X.mean(0), X.std(0)
    t = coh.scan_age.values
    curve = ms.curve_of(p.g)
    B, Z = splines.basis(t), ms.scale_design(t, curve=curve)
    z40 = ms.scale_design([40.0], curve=curve)

    # axes: eigenvectors of C(40), vs the constant-spread model's
    C40 = np.asarray(ms.covariance(p, z40, layer, q))
    ev, V = eig(C40)
    ev0, V0 = eig(np.asarray(ms.covariance(p0, ms.scale_design([40.0]), 0, q)))
    n_ax = 3
    cos_vs_const = np.abs((V[:, :n_ax] * V0[:, :n_ax]).sum(0))
    share = ev[:q] / ev.sum()

    # bootstrap
    reps = []
    for f in sorted(glob.glob(str(OUT / "boot" / "chunk*.npz"))):
        z = np.load(f)
        reps += [{k: z[k][i] for k in ms.Params._fields} for i in range(len(z["g"])) if z["ok"][i]]
    s0_b, s_b, L_b, ang_b = [], [], [], []
    for r in reps:
        pb = ms.Params(**{k: jnp.asarray(v) for k, v in r.items()})
        a, b = curves(pb.g, layer, q)
        s0_b.append(a); s_b.append(b)
        evb, Vb = eig(np.asarray(ms.covariance(pb, z40, layer, q)))
        sign = np.sign((Vb[:, :q] * V[:, :q]).sum(0))
        L_b.append(Vb[:, :q] * sign * np.sqrt(evb[:q]))
        ang_b.append(principal_angles(V[:, :n_ax], Vb[:, :n_ax]).max())
    s0_b, s_b, L_b = np.array(s0_b), np.array(s_b), np.array(L_b)
    s0_hat, s_hat = curves(p.g, layer, q)
    band = lambda a: np.percentile(a, [2.5, 97.5], axis=0)

    # positions for all infants (training-fitted model, training moments)
    allc = clean_covariates(pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv"))
    feat = np.load(paths.RESULTS / "features" / f"{paths.FEATURE_SET}.npz")["X"]
    prim = allc.in_primary.values.astype(bool)
    ac, Xa = allc[prim].reset_index(drop=True), feat[prim]
    Ea = (Xa - mu_x) / sd_x - splines.basis(ac.scan_age.values) @ np.asarray(p.beta)
    Za = ms.scale_design(ac.scan_age.values, curve=curve)
    pos = pd.DataFrame(dict(dhcp_id=ac.dhcp_id.astype(str), split=ac.split,
                            scan_age=ac.scan_age.values))
    Cs = np.array([np.asarray(ms.covariance(p, z, layer, q)) for z in Za])   # (N, D, D)
    for k in range(q):
        var_k = np.einsum("d,nde,e->n", V[:, k], Cs, V[:, k])
        pos[f"axis{k + 1}"] = Ea @ V[:, k] / np.sqrt(var_k)
    for d, nme in enumerate(GROUP_NAMES):
        pos[f"z_{nme}"] = Ea[:, d] / np.sqrt(Cs[:, d, d])
    low7 = set(pd.read_csv(paths.EXCLUDE).dhcp_id.astype(str)) if paths.EXCLUDE.exists() else set()
    pos["low7"] = pos.dhcp_id.isin(low7)
    pos.to_csv(paths.RESULTS / "positions_scaled.csv", index=False)

    # does the scaling remove the age pattern of the spread? SD of the scaled
    # axis scores by PMA bin, training infants (should be close to 1 in each bin)
    tr_pos = pos[(pos.split == "train") & ~pos.low7]
    bins = pd.cut(tr_pos.scan_age, PMA_BINS, right=False)
    sd_bins = {f"axis{k + 1}": tr_pos.groupby(bins, observed=False)[f"axis{k + 1}"].std().round(3).tolist()
               for k in range(n_ax)}
    # the same projections without scaling, i.e. against the spread at 40 weeks
    E = Xs - B @ np.asarray(p.beta)
    raw = {f"axis{k + 1}": pd.Series(E @ V[:, k] / np.sqrt(ev[k])).groupby(
        pd.cut(t, PMA_BINS, right=False), observed=False).std().round(3).tolist() for k in range(n_ax)}

    # shape check: leading axes in young vs older infants, against random splits
    young = t < YOUNG
    R = E - E.mean(0)
    def top(idx):
        return eig(np.cov(R[idx].T))[1][:, :n_ax]
    obs = principal_angles(top(young), top(~young))
    rng = np.random.default_rng(11)
    null = np.array([principal_angles(top(perm[:young.sum()]), top(perm[young.sum():]))
                     for perm in (rng.permutation(len(t)) for _ in range(1000))])
    p_shape = ((null >= obs).sum(0) + 1) / (len(null) + 1)

    # calibration (scripts/calibrate_scale.py): fitted s(t) when the truth is
    # no age scaling (null) or a known curve, layer 1, q = 6
    calib = None
    cf = sorted(glob.glob(str(OUT / "calib" / "chunk*.npz")))
    if cf:
        cz = [np.load(f) for f in cf]
        nul = np.concatenate([c["null"] for c in cz]); kno = np.concatenate([c["known"] for c in cz])
        q3 = lambda a: np.percentile(a, [50, 2.5, 97.5], axis=0).round(3).tolist()
        calib = dict(t_eval=cz[0]["t_eval"].tolist(), n_rep=len(nul),
                     null_median_lo_hi=q3(nul), known_truth=cz[0]["known_truth"].round(3).tolist(),
                     known_median_lo_hi=q3(kno),
                     known_median_log_bias=np.median(np.log(kno) - np.log(cz[0]["known_truth"]), 0).round(3).tolist())

    # spread at 30 weeks relative to 40, with bands
    i30 = np.searchsorted(T_GRID, 30)
    summ = dict(selected=sel["selected"], n_boot=len(reps),
                s0_at_30=[float(s0_hat[i30]), *band(s0_b[:, i30]).round(3).tolist()],
                s_axes_at_30=[[float(s_hat[i30, k]), *band(s_b[:, i30, k]).round(3).tolist()]
                              for k in range(q)],
                share_C40=share.round(4).tolist(),
                loadings=(V[:, :q] * np.sqrt(ev[:q])).round(3).tolist(),
                loadings_ci=np.percentile(L_b, [2.5, 97.5], axis=0).round(3).tolist() if len(L_b) else None,
                axes_cos_vs_constant=cos_vs_const.round(3).tolist(),
                boot_angle_1_3=[float(np.median(ang_b)), float(np.percentile(ang_b, 95))] if ang_b else None,
                sd_scaled_scores_by_bin=sd_bins, sd_unscaled_by_bin=raw, pma_bins=list(PMA_BINS),
                shape_check=dict(n_young=int(young.sum()), n_older=int((~young).sum()),
                                 angles=obs.round(1).tolist(),
                                 null_median=np.median(null, 0).round(1).tolist(),
                                 null_95th=np.percentile(null, 95, 0).round(1).tolist(),
                                 p=p_shape.round(3).tolist()),
                calibration=calib)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))

    fs.apply_style()
    figure(sel, layer, q, s0_hat, s_hat, s0_b, s_b, E, t, ev, V, Cs, obs, null)


def figure(sel, layer, q, s0_hat, s_hat, s0_b, s_b, E, t, ev, V, Cs, obs, null):
    df = pd.read_csv(OUT / "comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.1, 1.2, 0.8]))
    fig.subplots_adjust(wspace=0.3, bottom=0.27)

    # A: gain over the constant spread
    ax = axes[0]
    if "curve" not in df:
        df["curve"] = "quad"
    df["curve"] = df["curve"].fillna("quad")
    styles = {(1, False, "quad"): ("one age scale, quadratic", fs.COLOR_CYCLE[0], "o", "-", -0.18),
              (1, False, "hinge"): ("one age scale, hinged", fs.COLOR_CYCLE[2], "D", "-", -0.06),
              (2, False, "quad"): ("a scale per axis, quadratic", fs.COLOR_CYCLE[1], "s", "-", 0.06),
              (1, True, "quad"): ("quadratic + motion (check)", fs.NEUTRAL_GREY, "^", "--", 0.18)}
    off_of = {}
    for (lay, mot, cv), (lab, col, mk, ls, off) in styles.items():
        d = df[(df.layer == lay) & (df.motion == mot) & (df.curve == cv)].sort_values("q")
        if not len(d):
            continue
        off_of[(lay, cv)] = off
        ax.errorbar(d.q + off, d.gain_vs_L0, yerr=d.se_vs_L0, color=col, marker=mk, ls=ls,
                    capsize=3, label=lab)
    s = sel["selected"]
    srow = df[df.name == s["name"]].iloc[0]
    ax.plot(srow.q + off_of.get((s["layer"], s.get("curve", "quad")), 0), srow.gain_vs_L0, marker="*", ms=18,
            color=fs.TEXT_PRIMARY, ls="none", zorder=5, label=f"selected: {s['name']}")
    fs.reference_line(ax, 0)
    ax.set_xlabel("number of axes q"); ax.set_ylabel("held-out log density\nvs constant spread (total)")
    ax.set_ylim(-15, 285)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.8)
    fs.panel_title(ax, "A", "Does an age scale predict better?")

    # B: scale curves with bands, and the binned spread they should follow
    ax = axes[1]
    band = lambda a: np.percentile(a, [2.5, 97.5], axis=0)
    if layer == 1 or q == 0:
        lo, hi = band(s0_b) if len(s0_b) else (s0_hat, s0_hat)
        ax.fill_between(T_GRID, lo, hi, color=fs.COLOR_CYCLE[0], alpha=0.25, lw=0)
        ax.plot(T_GRID, s0_hat, color=fs.COLOR_CYCLE[0], lw=2.2, label="scale of the whole spread")
    else:
        lo, hi = band(s0_b)
        ax.fill_between(T_GRID, lo, hi, color=fs.NEUTRAL_GREY, alpha=0.25, lw=0)
        ax.plot(T_GRID, s0_hat, color=fs.NEUTRAL_GREY, lw=2, ls="--", label="lobe-specific part")
        for k in range(q):
            lo, hi = band(s_b[:, :, k])
            col = fs.COLOR_CYCLE[k % len(fs.COLOR_CYCLE)]
            ax.fill_between(T_GRID, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(T_GRID, s_hat[:, k], color=col, lw=2, label=f"factor {k + 1}")
    # observed: root mean square of the whitened scores against the 40-week
    # spread, separately for the leading q axes and the other directions
    W40 = (E @ V) / np.sqrt(ev)
    edges = np.array([27, 32, 35, 37, 39, 41, 43, 45.5])
    mids, lead, rest = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.sum() >= 10:
            mids.append(t[m].mean())
            lead.append(np.sqrt((W40[m, :q] ** 2).mean()))
            rest.append(np.sqrt((W40[m, q:] ** 2).mean()))
    print("panel B bins", np.round(mids, 1).tolist(), "leading", np.round(lead, 2).tolist(),
          "other", np.round(rest, 2).tolist())
    ax.plot(mids, lead, "o", color=fs.TEXT_PRIMARY, ms=7, label=f"observed, leading {q} axes")
    ax.plot(mids, rest, "o", mfc="none", mec=fs.TEXT_SECONDARY, mew=1.5, ms=7,
            label="observed, the other directions")
    fs.reference_line(ax, 1)
    ax.set_xlabel("PMA at scan (weeks)"); ax.set_ylabel("spread relative to 40 weeks")
    ax.legend(fontsize=9, loc="upper right")
    fs.panel_title(ax, "B", "How wide is the spread at each age?")

    # C: shape check
    ax = axes[2]
    parts = ax.violinplot([null[:, j] for j in range(null.shape[1])], positions=range(1, null.shape[1] + 1),
                          showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(fs.NEUTRAL_GREY); b.set_alpha(0.4)
    ax.plot(range(1, null.shape[1] + 1), obs, "D", color=fs.COLOR_CYCLE[4], ms=9,
            label=f"< {YOUNG:.0f} vs ≥ {YOUNG:.0f} weeks")
    ax.set_xticks(range(1, null.shape[1] + 1))
    ax.set_xlabel("principal angle (axes 1-3)"); ax.set_ylabel("degrees")
    ax.legend(handles=[Line2D([], [], marker="D", color=fs.COLOR_CYCLE[4], ls="none", ms=9,
                              label=f"< {YOUNG:.0f} vs ≥ {YOUNG:.0f} weeks"),
                       Line2D([], [], marker="s", color=fs.NEUTRAL_GREY, alpha=0.5, ls="none", ms=12,
                              label="random splits, same sizes")], fontsize=9, loc="upper left")
    fs.panel_title(ax, "C", "Same axes when young?")

    fig.text(0.05, 0.04,
             f"Run {paths.RUN}, {len(t)} training infants. A: total held-out log density over 5 CV folds "
             f"minus the constant-spread model at the same q; bars: 1 SE of the paired per-infant difference.\n"
             f"B: fitted scale curves (1 = the spread at 40 weeks), 95% bands from {len(s0_b)} parametric "
             f"bootstrap replicates; points: RMS of the scores whitened by the 40-week spread, bins of ≥ 10 infants.\n"
             f"C: angles between the leading 3 axes of infants scanned before and after {YOUNG:.0f} weeks, against "
             "1,000 random splits of the same sizes (an angle of 0 = the same axes).",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "b7b_spread_age_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("select", "report"), required=True)
    a = ap.parse_args()
    stage_select() if a.stage == "select" else stage_report()


if __name__ == "__main__":
    main()
