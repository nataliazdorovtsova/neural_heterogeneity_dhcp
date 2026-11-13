"""Step 3 of RESEARCH_PLAN_BIN10.md: trajectories and the spread around them.

    NH_RUN=bin10 python scripts/summarise_bin10_step3.py       # or NH_RUN=bin10_no7

For the selected one-group model (fitted on the training infants):

  trajectories  mu(t) = B(t) beta per lobe, in original units, with pointwise
                95% bands from the parametric bootstrap; the change from 30 to
                42 weeks in SDs of the spread (sqrt diag C)
  spread        C = W W' + Psi per lobe, split into the shared part (diag W W',
                carried by the axes) and the lobe-specific part (psi); and
                whether the spread changes with age (Brown-Forsythe test over
                PMA bins) or with prematurity at overlapping ages
  axes          eigenvectors of C, largest first, largest-magnitude loading
                positive (as S31). Bootstrap: each replicate's eigenvectors are
                matched to the reference (Hungarian on |cos|) and signed, giving
                intervals for loadings and shares and principal angles.
                Permutation null on the share of each rank; split-half |cos|
                between the halves' matched axes, with chance simulated in 12
                dimensions (which is high: random 12-d vectors are not orthogonal)
  artefacts     axis scores vs motion and pre-threshold density (age-adjusted),
                and the mean axis score of the 7 orbitofrontal-low infants and,
                for bin10, of the 24-infant small group of the K = 2 fit (S35)
  positions     axis scores and per-lobe z for all 670 infants (training-fitted
                model, training moments), saved for steps 4-6

Writes results/<run>/step3_summary.json, results/<run>/positions.csv and
figures/<run>/b6-b9.
"""

import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from scipy.stats import levene

import paths
import atlas
import figstyle as fs
from model import mfa, splines
from scripts.build_bin10 import GROUP_NAMES
from scripts.pilot_bin10 import Curve
from scripts.run_spec import load, clean_covariates

SIG = paths.RESULTS / "significance"
T = np.linspace(27, 45, 181)
PMA_BINS = (0, 35, 38, 41, 99)
LOBE_COLOURS = dict(zip(atlas.LOBES, fs.COLOR_CYCLE[:6]))
COL = [LOBE_COLOURS[n.rsplit("_", 1)[0]] for n in GROUP_NAMES]
SHORT = [f"{n[:4]} {n[-1]}" for n in GROUP_NAMES]
XPOS = np.array([i + 0.4 * (i // 2) for i in range(12)])      # lobe pairs, gap between lobes


def eig_axes(W, psi, q):
    C = W @ W.T + np.diag(psi)
    ev, V = np.linalg.eigh(C)
    ev, V = ev[::-1], V[:, ::-1]
    V = V[:, :q] * np.where(V[np.abs(V[:, :q]).argmax(0), np.arange(q)] < 0, -1.0, 1.0)
    return C, ev, V


def align(V_ref, V):
    """Match V's columns to V_ref's (Hungarian on |cos|), then sign them."""
    cos = V_ref.T @ V
    r, c = linear_sum_assignment(-np.abs(cos))
    out = V[:, c] * np.sign(cos[r, c])
    return out, c


def chance_cos(V_ref, n_sim=500, rng=np.random.default_rng(5)):
    """95th percentile of matched |cos| between two random q-axis sets in D dims."""
    D, q = V_ref.shape
    sims = []
    for _ in range(n_sim):
        A = np.linalg.qr(rng.standard_normal((D, q)))[0]
        Bm = np.linalg.qr(rng.standard_normal((D, q)))[0]
        A, _ = align(V_ref, A); Bm, _ = align(V_ref, Bm)
        sims.append(np.abs((A * Bm).sum(0)))
    return np.percentile(sims, 95, axis=0)


def resid_on_age(t, y):
    ok = np.isfinite(y)
    r = np.full(len(y), np.nan)
    r[ok] = Curve(t[ok], y[ok]).resid
    return r


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def small_group_ids():
    """bin10 only: the infants in the small group of the K = 2, q = 4 fit (S35)."""
    f = paths.PROJECT / "results" / "bin10" / "final" / "K2_q4.npz"
    if paths.RUN != "bin10" or not f.exists():
        return set()
    from scripts.validate_once import fitted
    coh, X = load()
    d, p = fitted(2, 4)
    R = np.asarray(mfa.responsibilities((X - d["mu"]) / d["sd"],
                                        splines.basis(coh.scan_age.values), p))
    g = R.argmax(1)
    small = np.argmin(np.bincount(g, minlength=2))
    return set(coh.dhcp_id.astype(str).values[g == small])


def main():
    sel = json.loads((paths.RESULTS / "selection.json").read_text())["selected"]
    assert sel["K"] == 1
    q = sel["q"]
    d = np.load(paths.RESULTS / "final" / f"K1_q{q}.npz")
    beta, W, psi, mu, sd = d["beta"][0], d["W"][0], d["psi"][0], d["mu"], d["sd"]
    C, ev_all, V = eig_axes(W, psi, q)
    ev = ev_all[:q]
    share = ev / np.trace(C)
    L = V * np.sqrt(ev)
    spread_sd = np.sqrt(np.diag(C))

    coh, Xraw = load()
    assert (coh.dhcp_id.astype(str).values == d["dhcp_id"]).all()
    t = coh.scan_age.values
    B = splines.basis(t)
    Z = (Xraw - mu) / sd
    R = Z - B @ beta
    scores = R @ V / np.sqrt(ev)
    lobe_z = R / spread_sd

    # --- bootstrap --------------------------------------------------------------
    files = sorted(glob.glob(str(SIG / "bootstrap" / "rep*.npz")))
    reps = [np.load(f) for f in files]
    conv = np.array([bool(r["converged"][0]) for r in reps])
    betas = np.array([r["beta"][0] for r in reps])
    Lb, shb, angles, block = [], [], [], []
    change_b = []
    BT = splines.basis(np.array([30.0, 42.0]))
    for r in reps:
        Cb, evb, Vb = eig_axes(r["W"][0], r["psi"][0], q)
        Vb, c = align(V, Vb)
        evb_m = evb[:q][c]
        Lb.append(Vb * np.sqrt(evb_m)); shb.append(evb_m / np.trace(Cb))
        angles.append(np.degrees(np.arccos(np.clip(np.abs((V * Vb).sum(0)), 0, 1))))
        k = min(3, q)
        s = np.linalg.svd(V[:, :k].T @ Vb[:, :k], compute_uv=False)
        block.append(np.degrees(np.arccos(np.clip(s.min(), 0, 1))))
        mb = BT @ r["beta"][0]
        change_b.append((mb[1] - mb[0]) / np.sqrt(np.diag(Cb)))
    Lb, shb, angles, change_b = map(np.array, (Lb, shb, angles, change_b))
    L_ci = np.percentile(Lb, [2.5, 97.5], axis=0)
    share_ci = np.percentile(shb, [2.5, 97.5], axis=0)

    # --- trajectories -------------------------------------------------------------
    BTT = splines.basis(T)
    traj = BTT @ beta * sd + mu
    traj_b = np.einsum("tm,bmd->btd", BTT, betas) * sd + mu
    band = np.percentile(traj_b, [2.5, 97.5], axis=0)
    m = BT @ beta
    change = (m[1] - m[0]) / spread_sd
    change_ci = np.percentile(change_b, [2.5, 97.5], axis=0)

    # --- permutation null ---------------------------------------------------------
    pf = sorted(glob.glob(str(SIG / "permutation" / "chunk*.npz")))
    perm = None
    if pf:
        P = [np.load(f) for f in pf]
        evp = np.concatenate([p["ev_c"] for p in P])
        null_share = evp / evp.sum(1, keepdims=True)
        n = len(null_share)
        perm = dict(n=n, converged=int(np.concatenate([p["converged"] for p in P]).sum()),
                    null_95th=np.percentile(null_share[:, :q], 95, axis=0).round(4).tolist(),
                    null_max=float(null_share[:, 0].max()),
                    p_rank=[(1 + int((null_share[:, j] >= share[j]).sum())) / (n + 1) for j in range(q)],
                    p_fwe=[(1 + int((null_share[:, 0] >= share[j]).sum())) / (n + 1) for j in range(q)])

    # --- split-half -----------------------------------------------------------------
    sf = sorted(glob.glob(str(SIG / f"split_half_q{q}" / "split*.npz")))
    split = None
    if sf:
        cos = []
        for f in sf:
            s = np.load(f)
            _, _, VA = eig_axes(s["W_A"], s["psi_A"], q)
            _, _, VB = eig_axes(s["W_B"], s["psi_B"], q)
            VA, _ = align(V, VA); VB, _ = align(V, VB)
            cos.append(np.abs((VA * VB).sum(0)))
        cos = np.array(cos)
        split = dict(n=len(sf), median=np.median(cos, 0).round(3).tolist(),
                     min=cos.min(0).round(3).tolist(), chance_95th=chance_cos(V).round(3).tolist())

    # --- does the spread change with age, or with prematurity? ------------------------
    bins = np.digitize(t, PMA_BINS[1:-1])
    pre = coh.preterm.values.astype(bool)
    late = t >= 37
    spread_age = {}
    for name, v in [(n, lobe_z[:, j]) for j, n in enumerate(GROUP_NAMES)] + \
                   [(f"axis {j + 1}", scores[:, j]) for j in range(q)]:
        groups = [v[bins == b] for b in range(len(PMA_BINS) - 1)]
        spread_age[name] = dict(sd_by_pma_bin=[round(float(g.std(ddof=1)), 3) for g in groups],
                                p_age=float(levene(*groups, center="median").pvalue),
                                sd_preterm_late=round(float(v[late & pre].std(ddof=1)), 3),
                                sd_term_late=round(float(v[late & ~pre].std(ddof=1)), 3),
                                p_preterm=float(levene(v[late & pre], v[late & ~pre],
                                                       center="median").pvalue))
    n_tests = 12 + q

    # --- artefacts and the low infants -------------------------------------------------
    motion = resid_on_age(t, coh.qc_dmri_eddy_rel_motion.values.astype(float))
    dens = resid_on_age(t, coh.density_unthresh.values.astype(float))
    low7 = set(pd.read_csv(paths.PROJECT / "results" / "bin10_no7" / "excluded_infants.csv")
               .dhcp_id.astype(str))
    small = small_group_ids()
    ids = coh.dhcp_id.astype(str).values
    in7, insmall = np.isin(ids, list(low7)), np.isin(ids, list(small))
    artefacts = dict(r_motion=[round(corr(scores[:, j], motion), 3) for j in range(q)],
                     r_density=[round(corr(scores[:, j], dens), 3) for j in range(q)],
                     mean_score_low7=(scores[in7].mean(0).round(2).tolist() if in7.any() else None),
                     mean_score_small24=(scores[insmall].mean(0).round(2).tolist()
                                         if insmall.any() else None))

    # --- positions for all 670 infants (steps 4-6) -------------------------------------
    allc = clean_covariates(pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv"))
    feat = np.load(paths.RESULTS / "features" / f"{paths.FEATURE_SET}.npz")["X"]
    prim = allc.in_primary.values.astype(bool)
    ac, Xa = allc[prim].reset_index(drop=True), feat[prim]
    Ra = (Xa - mu) / sd - splines.basis(ac.scan_age.values) @ beta
    pos = pd.DataFrame(dict(dhcp_id=ac.dhcp_id.astype(str), split=ac.split))
    for j in range(q):
        pos[f"axis{j + 1}"] = Ra @ V[:, j] / np.sqrt(ev[j])
    for j, n in enumerate(GROUP_NAMES):
        pos[f"z_{n}"] = Ra[:, j] / spread_sd[j]
    pos["low7"] = pos.dhcp_id.isin(low7)
    pos["small24"] = np.where(pos.split == "train", pos.dhcp_id.isin(small), np.nan) \
        if small else np.nan
    pos.to_csv(paths.RESULTS / "positions.csv", index=False)

    out = dict(run=paths.RUN, q=q, n_train=len(coh), lobes=GROUP_NAMES.tolist(),
               share=share.round(4).tolist(), share_ci=share_ci.T.round(4).tolist(),
               share_beyond_q=float(1 - share.sum()),
               loadings=L.round(3).tolist(), loadings_ci=np.moveaxis(L_ci, 0, -1).round(3).tolist(),
               loadings_exclude_zero=((L_ci[0] > 0) | (L_ci[1] < 0)).sum(0).tolist(),
               bootstrap=dict(n=len(reps), converged=int(conv.sum()),
                              angle_median=np.median(angles, 0).round(1).tolist(),
                              angle_95th=np.percentile(angles, 95, 0).round(1).tolist(),
                              block_1_3_max_angle_median=float(np.median(block)),
                              block_1_3_max_angle_95th=float(np.percentile(block, 95))),
               spread_sd_std_units=spread_sd.round(3).tolist(),
               shared_fraction=(np.diag(W @ W.T) / np.diag(C)).round(3).tolist(),
               change_30_42_spread_sd=change.round(3).tolist(),
               change_ci=change_ci.T.round(3).tolist(),
               change_excludes_zero=[bool(lo > 0 or hi < 0) for lo, hi in change_ci.T],
               value_30=(traj[np.searchsorted(T, 30)]).round(5).tolist(),
               value_42=(traj[np.searchsorted(T, 42)]).round(5).tolist(),
               permutation=perm, split_half=split,
               spread_by_age=dict(pma_bins=list(PMA_BINS), bonferroni_n=n_tests, tests=spread_age),
               artefacts=artefacts)
    (paths.RESULTS / "step3_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("loadings", "loadings_ci")}, indent=1))

    fs.apply_style()
    fig_trajectories(t, Xraw, traj, band, change, change_ci, len(reps))
    fig_standardised(beta, spread_sd)
    fig_spread(np.diag(W @ W.T), psi, spread_age, n_tests)
    fig_axes(L, L_ci, share, share_ci, perm, q, len(reps))
    fig_axes_brain(L, L_ci, share, share_ci, perm, len(reps))
    fig_significance(share, perm, split, angles, q)


def fig_trajectories(t, X, traj, band, change, change_ci, n_boot):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True)
    fig.subplots_adjust(hspace=0.38, wspace=0.3, bottom=0.14)
    for k, (lobe, ax) in enumerate(zip(atlas.LOBES, axes.flat)):
        lines = []
        for h, ls in (("L", "-"), ("R", "--")):
            g = list(GROUP_NAMES).index(f"{lobe}_{h}")
            ax.scatter(t, X[:, g], s=4, color=fs.NEUTRAL_GREY, alpha=0.15, linewidths=0, zorder=1)
            ax.fill_between(T, *band[:, :, g], color=LOBE_COLOURS[lobe], alpha=0.25, lw=0, zorder=2)
            ax.plot(T, traj[:, g], color=LOBE_COLOURS[lobe], ls=ls, lw=2, zorder=3)
            lo, hi = change_ci[:, g]
            lines.append(f"{h}: {change[g]:+.2f} ({lo:+.2f} to {hi:+.2f})")
        ax.text(0.03, 0.97, "30→42 wk, SDs of spread\n" + "\n".join(lines), transform=ax.transAxes,
                va="top", fontsize=9, color=fs.TEXT_SECONDARY)
        lo_, hi_ = np.nanpercentile(X[:, [list(GROUP_NAMES).index(f"{lobe}_{h}") for h in "LR"]], [1, 99])
        ax.set_ylim(lo_ - 0.35 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
        fs.panel_title(ax, "ABCDEF"[k], lobe)
        if k >= 3: ax.set_xlabel("PMA (weeks)")
        if k % 3 == 0: ax.set_ylabel("communicability")
    fig.legend(handles=[Line2D([], [], color=fs.TEXT_PRIMARY, ls="-", lw=2, label="left"),
                        Line2D([], [], color=fs.TEXT_PRIMARY, ls="--", lw=2, label="right")],
               loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2)
    fig.text(0.07, 0.0, f"Selected one-group model, {len(t)} training infants; communicability "
             f"averaged over 8-12% density. Shaded: pointwise 95% band from {n_boot:,} parametric\n"
             "bootstrap replicates. Grey: each infant's lobe value. Each panel has its own y-scale "
             "(1st-99th percentile of the infants).", fontsize=fs.NOTE_SIZE,
             color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "b6_trajectories")


def fig_standardised(beta, spread_sd):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    m = splines.basis(T) @ beta
    m40 = splines.basis(np.array([40.0])) @ beta
    for g, n in enumerate(GROUP_NAMES):
        ax.plot(T, (m[:, g] - m40[0, g]) / spread_sd[g], color=COL[g],
                ls="-" if n.endswith("L") else "--", lw=2)
    fs.reference_line(ax, 0)
    ax.set_xlabel("PMA (weeks)"); ax.set_ylabel("difference from 40 weeks\n(SDs of the spread)")
    ax.legend(handles=[Patch(color=c, label=l) for l, c in LOBE_COLOURS.items()] +
              [Line2D([], [], color=fs.TEXT_PRIMARY, ls="-", label="left"),
               Line2D([], [], color=fs.TEXT_PRIMARY, ls="--", label="right")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fs.save_figure(fig, paths.FIGURES / "b6b_trajectories_standardised")


def fig_spread(shared, psi, spread_age, n_tests):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(wspace=0.28, bottom=0.3)
    ax = axes[0]
    ax.bar(XPOS, shared, color=COL, edgecolor=fs.TEXT_PRIMARY, lw=0.6, width=0.8)
    ax.bar(XPOS, psi, bottom=shared, color=COL, alpha=0.35, edgecolor=fs.TEXT_PRIMARY, lw=0.6,
           width=0.8, hatch="//")
    ax.set_xticks(XPOS, SHORT, rotation=60, ha="right")
    ax.set_ylabel("variance beyond age (standardised)")
    fs.panel_title(ax, "A", "Size of the spread")
    ax.legend(handles=[Patch(facecolor=fs.NEUTRAL_GREY, edgecolor=fs.TEXT_PRIMARY, label="shared (axes)"),
                       Patch(facecolor=fs.NEUTRAL_GREY, alpha=0.35, hatch="//",
                             edgecolor=fs.TEXT_PRIMARY, label="lobe-specific")],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax = axes[1]
    mids = ["< 35", "35-38", "38-41", "≥ 41"]
    x = np.arange(len(mids))
    for g, n in enumerate(GROUP_NAMES):
        s = spread_age[n]
        ax.plot(x, s["sd_by_pma_bin"], color=COL[g], ls="-" if n.endswith("L") else "--",
                marker="o", ms=5, lw=1.6)
    fs.reference_line(ax, 1.0)
    ax.set_xticks(x, mids); ax.set_xlabel("PMA at scan (weeks)")
    ax.set_ylabel("SD around the curve (1 = average)")
    sig = [n.replace("_", " ") for n in GROUP_NAMES if spread_age[n]["p_age"] < 0.05 / n_tests]
    ax.text(0.98, 0.97, "spread changes with age (Bonferroni):\n" + (", ".join(sig) if sig else "none"),
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=fs.TEXT_SECONDARY)
    fs.panel_title(ax, "B", "Does the spread change with age?")
    ax.legend(handles=[Patch(color=c, label=l) for l, c in LOBE_COLOURS.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fs.save_figure(fig, paths.FIGURES / "b7_spread")


def fig_axes(L, L_ci, share, share_ci, perm, q, n_boot):
    ncol = 2
    nrow = int(np.ceil(q / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.9 * nrow), sharey=True)
    fig.subplots_adjust(hspace=0.75, wspace=0.12)
    lim = np.nanmax(np.abs(L_ci)) * 1.1
    for j, ax in enumerate(np.ravel(axes)):
        if j >= q:
            ax.axis("off"); continue
        ax.set_xlim(XPOS[0] - 0.7, XPOS[-1] + 0.7); ax.set_ylim(-lim, lim)
        excl = (L_ci[0, :, j] > 0) | (L_ci[1, :, j] < 0)
        fs.rounded_bars(ax, XPOS, L[:, j], width=0.8,
                        color=[c if e else fs.NEUTRAL_GREY for c, e in zip(COL, excl)])
        ax.errorbar(XPOS, L[:, j], yerr=[L[:, j] - L_ci[0, :, j], L_ci[1, :, j] - L[:, j]],
                    fmt="none", ecolor=fs.TEXT_PRIMARY, elinewidth=0.8, capsize=2, zorder=4)
        fs.reference_line(ax, 0)
        ax.set_xticks(XPOS, SHORT, rotation=60, ha="right", fontsize=9)
        p = f"\npermutation p = {perm['p_fwe'][j]:.3f} (corrected)" if perm else ""
        ax.set_title(f"Axis {j + 1}: {100 * share[j]:.1f}% of the spread "
                     f"({100 * share_ci[0, j]:.1f}-{100 * share_ci[1, j]:.1f}){p}",
                     fontsize=12, pad=8, loc="left")
        if j % ncol == 0: ax.set_ylabel("loading (SD)")
    fig.text(0.07, 0.0, f"Axes = eigenvectors of the spread C; loading = covariance of each lobe "
             f"value (standardised) with the axis score. Error bars: 95% intervals from {n_boot:,}\n"
             "bootstrap replicates; grey bars: interval includes 0. Shares in brackets: bootstrap "
             "95% intervals (biased upward for small axes).", fontsize=fs.NOTE_SIZE,
             color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "b8b_all_axes")


def fig_axes_brain(L, L_ci, share, share_ci, perm, n_boot):
    """b8: one row per axis above noise (corrected permutation p < 0.05): the 12
    lobe loadings as bars, and the same loadings painted on the brain, every
    region taking its lobe x hemisphere group's loading (12 distinct values).
    Surfaces and the deep slice are drawn as in fig5 (make_brain_figures.py),
    from the cached template mesh."""
    from matplotlib.colors import Normalize
    from scripts import make_brain_figures as mb
    from scripts.build_bin10 import GROUP_OF

    keep = [j for j in range(L.shape[1]) if perm and perm["p_fwe"][j] < 0.05]
    lab, t2, mask, aff = mb.load_display_atlas()
    mb.check_same_numbering(lab, aff)
    z = np.load(paths.PROJECT / "results" / "final" / "brain_mesh.npz")   # template mesh, run-independent
    meshes = {h: (z[f"v{h}"], z[f"f{h}"], z[f"l{h}"]) for h in "LR"}
    k_deep = int(round(np.argwhere(np.isin(lab, [77, 78]))[:, 2].mean()))   # thalamus level

    vmax = float(np.abs(L[:, keep]).max())
    cmap, norm = fs.DIVERGING, Normalize(-vmax, vmax)
    views = list(mb.VIEWS) + ["deep (axial)"]
    n = len(keep)
    H = 3.0 * n + 1.4
    fig = plt.figure(figsize=(18, H))
    gs = fig.add_gridspec(n, 1 + len(views), width_ratios=[2.1, 1, 1, 1, 1, 0.75],
                          left=0.05, right=0.99, top=1 - 0.6 / H,
                          bottom=1.15 / H, wspace=0.03, hspace=0.6)
    for r, j in enumerate(keep):
        excl = (L_ci[0, :, j] > 0) | (L_ci[1, :, j] < 0)
        ax = fig.add_subplot(gs[r, 0])
        ax.set_xlim(XPOS[0] - 0.7, XPOS[-1] + 0.7); ax.set_ylim(-1, 1)
        fs.rounded_bars(ax, XPOS, L[:, j], width=0.8,
                        color=[c if e else fs.NEUTRAL_GREY for c, e in zip(COL, excl)])
        ax.errorbar(XPOS, L[:, j], yerr=[L[:, j] - L_ci[0, :, j], L_ci[1, :, j] - L[:, j]],
                    fmt="none", ecolor=fs.TEXT_PRIMARY, elinewidth=0.8, capsize=2, zorder=4)
        fs.reference_line(ax, 0)
        ax.set_xticks(XPOS, SHORT, rotation=60, ha="right", fontsize=9)
        ax.set_ylabel("loading (SD)")
        ax.set_title(f"Axis {j + 1}: {100 * share[j]:.1f}% of the spread "
                     f"({100 * share_ci[0, j]:.1f}-{100 * share_ci[1, j]:.1f}), "
                     f"p = {perm['p_fwe'][j]:.3f}", fontsize=12, loc="left", pad=8)
        group_col = mb.region_colours(L[:, j], cmap, norm, excl)     # (12, 3)
        colours = group_col[GROUP_OF]                                 # (90, 3), AAL order
        for c, view in enumerate(views):
            bx = fig.add_subplot(gs[r, 1 + c])
            if view == "deep (axial)":
                mb.draw_deep_slice(bx, lab, t2, colours, k_deep)
            else:
                mb.draw_surface(bx, meshes, colours, view)
            if r == 0:
                pos = gs[0, 1 + c].get_position(fig)
                fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + 0.1 / H, view, ha="center",
                         va="bottom", fontsize=12, fontweight="bold")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fs.horizontal_colorbar(fig, sm, [0.45, 0.6 / H, 0.3, 0.1 / H],
                           "loading (SD): red = higher in infants high on the axis")
    fig.text(0.05, 0.0, f"Axes above noise only (p: corrected permutation p < 0.05); all axes in b8b. "
             f"Each region takes its lobe's loading. Error bars and light grey: 95% interval "
             f"from {n_boot:,} bootstrap replicates\nincludes 0. Dark grey: tissue outside the 90 "
             "AAL regions. Drawn on the UNC neonatal template; the deep slice is at the level of "
             "the thalamus.", fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "b8_axes")


def fig_significance(share, perm, split, angles, q):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(wspace=0.32, bottom=0.25)
    x = np.arange(1, q + 1)
    ax = axes[0]
    ax.set_xlim(0.4, q + 0.6); ax.set_ylim(0, max(share) * 1.25)
    fs.rounded_bars(ax, x, share, color=fs.COLORS["blue"])
    if perm:
        ax.plot(x, perm["null_95th"], color=fs.COLORS["vermillion"], marker="s", lw=1.5,
                label="null, 95th percentile")
        ax.legend(loc="upper right")
    ax.set_xlabel("axis"); ax.set_ylabel("share of the spread")
    fs.panel_title(ax, "A", "Bigger than noise?")
    ax = axes[1]
    if split:
        ax.plot(x, split["median"], color=fs.TEXT_PRIMARY, marker="D", lw=2, label="median of splits")
        ax.plot(x, split["min"], color=fs.NEUTRAL_GREY, marker="o", ls="--", label="worst split")
        ax.plot(x, split["chance_95th"], color=fs.COLORS["vermillion"], ls=":", lw=1.5,
                label="chance, 95th pct")
        ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0, 1.02); ax.set_xticks(x)
    ax.set_xlabel("axis"); ax.set_ylabel("|cos| between halves")
    fs.panel_title(ax, "B", "Same axes in each half?")
    ax = axes[2]
    ax.plot(x, np.median(angles, 0), color=fs.TEXT_PRIMARY, marker="D", lw=2, label="median")
    ax.plot(x, np.percentile(angles, 95, 0), color=fs.NEUTRAL_GREY, marker="o", ls="--",
            label="95th percentile")
    ax.set_xticks(x); ax.set_ylim(0, 90)
    ax.set_xlabel("axis"); ax.set_ylabel("angle to the fitted axis (degrees)")
    ax.legend(loc="upper left")
    fs.panel_title(ax, "C", "How stable is each axis?")
    fs.save_figure(fig, paths.FIGURES / "b9_axis_significance")


if __name__ == "__main__":
    main()
