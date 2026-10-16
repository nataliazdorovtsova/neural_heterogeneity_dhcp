"""Figures 3-4: what the within-population axes of variation are.

    python scripts/make_axes_figures.py

Reads the fitted one-group models from scripts/fit_final.py
(results/final/K1_q{q}.npz). The primary model is q = 15; q = 5 and q = 25 are
used only to check whether the leading axes stay the same as q changes.

Axes are a set: any rotation of them fits equally well. To give them an order,
they are rotated to principal axes (largest variance first; the same rotation
PCA uses) and each is signed so that its loadings sum positive. A higher score
then means "more of these measures overall".

fig3_axes_overview       size, content, correlates and stability of the axes
fig4_axis_maps           brain maps of the three largest axes
results/final/axes_K1_q{q}.csv   every region's loadings, for the tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import paths
import atlas
import figstyle as fs
from features import MEASURES
from scripts.run_spec import load

FINAL = paths.RESULTS / "final"
OUT = paths.FIGURES
PRIMARY_Q = 15
COMPARE_Q = (5, 25)
N_SHOW = 6                                    # axes shown in the content panels
MEASURE_LABEL = {"strength": "strength", "log_clustering": "clustering",
                 "communicability": "communicability"}

COVARIATES = {                                # label -> cohort column (or derived)
    "GA at birth": "birth_age",
    "PMA at scan": "scan_age",
    "age since birth": "pna",
    "female": "female",
    "head motion": "qc_dmri_eddy_rel_motion",
    "network density": "density_thr",
    "radiology score": "radiology_score",
}


def principal_axes(q):
    f = FINAL / f"K1_q{q}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    W, psi = d["W"][0], d["psi"][0]              # K = 1: (270, q), (270,)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    L = U * S                                    # loadings, largest axis first
    scores = d["scores"][:, 0, :] @ Vt.T         # W theta = (U S)(Vt theta)
    sign = np.where(L.sum(0) < 0, -1.0, 1.0)
    total = (S ** 2).sum() + psi.sum()
    return dict(q=q, L=L * sign, scores=scores * sign, share=S ** 2 / total,
                noise_share=psi.sum() / total, ids=d["dhcp_id"])


def lobe_table(L, n):
    """Mean loading per (measure, lobe) for the first n axes: (n, 3 * 6)."""
    R = L.reshape(90, 3, -1)
    cols, vals = [], []
    for m, meas in enumerate(MEASURES):
        for lobe in atlas.LOBES:
            cols.append((MEASURE_LABEL[meas], lobe))
            vals.append(R[atlas.LOBE == lobe, m, :n].mean(0))
    return np.array(vals).T, cols


def correlates(ax_scores, coh, n):
    df = coh.assign(female=(coh.sex == "female").astype(float))
    r = np.full((n, len(COVARIATES)), np.nan)
    for c, col in enumerate(COVARIATES.values()):
        v = df[col].values.astype(float)
        ok = np.isfinite(v)
        for j in range(n):
            r[j, c] = np.corrcoef(ax_scores[ok, j], v[ok])[0, 1]
    # Bonferroni over every cell shown, at the smallest n used.
    n_obs = int(min(np.isfinite(df[c].values.astype(float)).sum() for c in COVARIATES.values()))
    alpha = 0.05 / r.size
    t = stats.t.ppf(1 - alpha / 2, n_obs - 2)
    r_crit = t / np.sqrt(n_obs - 2 + t ** 2)
    return r, r_crit


def best_match(La, Lb, n):
    """For each of La's first n axes, |cosine| with its best match among Lb's axes."""
    A = La[:, :n] / np.linalg.norm(La[:, :n], axis=0)
    B = Lb / np.linalg.norm(Lb, axis=0)
    return np.abs(A.T @ B).max(1)


def ink_on(rgba):
    """Text colour that stays legible on a filled cell."""
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if lum < 0.5 else fs.TEXT_PRIMARY


def _heatmap_frame(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig_overview(P, others, coh):
    fig, axes = plt.subplots(2, 2, figsize=fs.FIGURE_SIZES["full_page"],
                             gridspec_kw=dict(width_ratios=[1, 1.2]))
    fig.subplots_adjust(wspace=0.38, hspace=0.62, left=0.09, right=0.93, top=0.94, bottom=0.15)
    q = P["q"]

    # A: how much variation each axis carries
    ax = axes[0, 0]
    x = np.arange(1, q + 1)
    ax.set_xlim(0.3, q + 0.7); ax.set_ylim(0, P["share"].max() * 100 * 1.45)
    fs.rounded_bars(ax, x[:N_SHOW], P["share"][:N_SHOW] * 100, color=fs.Q_COLOURS[q],
                    value_fmt="{:.1f}")
    fs.rounded_bars(ax, x[N_SHOW:], P["share"][N_SHOW:] * 100, color=fs.Q_COLOURS[q])
    cum = np.cumsum(P["share"]) * 100
    ax.annotate(f"top 3: {cum[2]:.0f}%  ·  top {N_SHOW}: {cum[N_SHOW - 1]:.0f}%  ·  "
                f"all {q}: {cum[-1]:.0f}%\nregion-specific: {P['noise_share'] * 100:.0f}%",
                (0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
                fontsize=fs.NOTE_SIZE)
    ax.set_xticks(x[::2])
    ax.set_xlabel("axis (largest first)"); ax.set_ylabel("% of variation")
    fs.panel_title(ax, "A", "Size of each axis")

    # B: what each axis is made of
    ax = axes[0, 1]
    T, cols = lobe_table(P["L"], N_SHOW)
    vmax = np.abs(T).max()
    im = ax.imshow(T, cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(N_SHOW), [f"axis {j + 1}" for j in range(N_SHOW)])
    ax.set_xticks(range(len(cols)), [c[1] for c in cols], rotation=90)
    for b in (5.5, 11.5):
        ax.axvline(b, color="white", lw=3)
    for m, meas in enumerate(MEASURES):
        ax.text(m * 6 + 2.5, -0.7, MEASURE_LABEL[meas], ha="center", va="bottom",
                fontsize=fs.NOTE_SIZE)
    _heatmap_frame(ax)
    ax.set_title("B   Content of each axis", pad=30)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_linewidth(0.6)
    cb.set_label("mean loading in lobe", fontsize=fs.NOTE_SIZE)

    # C: whether the axes stay the same as q changes
    ax = axes[1, 0]
    n = min(8, q)
    styles = {25: ("-", "s"), 5: ("--", "D")}
    for o in others:
        ls, m = styles.get(o["q"], (":", "o"))
        nn = min(n, o["q"])
        y = best_match(P["L"], o["L"], nn)
        ax.plot(np.arange(1, nn + 1), y, color=fs.Q_COLOURS[o["q"]], ls=ls, marker=m,
                markersize=6, markeredgecolor="white", markeredgewidth=0.5)
    fs.reference_line(ax, 0.9)
    ax.set_ylim(0, 1.05); ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel(f"axis of the q = {q} model"); ax.set_ylabel("similarity (|cos|)")
    fs.panel_title(ax, "C", "Stability across q")

    # D: what each axis relates to
    ax = axes[1, 1]
    r, r_crit = correlates(P["scores"], coh, N_SHOW)
    rmax = max(0.3, np.nanmax(np.abs(r)))
    im = ax.imshow(r, cmap=fs.DIVERGING, vmin=-rmax, vmax=rmax, aspect="auto")
    for j in range(r.shape[0]):
        for c in range(r.shape[1]):
            v = r[j, c]
            ax.text(c, j, f"{v:+.2f}", ha="center", va="center", fontsize=fs.VALUE_SIZE,
                    color=ink_on(fs.DIVERGING((v + rmax) / (2 * rmax))),
                    fontweight="bold" if abs(v) >= r_crit else "normal")
    ax.set_yticks(range(N_SHOW), [f"axis {j + 1}" for j in range(N_SHOW)])
    ax.set_xticks(range(len(COVARIATES)), list(COVARIATES), rotation=40, ha="right")
    _heatmap_frame(ax)
    fs.panel_title(ax, "D", "Correlates of each axis")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_linewidth(0.6)
    cb.set_label("correlation r", fontsize=fs.NOTE_SIZE)

    # shared legend for the q encodings in A and C
    handles = []
    for o in others:
        ls, m = styles.get(o["q"], (":", "o"))
        handles.append(Line2D([], [], color=fs.Q_COLOURS[o["q"]], ls=ls, marker=m,
                              label=f"C: compared with q = {o['q']}"))
    axes[1, 0].legend(handles=handles, loc="lower left", handlelength=2.2)
    fig.text(0.06, 0.035, f"One-group model (K = 1), q = {q}, all {len(coh)} training infants; "
             "axes ordered by size and signed so higher = more of the measures.\n"
             f"D: bold = |r| ≥ {r_crit:.2f} (Bonferroni, {r.size} tests); age at scan is "
             "modelled by the trajectories, so its column is ~0.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    return fs.save_figure(fig, OUT / "fig3_axes_overview")


def fig_maps(P, n_axes=3):
    from nibabel.affines import apply_affine
    lab, aff = atlas.load_atlas()
    cen = atlas.centroids(lab, aff)
    proj = (lab > 0).any(axis=2).astype(float)
    ii, jj = np.meshgrid(np.arange(lab.shape[0]), np.arange(lab.shape[1]), indexing="ij")
    w = apply_affine(aff, np.stack([ii, jj, np.full_like(ii, lab.shape[2] // 2)], -1))
    xw, yw = w[..., 0], w[..., 1]

    R = P["L"].reshape(90, 3, -1)
    vmax = np.abs(R[:, :, :n_axes]).max()
    fig, axes = plt.subplots(n_axes, 3, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.12, right=0.99, top=0.93, bottom=0.12)
    for j in range(n_axes):
        for m, meas in enumerate(MEASURES):
            ax = axes[j, m]
            ax.contourf(xw, yw, proj, levels=[0.5, 1.5], colors=[fs.NEUTRAL_GREY], alpha=0.25)
            ax.contour(xw, yw, proj, levels=[0.5], colors=[fs.NEUTRAL_GREY], linewidths=0.8)
            v = R[:, m, j]
            order = np.argsort(np.abs(v))           # strongest drawn on top
            sc = ax.scatter(cen[order, 0], cen[order, 1], c=v[order], cmap=fs.DIVERGING,
                            vmin=-vmax, vmax=vmax, s=16 + 80 * np.abs(v[order]) / vmax,
                            edgecolors=fs.TEXT_MUTED, linewidths=0.4, zorder=2)
            ax.set_aspect("equal"); ax.axis("off")
            if j == 0:
                ax.set_title(MEASURE_LABEL[meas], loc="center", fontsize=14)
            if m == 0:
                ax.text(-0.04, 0.5, f"axis {j + 1}\n{P['share'][j] * 100:.0f}%",
                        transform=ax.transAxes, ha="right", va="center", fontsize=14,
                        fontweight="bold")
    axes[0, 0].text(0.02, 0.02, "L", transform=axes[0, 0].transAxes, fontsize=fs.NOTE_SIZE)
    axes[0, 0].text(0.98, 0.02, "R", transform=axes[0, 0].transAxes, fontsize=fs.NOTE_SIZE,
                    ha="right")
    fs.horizontal_colorbar(fig, sc, [0.33, 0.085, 0.4, 0.014],
                           "loading (SD per SD of axis score)")
    fig.text(0.12, 0.012, "View from above, anterior up, regions at their centroids.\n"
             "Red = higher in infants scoring high on the axis, blue = lower; "
             "dot size = |loading|.", fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY)
    return fs.save_figure(fig, OUT / "fig4_axis_maps")


def main():
    atlas.check_node_index()
    P = principal_axes(PRIMARY_Q)
    if P is None:
        raise SystemExit(f"no fitted model at {FINAL}/K1_q{PRIMARY_Q}.npz — run fit_final.py")
    others = [o for o in (principal_axes(q) for q in COMPARE_Q) if o is not None]
    coh, _ = load()
    assert (coh.dhcp_id.astype(str).values == P["ids"]).all(), "infant order mismatch"

    rows = []
    for r in range(90):
        for m, meas in enumerate(MEASURES):
            rows.append(dict(region=atlas.NAMES[r], lobe=atlas.LOBE[r], measure=meas,
                             **{f"axis{j + 1}": P["L"][3 * r + m, j] for j in range(PRIMARY_Q)}))
    pd.DataFrame(rows).to_csv(FINAL / f"axes_K1_q{PRIMARY_Q}.csv", index=False)

    fs.apply_style()
    print("variance share per axis (%):", np.round(P["share"] * 100, 1))
    print(f"noise share {P['noise_share'] * 100:.1f}%")
    for j in range(N_SHOW):
        v = P["L"][:, j]
        top = np.argsort(-np.abs(v))[:6]
        print(f"axis {j + 1}: " + ", ".join(
            f"{atlas.NAMES[d // 3]}/{MEASURE_LABEL[MEASURES[d % 3]]} {v[d]:+.2f}" for d in top))
    for o in others:
        print(f"best-match |cos| vs q={o['q']}:",
              np.round(best_match(P["L"], o["L"], min(8, PRIMARY_Q, o["q"])), 2))
    for fn in (lambda: fig_overview(P, others, coh), lambda: fig_maps(P)):
        print("wrote " + ", ".join(str(x) for x in fn()))


if __name__ == "__main__":
    main()
