"""Figures 3-4: what the within-population axes of variation are.

    python scripts/make_axes_figures.py

Reads the fitted one-group models from scripts/fit_final.py
(results/final/K1_q{q}.npz). The primary model is q = 15; q = 5 and q = 25 are
used only to check whether the leading axes stay the same as q changes.

Axes are a set: any rotation of them fits equally well. To give them an order,
they are rotated to principal axes (largest variance first; the same rotation
PCA uses) and each is signed so that its loadings sum positive. A higher score
then means "more of these measures overall".

fig3_axes_overview.png   size, content, correlates and stability of the axes
fig4_axis_maps.png       brain maps of the three largest axes
results/final/axes_K1_q{q}.csv   every region's loadings, for the tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    """Text token that stays legible on a filled cell."""
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return fs.SURFACE if lum < 0.5 else fs.INK


def fig_overview(P, others, coh):
    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.62, left=0.07, right=0.97,
                          top=0.87, bottom=0.1, width_ratios=[1, 1.25])
    q = P["q"]

    # A: how big is each axis?
    ax = fig.add_subplot(gs[0, 0]); fs.tidy(ax)
    x = np.arange(1, q + 1)
    ax.bar(x, P["share"] * 100, color=fs.SERIES[0], width=0.72, zorder=2,
           label=f"q = {q} (primary)")
    marks = {5: ("D", fs.SERIES[2]), 25: ("o", fs.SERIES[1])}
    for o in others:
        m, col = marks.get(o["q"], ("s", fs.NEUTRAL))
        k = min(o["q"], q)
        ax.scatter(np.arange(1, k + 1), o["share"][:k] * 100, marker=m, s=22, zorder=3,
                   facecolors=fs.SURFACE, edgecolors=col, linewidths=1.3,
                   label=f"q = {o['q']}")
    cum = np.cumsum(P["share"]) * 100
    ax.annotate(f"top 3 axes: {cum[2]:.0f}%   top {N_SHOW}: {cum[N_SHOW - 1]:.0f}%   "
                f"all {q}: {cum[-1]:.0f}%\nregion-specific noise: {P['noise_share'] * 100:.0f}%",
                (0.98, 0.96), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, color=fs.INK_2)
    ax.set_xticks(x)
    ax.set_xlabel("axis (largest first)")
    ax.set_ylabel("% of variation beyond age")
    ax.legend(loc="center right", fontsize=7.5)
    fs.panel_letter(ax, "A", "How much variation does each axis carry?")

    # B: what is each axis made of?
    ax = fig.add_subplot(gs[0, 1])
    T, cols = lobe_table(P["L"], N_SHOW)
    vmax = np.abs(T).max()
    im = ax.imshow(T, cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(N_SHOW), [f"axis {j + 1}" for j in range(N_SHOW)])
    ax.set_xticks(range(len(cols)), [c[1] for c in cols], rotation=60, ha="right",
                  fontsize=7)
    for b in (5.5, 11.5):
        ax.axvline(b, color=fs.SURFACE, lw=3)
    for m, meas in enumerate(MEASURES):
        ax.text(m * 6 + 2.5, -0.85, MEASURE_LABEL[meas], ha="center", va="bottom",
                fontsize=8, color=fs.INK_2)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width * 0.25, pos.y0 - 0.115, pos.width * 0.5, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=fs.INK_MUTED, length=0)
    cb.set_label("mean loading in lobe (SD per SD of axis score)", fontsize=7.5,
                 color=fs.INK_2)
    fs.panel_letter(ax, "B", "What is each axis made of?")
    ax.texts[-1].set_y(1.17); ax.texts[-2].set_y(1.17)

    # D: what does each axis relate to? (bottom right)
    ax = fig.add_subplot(gs[1, 1])
    r, r_crit = correlates(P["scores"], coh, N_SHOW)
    rmax = max(0.3, np.nanmax(np.abs(r)))
    im = ax.imshow(r, cmap=fs.DIVERGING, vmin=-rmax, vmax=rmax, aspect="auto")
    for j in range(r.shape[0]):
        for c in range(r.shape[1]):
            v = r[j, c]
            ax.text(c, j, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                    color=ink_on(fs.DIVERGING((v + rmax) / (2 * rmax))),
                    fontweight="bold" if abs(v) >= r_crit else "normal")
    ax.set_yticks(range(N_SHOW), [f"axis {j + 1}" for j in range(N_SHOW)])
    ax.set_xticks(range(len(COVARIATES)), list(COVARIATES), rotation=30, ha="right",
                  fontsize=7.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(0.0, -0.36, f"correlation of infants' axis scores with each variable; bold = "
            f"|r| ≥ {r_crit:.2f} (Bonferroni over {r.size} tests). Age at scan is already "
            "modelled by the trajectories, so its column should be ~0.",
            transform=ax.transAxes, fontsize=7, color=fs.INK_2, va="top", wrap=True)
    fs.panel_letter(ax, "D", "What does each axis relate to?")

    # C: are the axes the same when q changes? (bottom left)
    ax = fig.add_subplot(gs[1, 0]); fs.tidy(ax)
    n = min(8, q)
    for o in others:
        m, col = marks.get(o["q"], ("s", fs.NEUTRAL))
        nn = min(n, o["q"])
        y = best_match(P["L"], o["L"], nn)
        ax.plot(np.arange(1, nn + 1), y, color=col, lw=1.6, zorder=2)
        ax.scatter(np.arange(1, nn + 1), y, marker=m, s=26, color=col, zorder=3,
                   edgecolors=fs.SURFACE, linewidths=0.5)
        ax.annotate(f"vs q = {o['q']}", (nn, y[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, color=fs.INK_2, va="center")
    ax.axhline(0.9, color=fs.INK_MUTED, lw=1, ls=(0, (4, 2)))
    ax.annotate("≥ 0.9: essentially the same pattern", (0.02, 0.9),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", fontsize=7.5, color=fs.INK_2)
    ax.set_ylim(0, 1.05); ax.set_xlim(0.5, n + 1.6)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel(f"axis of the q = {q} model")
    ax.set_ylabel("similarity to best-matching axis (|cos|)")
    fs.panel_letter(ax, "C", "Do the axes stay the same as q changes?")

    fig.suptitle("Within-population axes of variation: which ones matter?",
                 x=0.07, y=0.975, ha="left", fontsize=11, color=fs.INK)
    fig.text(0.07, 0.94, f"one-group model (K = 1), q = {q}, fitted on all {len(coh)} "
             "training infants · axes ordered by size and signed so higher = more of the "
             "measures overall", ha="left", fontsize=8, color=fs.INK_2)
    p = OUT / "fig3_axes_overview.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


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
    fig, axes = plt.subplots(n_axes, 3, figsize=(9.6, 3.3 * n_axes))
    fig.subplots_adjust(wspace=0.04, hspace=0.18, left=0.13, right=0.98, top=0.9, bottom=0.1)
    for j in range(n_axes):
        for m, meas in enumerate(MEASURES):
            ax = axes[j, m]
            ax.contourf(xw, yw, proj, levels=[0.5, 1.5], colors=[fs.GRID], zorder=0)
            ax.contour(xw, yw, proj, levels=[0.5], colors=[fs.PANEL_EDGE], linewidths=0.8)
            v = R[:, m, j]
            order = np.argsort(np.abs(v))           # strongest drawn on top
            sc = ax.scatter(cen[order, 0], cen[order, 1], c=v[order], cmap=fs.DIVERGING,
                            vmin=-vmax, vmax=vmax, s=14 + 70 * np.abs(v[order]) / vmax,
                            edgecolors=fs.NEUTRAL, linewidths=0.4, zorder=2)
            ax.set_aspect("equal"); ax.axis("off")
            if j == 0:
                ax.set_title(MEASURE_LABEL[meas], loc="center", fontsize=9, color=fs.INK_2)
            if m == 0:
                ax.text(-0.06, 0.5, f"axis {j + 1}\n{P['share'][j] * 100:.0f}% of\nvariation",
                        transform=ax.transAxes, ha="right", va="center", fontsize=8.5,
                        color=fs.INK)
            if j == 0 and m == 0:
                ax.text(0.02, 0.02, "L", transform=ax.transAxes, fontsize=7.5, color=fs.INK_MUTED)
                ax.text(0.98, 0.02, "R", transform=ax.transAxes, fontsize=7.5,
                        color=fs.INK_MUTED, ha="right")
                ax.text(0.5, 0.99, "anterior", transform=ax.transAxes, fontsize=7,
                        color=fs.INK_MUTED, ha="center", va="top")
    cax = fig.add_axes([0.38, 0.05, 0.36, 0.013])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=fs.INK_MUTED, length=0)
    cb.set_label("loading: SD change in the measure per SD of axis score", fontsize=8,
                 color=fs.INK_2)
    fig.suptitle("Where in the brain the three largest axes sit",
                 x=0.13, y=0.975, ha="left", fontsize=11, color=fs.INK)
    fig.text(0.13, 0.945, "view from above, regions at their centroids · red = higher "
             "when the axis score is higher, blue = lower · dot size = |loading|",
             ha="left", fontsize=8, color=fs.INK_2)
    p = OUT / "fig4_axis_maps.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


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

    fs.apply()
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
        print(f"wrote {fn()}")


if __name__ == "__main__":
    main()
