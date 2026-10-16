"""Figures 5-6: the axes drawn on a neonatal brain, region by region.

    python scripts/make_brain_figures.py

fig5_axis{j}_brain      one per axis: each AAL region coloured by its loading,
                        seen from the side and from the midline, an axial slice
                        for the deep structures, and the ten strongest loadings
fig6_region_profiles    every region's loading on the leading axes, by hemisphere

Every figure is written as PNG (150 DPI) and PDF, in the project style.

Display only: regions are drawn on the UNC neonatal T2w template with its own
AAL90 labels (data/atlases/UNC_neonatal_*), which share a 1 mm grid. The
connectomes used AAL90 in GA40 space; both carry the same AAL numbering, which
is checked here by comparing the two atlases' centroid layouts.

The brain views are rendered from the label volume itself: along each viewing
direction the first voxel hit is shown, shaded by the surface's orientation,
so the result looks like a surface render without needing surface meshes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch
from scipy import ndimage

import paths
import atlas
import figstyle as fs
from features import MEASURES
from scripts.run_spec import load
from scripts.make_axes_figures import (principal_axes, correlates, COVARIATES,
                                       MEASURE_LABEL, PRIMARY_Q)

OUT = paths.FIGURES
N_AXES = 5                       # axes 1-5 are stable across q (SUPPLEMENT S14)
BRAIN = to_rgb(fs.NEUTRAL_GREY)  # unlabelled tissue (cerebellum, brainstem)
BG = np.array(to_rgb("white"))
EDGE_DARKEN = 0.72               # parcel borders
SURFACE_TOL = 4                  # voxels: a label this close under the brain surface counts as visible


def load_display_atlas():
    import nibabel as nib
    d = paths.ATLAS_DIR
    lab = np.asarray(nib.load(d / "UNC_neonatal_AAL90_templateflow.nii.gz").dataobj).astype(int)
    t2 = np.asarray(nib.load(d / "UNC_neonatal_T2w_brain_template.nii.gz").dataobj).astype(float)
    mask = np.asarray(nib.load(d / "UNC_neonatal_brain_mask.nii.gz").dataobj) > 0
    img = nib.load(d / "UNC_neonatal_AAL90_templateflow.nii.gz")
    assert nib.aff2axcodes(img.affine) == ("R", "A", "S"), "renderer assumes RAS voxel axes"
    assert set(np.unique(lab)) - {0} == set(range(1, 91)), "display atlas is not AAL1-90"
    return lab, t2, mask | (lab > 0), img.affine


def check_same_numbering(lab, aff):
    """The display atlas and the connectome atlas must number regions alike."""
    c_disp = atlas.centroids(lab, aff)
    c_conn = atlas.centroids()
    a, b = c_disp - c_disp.mean(0), c_conn - c_conn.mean(0)
    r = [np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(3)]
    assert min(r) > 0.95, f"atlas centroid layouts disagree (r = {np.round(r, 3)})"
    return r


# --- rendering ---------------------------------------------------------------

def _first_hit(vol, axis, reverse):
    v = np.flip(vol, axis) if reverse else vol
    return v.any(axis), np.argmax(v, axis)


def _to_image(a, axis, horizontal_flip):
    """2D array over the two remaining axes -> image rows (top = up/anterior)."""
    img = np.flipud(a.T)                       # x-views: rows z; z-view: rows y
    return np.fliplr(img) if horizontal_flip else img


VIEWS = {   # name: (projection axis, look from the + side?, hemisphere, flip columns)
    "left lateral":  (0, False, "L", True),    # anterior on the left
    "left medial":   (0, True, "L", False),    # anterior on the right
    "right medial":  (0, False, "R", True),
    "right lateral": (0, True, "R", False),
}


def render(lab, mask, values, cmap, norm, view, mid):
    axis, reverse, hemi, hflip = VIEWS[view]
    x = np.arange(lab.shape[0])[:, None, None]
    keep = np.ones_like(mask) if hemi is None else ((x < mid) if hemi == "L" else (x >= mid))
    m, l = mask & keep, (lab > 0) & keep
    hit_m, d_m = _first_hit(m, axis, reverse)
    hit_l, d_l = _first_hit(l, axis, reverse)
    labv = np.flip(lab, axis) if reverse else lab
    first_lab = np.take_along_axis(labv, np.expand_dims(d_l, axis), axis).squeeze(axis)
    visible = hit_l & (d_l - d_m <= SURFACE_TOL)

    # Lambertian shading from the smoothed depth map.
    depth = ndimage.gaussian_filter(np.where(hit_m, d_m, d_m[hit_m].max()).astype(float), 1.5)
    gy, gx = np.gradient(depth)
    n = np.stack([-gx, -gy, np.ones_like(depth)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    light = np.array([-0.35, 0.45, 0.82]); light /= np.linalg.norm(light)
    shade = 0.68 + 0.32 * np.clip(n @ light, 0, 1)

    rgb = np.ones(hit_m.shape + (3,)) * BG
    rgb[hit_m] = BRAIN
    lab_vis = np.where(visible, first_lab, 0)
    colours = cmap(norm(values))[:, :3]                      # (90, 3)
    rgb[visible] = colours[lab_vis[visible] - 1]
    rgb[hit_m] *= shade[hit_m][:, None]
    # parcel borders
    edge = np.zeros_like(visible)
    for sh in ((1, 0), (0, 1)):
        diff = lab_vis != np.roll(lab_vis, sh, axis=(0, 1))
        edge |= diff & visible
    rgb[edge] *= EDGE_DARKEN
    return np.stack([_to_image(rgb[..., c], axis, hflip) for c in range(3)], -1)


def deep_slice(lab, t2, values, cmap, norm, k):
    """Axial slice at voxel z = k: T2 in grey, regions coloured on top."""
    sl, bg = lab[:, :, k], t2[:, :, k]
    g = np.clip(bg / np.percentile(t2[t2 > 0], 99), 0, 1)
    rgb = np.where(bg[..., None] > 0, 0.25 + 0.6 * g[..., None], 1.0) * np.ones(3)
    rgb[bg <= 0] = BG
    on = sl > 0
    rgb[on] = 0.15 * rgb[on] + 0.85 * cmap(norm(values))[sl[on] - 1, :3]
    return np.stack([np.flipud(rgb[..., c].T) for c in range(3)], -1)


def crop(img, pad=4):
    fg = np.any(np.abs(img - BG) > 1e-3, -1)
    r, c = np.where(fg)
    return img[max(r.min() - pad, 0):r.max() + pad, max(c.min() - pad, 0):c.max() + pad]


# --- figures -----------------------------------------------------------------

def fig_axis_brain(j, P, lab, t2, mask, mid, k_deep, vmax, corr_text):
    R = P["L"].reshape(90, 3, -1)
    cmap, norm = fs.DIVERGING, Normalize(-vmax, vmax)
    cols = list(VIEWS) + ["deep (axial)"]
    fig = plt.figure(figsize=fs.FIGURE_SIZES["full_page"])
    gs = fig.add_gridspec(5, len(cols), height_ratios=[1, 1, 1, 0.85, 1.75],
                          width_ratios=[1.2, 1.2, 1.2, 1.2, 0.95],
                          left=0.07, right=0.99, top=0.88, bottom=0.1, wspace=0.03, hspace=0.08)
    for m, meas in enumerate(MEASURES):
        v = R[:, m, j]
        for c, view in enumerate(cols):
            ax = fig.add_subplot(gs[m, c])
            img = deep_slice(lab, t2, v, cmap, norm, k_deep) if c == len(cols) - 1 \
                else render(lab, mask, v, cmap, norm, view, mid)
            ax.imshow(crop(img), interpolation="lanczos")
            ax.axis("off")
            if m == 0:
                ax.set_title(view, loc="center", fontsize=14, fontweight="bold", pad=6)
            if c == 0:
                ax.text(-0.06, 0.5, MEASURE_LABEL[meas], transform=ax.transAxes, ha="right",
                        va="center", fontsize=fs.NOTE_SIZE, fontweight="bold", rotation=90)
            if m == 0 and c == 0:
                ax.text(-0.32, 1.1, "A", transform=ax.transAxes, fontsize=16,
                        fontweight="bold", va="bottom")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    spacer = gs[3, :].get_position(fig)
    fs.horizontal_colorbar(fig, sm, [0.3, spacer.y1 - 0.012, 0.42, 0.012],
                           "loading (SD per SD of axis score)")

    # B: the ten strongest loadings
    ax = fig.add_subplot(gs[4, 1:4])
    v = P["L"][:, j]
    top = np.argsort(-np.abs(v))[:10][::-1]
    vals = v[top]
    lim = vmax * 1.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-0.6, len(top) - 0.4)
    fs.rounded_bars(ax, np.arange(len(top)), vals, width=0.7, horizontal=True,
                    color=[fs.POSITIVE_COLOUR if x > 0 else fs.NEGATIVE_COLOUR for x in vals],
                    value_fmt="{:+.2f}")
    ax.axvline(0, color=fs.TEXT_PRIMARY, lw=0.8)
    ax.set_yticks(range(len(top)),
                  [f"{short(atlas.NAMES[d // 3])} · {MEASURE_LABEL[MEASURES[d % 3]]}" for d in top])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("loading")
    fs.panel_title(ax, "B", "Strongest loadings")

    fig.suptitle(f"Axis {j + 1}: {P['share'][j] * 100:.1f}% of variation beyond age",
                 x=0.07, y=0.985, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.07, 0.945, corr_text, ha="left", fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY)
    fig.text(0.07, 0.0, "One-group model, q = 15. Red = higher in infants scoring high on "
             "the axis, blue = lower;\ngrey = tissue outside the 90 AAL regions. Same colour "
             "scale for every axis.", fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    return fs.save_figure(fig, OUT / f"fig5_axis{j + 1}_brain")


def short(name):
    base, h = name.rsplit("_", 1)
    return base.replace("_", " ") + f" {h}"


def fig_region_profiles(P, vmax):
    R = P["L"].reshape(90, 3, -1)
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGURE_SIZES["multi_panel_tall"])
    fig.subplots_adjust(left=0.16, right=0.99, top=0.9, bottom=0.08, wspace=0.62)
    for ax, (hemi, letter, title) in zip(axes, (("L", "A", "Left hemisphere"),
                                                ("R", "B", "Right hemisphere"))):
        order = [r for lobe in atlas.LOBES
                 for r in np.where((atlas.LOBE == lobe) & (atlas.HEMI == hemi))[0]]
        M = np.concatenate([R[order, :, j] for j in range(N_AXES)], axis=1)
        im = ax.imshow(M, cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="nearest")
        ax.set_yticks(range(len(order)), [short(atlas.NAMES[r])[:-2] for r in order])
        ax.set_xticks(range(3 * N_AXES), ["str", "clu", "com"] * N_AXES, rotation=90)
        for j in range(N_AXES):
            ax.text(3 * j + 1, -0.8, f"axis {j + 1}", ha="center", va="bottom",
                    fontsize=fs.NOTE_SIZE, fontweight="bold")
            if j:
                ax.axvline(3 * j - 0.5, color="white", lw=3)
        start = 0
        for lobe in atlas.LOBES:
            n = int(((atlas.LOBE == lobe) & (atlas.HEMI == hemi)).sum())
            if start:
                ax.axhline(start - 0.5, color="white", lw=3)
            ax.text(-0.43, start + (n - 1) / 2, lobe, transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=fs.NOTE_SIZE, fontweight="bold",
                    rotation=90)
            start += n
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{letter}   {title}", pad=30)
    fs.horizontal_colorbar(fig, im, [0.35, 0.04, 0.4, 0.008],
                           "loading (SD per SD of axis score)")
    fig.text(0.02, 0.0, "str = strength, clu = clustering, com = communicability. Red = higher "
             "in infants scoring high on the axis, blue = lower. One-group model, q = 15.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    return fs.save_figure(fig, OUT / "fig6_region_profiles")


def main():
    atlas.check_node_index()
    P = principal_axes(PRIMARY_Q)
    coh, _ = load()
    assert (coh.dhcp_id.astype(str).values == P["ids"]).all()
    lab, t2, mask, aff = load_display_atlas()
    r = check_same_numbering(lab, aff)
    print("display vs connectome atlas centroid r (x, y, z):", np.round(r, 3))

    left = np.isin(lab, np.arange(1, 91, 2)); right = np.isin(lab, np.arange(2, 91, 2))
    xs = np.arange(lab.shape[0])
    mid = 0.5 * ((xs[:, None, None] * left).sum() / left.sum()
                 + (xs[:, None, None] * right).sum() / right.sum())
    thal = np.isin(lab, [77, 78])
    k_deep = int(round(np.argwhere(thal)[:, 2].mean()))

    R = P["L"].reshape(90, 3, -1)
    vmax = float(np.abs(R[:, :, :N_AXES]).max())
    rc, r_crit = correlates(P["scores"], coh, N_AXES)
    fs.apply_style()
    for j in range(N_AXES):
        sig = [(abs(rc[j, c]), f"{name} {rc[j, c]:+.2f}")
               for c, name in enumerate(COVARIATES) if abs(rc[j, c]) >= r_crit]
        text = ("related to: " + ", ".join(s for _, s in sorted(sig, reverse=True))) if sig \
            else "no covariate passes the Bonferroni threshold"
        print("wrote", fig_axis_brain(j, P, lab, t2, mask, mid, k_deep, vmax, text)[0])
        for m, meas in enumerate(MEASURES):
            v = R[:, m, j]
            top = np.argsort(-v)[:4]; bot = np.argsort(v)[:3]
            print(f"   {MEASURE_LABEL[meas]:>15}: + " +
                  ", ".join(f"{short(atlas.NAMES[i])} {v[i]:+.2f}" for i in top) +
                  "  | - " + ", ".join(f"{short(atlas.NAMES[i])} {v[i]:+.2f}" for i in bot))
    print("wrote", fig_region_profiles(P, vmax)[0])


if __name__ == "__main__":
    main()
