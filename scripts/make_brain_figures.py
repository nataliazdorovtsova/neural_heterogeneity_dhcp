"""Figures 5-6: the axes drawn on a neonatal brain, region by region.

    python scripts/make_brain_figures.py [--q Q]

fig5_axis{j}_brain      one per axis: each AAL region coloured by its loading on
                        a surface render of each hemisphere (lateral and medial),
                        an axial slice for the deep structures, and the ten
                        strongest loadings (with bootstrap intervals once
                        scripts/summarise_bootstrap.py has run)
fig6_region_profiles    every region's loading on the leading axes, by hemisphere

Every figure is written as PNG (150 DPI) and PDF, in the project style.

Display only: regions are drawn on the UNC neonatal T2w template with its own
AAL90 labels (data/atlases/UNC_neonatal_*), which share a 1 mm grid. The
connectomes used AAL90 in GA40 space; both carry the same AAL numbering, which
is checked here by comparing the two atlases' centroid layouts.

Surface renders. Each hemisphere's surface is extracted from the template brain
mask by marching cubes, smoothed (Taubin) and subdivided once, so it is a
smooth mesh of ~0.5 mm triangles rather than 1 mm voxels. Each vertex takes the
region whose smoothed mask is strongest there, which gives smooth region
borders instead of voxel staircases. Triangles are drawn as polygons, shaded by
their orientation to the light and painted back to front, so nothing is
resampled or interpolated. The mesh is cached in results/final/brain_mesh.npz.

When bootstrap intervals exist for the plotted model, loadings whose 95%
interval includes zero are drawn in light grey.
"""

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgb
from scipy import ndimage

import paths
import atlas
import figstyle as fs
from features import MEASURES
from scripts.run_spec import load
from scripts.make_axes_figures import principal_axes, correlates, COVARIATES, MEASURE_LABEL

OUT = paths.FIGURES
N_AXES = 5                        # axes 1-5 are stable across q (SUPPLEMENT S14)
BRAIN = np.array(to_rgb(fs.NEUTRAL_GREY))     # tissue outside the 90 AAL regions
NONSIG = np.array(to_rgb("#E2E2E2"))           # loading whose interval includes zero
BORDER_DARKEN = 0.7
MESH_CACHE = paths.RESULTS / "final" / "brain_mesh.npz"
LABEL_SMOOTH = 1.0                # voxels: smoothing of each region's mask for labelling
UNION_SMOOTH = 1.5                # voxels: smoothing of the all-regions mask...
UNION_MIN = 0.12                  # ...below which a vertex is outside every region
UP = np.array([0.0, 0.0, 1.0])


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


# --- surface meshes ------------------------------------------------------------

def _edges(faces):
    """All face edges, in blocks (01 | 12 | 20), each row sorted."""
    return np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)


def _taubin(v, faces, iters, lam=0.5, mu=-0.53):
    """Taubin smoothing: removes the voxel staircase without shrinking the mesh."""
    from scipy import sparse
    e = _edges(faces)
    n = len(v)
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    W = sparse.diags(1.0 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    for _ in range(iters):
        v = v + lam * (W @ v - v)
        v = v + mu * (W @ v - v)
    return v


def _subdivide(v, faces):
    """Split every triangle into four at its edge midpoints."""
    uniq, inv = np.unique(_edges(faces), axis=0, return_inverse=True)
    mid = len(v) + inv.ravel().reshape(3, -1).T          # (F, 3): midpoints of 01, 12, 20
    v2 = np.vstack([v, 0.5 * (v[uniq[:, 0]] + v[uniq[:, 1]])])
    a, b, c = faces.T
    ab, bc, ca = mid.T
    return v2, np.concatenate([np.c_[a, ab, ca], np.c_[ab, b, bc], np.c_[ca, bc, c],
                               np.c_[ab, bc, ca]])


def _outward(v, faces):
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = ((tri.mean(1) - v.mean(0)) * n).sum(1)
    return faces if (out > 0).mean() > 0.5 else faces[:, ::-1]


def _vertex_labels(v_vox, lab, labels):
    best = np.zeros(len(v_vox), np.float32)
    out = np.zeros(len(v_vox), int)
    for l in labels:
        sm = ndimage.gaussian_filter((lab == l).astype(np.float32), LABEL_SMOOTH)
        val = ndimage.map_coordinates(sm, v_vox.T, order=1)
        better = val > best
        best[better], out[better] = val[better], l
    # Whether a vertex is in any region at all is decided by a smoothed union of
    # the hemisphere's regions, so the border with unlabelled tissue (callosum,
    # brainstem) is a smooth curve too. Medial cortex sitting slightly back from
    # the midline cut still reaches the threshold.
    union = ndimage.gaussian_filter(np.isin(lab, list(labels)).astype(np.float32), UNION_SMOOTH)
    out[ndimage.map_coordinates(union, v_vox.T, order=1) < UNION_MIN] = 0
    return out


def _hemisphere_mesh(mask, lab, aff, mid, hemi):
    from skimage.measure import marching_cubes
    from nibabel.affines import apply_affine
    x = np.arange(mask.shape[0])[:, None, None]
    keep = (x < mid) if hemi == "L" else (x >= mid)
    vol = ndimage.gaussian_filter((mask & keep).astype(np.float32), 1.3)
    v, f, _, _ = marching_cubes(vol, 0.5)
    v = _taubin(v, f, iters=25)
    v, f = _subdivide(v, f)
    v = _taubin(v, f, iters=10)
    f = _outward(v, f)
    vl = _vertex_labels(v, lab, range(1, 91, 2) if hemi == "L" else range(2, 91, 2))
    return apply_affine(aff, v), f, vl


def brain_meshes(lab, mask, aff, mid):
    if MESH_CACHE.exists():
        z = np.load(MESH_CACHE)
        return {h: (z[f"v{h}"], z[f"f{h}"], z[f"l{h}"]) for h in "LR"}
    meshes = {h: _hemisphere_mesh(mask, lab, aff, mid, h) for h in "LR"}
    np.savez(MESH_CACHE, **{f"{k}{h}": x for h, m in meshes.items()
                           for k, x in zip("vfl", m)})
    return meshes


def _face_labels(f, vl):
    a, b, c = vl[f].T
    fl = np.where(a == b, a, np.where(b == c, b, np.where(a == c, a, a)))
    border = ~((a == b) & (b == c))
    return fl, border


# --- drawing -------------------------------------------------------------------

VIEWS = {   # name: (hemisphere, direction towards the viewer, screen-right direction)
    "left lateral":  ("L", (-1, 0, 0), (0, -1, 0)),    # anterior on the left
    "left medial":   ("L", (1, 0, 0), (0, 1, 0)),      # anterior on the right
    "right medial":  ("R", (-1, 0, 0), (0, -1, 0)),
    "right lateral": ("R", (1, 0, 0), (0, 1, 0)),
}


def region_colours(values, cmap, norm, sig=None):
    col = cmap(norm(values))[:, :3]
    if sig is not None:
        col[~sig] = NONSIG
    return col


def draw_surface(ax, mesh, colours, view):
    hemi, d, right = VIEWS[view]
    d, right = np.array(d, float), np.array(right, float)
    v, f, vl = mesh[hemi]
    fl, border = _face_labels(f, vl)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    vis = n @ d > 0                                   # faces turned towards the viewer
    order = np.argsort(tri[vis].mean(1) @ d)          # far first (painter's algorithm)
    tri, n, fl, border = tri[vis][order], n[vis][order], fl[vis][order], border[vis][order]

    rgb = np.tile(BRAIN, (len(tri), 1))
    on = fl > 0
    rgb[on] = colours[fl[on] - 1]
    light = d + 0.45 * UP - 0.35 * right
    light /= np.linalg.norm(light)
    rgb *= (0.6 + 0.4 * np.clip(n @ light, 0, 1))[:, None]
    rgb[border & on] *= BORDER_DARKEN
    xy = np.stack([tri @ right, tri @ UP], -1)
    ax.add_collection(PolyCollection(xy, facecolors=rgb, edgecolors=rgb, linewidths=0.2,
                                     rasterized=True))
    ax.autoscale_view()
    ax.set_aspect("equal"); ax.axis("off")


def draw_deep_slice(ax, lab, t2, colours, k, up=4):
    """Axial slice at voxel z = k at `up` x resolution, with smooth region borders."""
    sl = lab[:, :, k]
    bg = ndimage.zoom(t2[:, :, k], up, order=1)
    best = np.zeros(bg.shape, np.float32)
    lab_up = np.zeros(bg.shape, int)
    for l in np.unique(sl[sl > 0]):
        sm = ndimage.zoom(ndimage.gaussian_filter((sl == l).astype(np.float32), 0.8), up, order=1)
        better = sm > best
        best[better], lab_up[better] = sm[better], l
    lab_up[best < 0.35] = 0
    g = np.clip(bg / np.percentile(t2[t2 > 0], 99), 0, 1)
    # a smooth brain outline: the raw 1 mm mask edge would be a staircase
    inside = ndimage.zoom(ndimage.gaussian_filter((t2[:, :, k] > 0).astype(np.float32), 1.0),
                          up, order=1) > 0.65
    # the template's skull-stripped edge is dark and speckled; even it out
    rim = inside & ~ndimage.binary_erosion(inside, iterations=3 * up)
    g = np.where(rim, np.maximum(g, 0.6), g)
    rgb = np.where(inside[..., None], (0.25 + 0.6 * g)[..., None] * np.ones(3), 1.0)
    on = (lab_up > 0) & inside
    rgb[on] = 0.12 * rgb[on] + 0.88 * colours[lab_up[on] - 1]
    edge = np.zeros_like(on)
    for sh in ((1, 0), (0, 1)):
        edge |= (lab_up != np.roll(lab_up, sh, axis=(0, 1))) & on
    rgb[edge] *= BORDER_DARKEN
    outline = inside & ~ndimage.binary_erosion(inside, iterations=2)
    rgb[outline] = 0.45
    img = np.flipud(np.transpose(rgb, (1, 0, 2)))     # anterior up, right on the right
    rows, cols = np.where(np.any(img < 0.999, -1))
    img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")


# --- figures -----------------------------------------------------------------

def fig_axis_brain(j, P, meshes, lab, t2, k_deep, vmax, corr_text, boot=None):
    R = P["L"].reshape(90, 3, -1)
    cmap, norm = fs.DIVERGING, Normalize(-vmax, vmax)
    sigR = None if boot is None else boot["significant"].reshape(90, 3, -1)
    cols = list(VIEWS) + ["deep (axial)"]
    fig = plt.figure(figsize=fs.FIGURE_SIZES["full_page"])
    gs = fig.add_gridspec(5, len(cols), height_ratios=[1, 1, 1, 0.85, 1.75],
                          width_ratios=[1.2, 1.2, 1.2, 1.2, 0.95],
                          left=0.07, right=0.99, top=0.88, bottom=0.1, wspace=0.03, hspace=0.08)
    for m, meas in enumerate(MEASURES):
        colours = region_colours(R[:, m, j], cmap, norm,
                                 None if sigR is None else sigR[:, m, j])
        for c, view in enumerate(cols):
            ax = fig.add_subplot(gs[m, c])
            if c == len(cols) - 1:
                draw_deep_slice(ax, lab, t2, colours, k_deep)
            else:
                draw_surface(ax, meshes, colours, view)
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

    # B: the ten strongest loadings, with bootstrap intervals when available
    ax = fig.add_subplot(gs[4, 1:4])
    v = P["L"][:, j]
    top = np.argsort(-np.abs(v))[:10][::-1]
    vals = v[top]
    lim = vmax * 1.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-0.6, len(top) - 0.4)
    fs.rounded_bars(ax, np.arange(len(top)), vals, width=0.7, horizontal=True,
                    color=[fs.POSITIVE_COLOUR if x > 0 else fs.NEGATIVE_COLOUR for x in vals],
                    value_fmt=None if boot is not None else "{:+.2f}")
    if boot is not None:
        lo, hi = boot["lo"][top, j], boot["hi"][top, j]
        ax.errorbar(vals, np.arange(len(top)), xerr=[vals - lo, hi - vals], fmt="none",
                    ecolor=fs.TEXT_PRIMARY, elinewidth=0.8, capsize=3, zorder=4)
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
    note = (f"One-group model, q = {P['q']}. Red = higher in infants scoring high on the axis, "
            "blue = lower;\ndark grey = tissue outside the 90 AAL regions. Same colour scale "
            "for every axis.")
    if boot is not None:
        note += (f"\nLight grey = 95% bootstrap interval includes zero ({boot['n_boot']} "
                 "replicates); B: error bars = 95% intervals.")
    fig.text(0.07, 0.0, note, fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
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
             f"in infants scoring high on the axis, blue = lower. One-group model, q = {P['q']}.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    return fs.save_figure(fig, OUT / "fig6_region_profiles")


def analysis_q(cli_q):
    if cli_q is not None:
        return cli_q
    f = paths.RESULTS / "selection.json"
    if f.exists():
        sel = json.loads(f.read_text())["selected"]
        if sel["K"] == 1:
            return sel["q"]
    return 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=None,
                    help="model to draw (default: the selected one-group model, else 15)")
    a = ap.parse_args()
    q = analysis_q(a.q)

    atlas.check_node_index()
    P = principal_axes(q)
    coh, _ = load()
    assert (coh.dhcp_id.astype(str).values == P["ids"]).all()
    lab, t2, mask, aff = load_display_atlas()
    r = check_same_numbering(lab, aff)
    print(f"q = {q}; display vs connectome atlas centroid r (x, y, z):", np.round(r, 3))

    bf = paths.RESULTS / "significance" / "bootstrap_summary.npz"
    boot = None
    if bf.exists():
        z = np.load(bf)
        if int(z["q"]) == q:
            boot = {k: z[k] for k in z.files}
            print(f"using bootstrap intervals ({int(z['n_boot'])} replicates)")

    left = np.isin(lab, np.arange(1, 91, 2)); right = np.isin(lab, np.arange(2, 91, 2))
    xs = np.arange(lab.shape[0])
    mid = 0.5 * ((xs[:, None, None] * left).sum() / left.sum()
                 + (xs[:, None, None] * right).sum() / right.sum())
    k_deep = int(round(np.argwhere(np.isin(lab, [77, 78]))[:, 2].mean()))
    meshes = brain_meshes(lab, mask, aff, mid)
    print("mesh faces:", {h: len(m[1]) for h, m in meshes.items()})

    R = P["L"].reshape(90, 3, -1)
    vmax = float(np.abs(R[:, :, :N_AXES]).max())
    rc, r_crit = correlates(P["scores"], coh, N_AXES)
    fs.apply_style()
    for j in range(N_AXES):
        sig = [(abs(rc[j, c]), f"{name} {rc[j, c]:+.2f}")
               for c, name in enumerate(COVARIATES) if abs(rc[j, c]) >= r_crit]
        text = ("related to: " + ", ".join(s for _, s in sorted(sig, reverse=True))) if sig \
            else "no covariate passes the Bonferroni threshold"
        print("wrote", fig_axis_brain(j, P, meshes, lab, t2, k_deep, vmax, text, boot)[0])
    print("wrote", fig_region_profiles(P, vmax)[0])


if __name__ == "__main__":
    main()
