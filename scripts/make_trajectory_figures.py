"""Research question 2: how does connectome organisation change with age?

    python scripts/make_trajectory_figures.py

The selected model's mean trajectory, mu(t) = B(t) beta, with pointwise 95%
bands from the 1,000 parametric bootstrap replicates (SUPPLEMENT S25), at the
methods plan's three levels (section 5):

  fig15_trajectories       each measure averaged over the 90 regions (original
                           units, training infants behind it) and by lobe
  fig16_trajectory_maps    regional values at 30, 34, 38 and 42 weeks PMA in SD
                           units of each coordinate (0 = the training mean over
                           all ages); regions whose 95% band includes 0 at that
                           age are grey

Bands are conditional on the selected model (K = 1, q = 70); they do not
include model-selection uncertainty.

Writes results/trajectories.json (summary changes with intervals).
"""

import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

import paths
import atlas
import figstyle as fs
from features import MEASURES
from model import splines
from scripts.run_spec import load
from scripts.make_brain_figures import (load_display_atlas, brain_meshes, region_colours,
                                        draw_surface)

Q = 70
T = np.linspace(27, 45, 73)
AGES = (30, 34, 38, 42)
UNITS = {"strength": "strength (edge-weight units)", "log_clustering": "log clustering",
         "communicability": "communicability"}
LABEL = {"strength": "strength", "log_clustering": "clustering", "communicability": "communicability"}
LOBE_COLOURS = dict(zip(atlas.LOBES, fs.COLOR_CYCLE[:6]))
BANDS = True
N_BOOT = 0


def bootstrap_betas():
    d = paths.RESULTS / "significance" / "bootstrap"
    files = sorted(glob.glob(str(d / "chunk*.npz"))) + sorted(glob.glob(str(d / "rep*.npz")))
    if not files:
        return None                       # bootstrap not run yet: curves without bands
    return np.concatenate([np.load(f)["beta"] for f in files])          # (n_boot, M, D)


def main():
    fit = np.load(paths.RESULTS / "final" / f"K1_q{Q}.npz")
    beta, mu, sd = fit["beta"][0], fit["mu"], fit["sd"]
    betas = bootstrap_betas()
    global BANDS, N_BOOT
    BANDS = betas is not None
    N_BOOT = 0 if betas is None else len(betas)
    if not BANDS:
        print("no bootstrap replicates yet: drawing point estimates without bands")
        betas = beta[None]
    B = splines.basis(T)
    std = B @ beta                                                         # (T, D) SD units
    std_b = np.einsum("tm,bmd->btd", B, betas)                              # (n_boot, T, D)
    orig, orig_b = mu + sd * std, mu + sd * std_b

    def regional_mean(a, m, mask=None):
        cols = np.arange(90) if mask is None else np.where(mask)[0]
        return a[..., 3 * cols + m].mean(-1)

    summary = {"n_boot": len(betas) if BANDS else 0, "q": Q, "measures": {}}
    i30, i42 = np.argmin(np.abs(T - 30)), np.argmin(np.abs(T - 42))
    for m, meas in enumerate(MEASURES):
        est = regional_mean(orig, m)
        bt = regional_mean(orig_b, m)
        ch = bt[:, i42] - bt[:, i30]
        summary["measures"][meas] = dict(
            at_30=float(est[i30]), at_42=float(est[i42]), change_30_42=float(est[i42] - est[i30]),
            change_ci=[float(x) for x in np.percentile(ch, [2.5, 97.5])] if BANDS else None,
            change_sd_units=float(regional_mean(std, m)[i42] - regional_mean(std, m)[i30]))
    (paths.RESULTS / "trajectories.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fs.apply_style()
    fig_curves(orig, orig_b, regional_mean)
    fig_maps(std, std_b)


def fig_curves(orig, orig_b, regional_mean):
    coh, Xraw = load()
    X = Xraw.reshape(len(Xraw), 90, 3)
    fig, axes = plt.subplots(2, 3, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(hspace=0.45, wspace=0.42, bottom=0.14, top=0.95)
    for m, meas in enumerate(MEASURES):
        ax = axes[0, m]
        ax.scatter(coh.scan_age, X[:, :, m].mean(1), s=6, color=fs.NEUTRAL_GREY, alpha=0.35,
                   linewidths=0)
        lo, hi = np.percentile(regional_mean(orig_b, m), [2.5, 97.5], axis=0)
        ax.fill_between(T, lo, hi, color=fs.MEASURE_COLOURS[meas], alpha=0.3, linewidth=0)
        ax.plot(T, regional_mean(orig, m), color=fs.MEASURE_COLOURS[meas], lw=2.0)
        ax.set_xlabel("PMA (weeks)"); ax.set_ylabel(UNITS[meas])
        fs.panel_title(ax, "ABC"[m], LABEL[meas])

        ax = axes[1, m]
        for lobe in atlas.LOBES:
            mask = atlas.LOBE == lobe
            lo, hi = np.percentile(regional_mean(orig_b, m, mask), [2.5, 97.5], axis=0)
            ax.fill_between(T, lo, hi, color=LOBE_COLOURS[lobe], alpha=0.15, linewidth=0)
            ax.plot(T, regional_mean(orig, m, mask), color=LOBE_COLOURS[lobe], lw=1.5)
        ax.set_xlabel("PMA (weeks)"); ax.set_ylabel(UNITS[meas])
        fs.panel_title(ax, "DEF"[m], "by lobe")
    fig.legend(handles=[Patch(color=c, label=l) for l, c in LOBE_COLOURS.items()],
               loc="lower left", bbox_to_anchor=(0.06, 0.035), ncol=6, handlelength=1.2,
               columnspacing=1.0)
    band = (f"shaded = pointwise 95% bootstrap band ({N_BOOT:,} replicates)" if BANDS
            else "bootstrap bands pending")
    fig.text(0.06, 0.02, f"Mean trajectory of the selected model (K = 1, q = {Q}); {band}.\nA–C: averaged over the 90 "
             "regions, with each training infant's regional average in grey. D–F: averaged "
             "within lobes.", fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "fig15_trajectories")


def fig_maps(std, std_b):
    lab, t2, mask, aff = load_display_atlas()
    left = np.isin(lab, np.arange(1, 91, 2)); right = np.isin(lab, np.arange(2, 91, 2))
    xs = np.arange(lab.shape[0])
    mid = 0.5 * ((xs[:, None, None] * left).sum() / left.sum()
                 + (xs[:, None, None] * right).sum() / right.sum())
    meshes = brain_meshes(lab, mask, aff, mid)

    idx = [int(np.argmin(np.abs(T - a))) for a in AGES]
    vmax = float(np.abs(std[idx]).max())
    cmap, norm = fs.DIVERGING, Normalize(-vmax, vmax)
    fig = plt.figure(figsize=fs.FIGURE_SIZES["double_column"])
    gs = fig.add_gridspec(3, 2 * len(AGES), left=0.12, right=0.99, top=0.84, bottom=0.2,
                          wspace=0.02, hspace=0.05)
    for m, meas in enumerate(MEASURES):
        for a, ti in enumerate(idx):
            v = std[ti, m::3]
            lo, hi = np.percentile(std_b[:, ti, m::3], [2.5, 97.5], axis=0)
            colours = region_colours(v, cmap, norm, (lo > 0) | (hi < 0))
            for c, view in enumerate(("left lateral", "left medial")):
                ax = fig.add_subplot(gs[m, 2 * a + c])
                draw_surface(ax, meshes, colours, view)
                if m == 0 and c == 0:
                    ax.text(1.0, 1.08, f"{AGES[a]} weeks", transform=ax.transAxes, ha="center",
                            fontsize=14, fontweight="bold")
                if a == 0 and c == 0:
                    ax.text(-0.06, 0.5, LABEL[meas].replace("communicability", "communic-\nability"),
                            transform=ax.transAxes, ha="right", va="center",
                            fontsize=fs.NOTE_SIZE, fontweight="bold")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fs.horizontal_colorbar(fig, sm, [0.3, 0.12, 0.42, 0.025],
                           "SD from the training mean over all ages")
    fig.suptitle("How each region develops", x=0.06, y=1.0, ha="left", fontsize=16,
                 fontweight="bold")
    grey = (f"Light grey = 95% bootstrap band ({N_BOOT:,} replicates)\nincludes 0 at that age; dark grey" if BANDS
            else "Bootstrap pending, so no region\nis greyed for uncertainty; dark grey")
    fig.text(0.06, -0.02, f"Selected model (K = 1, q = {Q}), left hemisphere, lateral and medial. "
             f"{grey} = tissue outside the 90 AAL regions.", fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY,
             va="top")
    fs.save_figure(fig, paths.FIGURES / "fig16_trajectory_maps")


if __name__ == "__main__":
    main()
