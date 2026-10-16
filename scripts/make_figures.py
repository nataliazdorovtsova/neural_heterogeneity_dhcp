"""Figures 1-2, in the project figure style (figstyle.py).

    python scripts/make_figures.py

fig1_data_space          the data: topological space, prematurity, measures and density vs age
fig2_model_selection     held-out score against q and against K
fig2b_group_collapse     whether multi-group models kept all their groups

Every figure is written as PNG (150 DPI) and PDF.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paths
import figstyle as fs
from features import standardise, MEASURES

OUT = paths.FIGURES
OUT.mkdir(parents=True, exist_ok=True)
MEASURE_LABEL = {"strength": "strength", "log_clustering": "clustering (log)",
                 "communicability": "communicability"}


def load_primary():
    coh = pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv")
    d = np.load(paths.RESULTS / "features" / "perbin25_mst.npz")
    m = coh.in_primary.values
    return coh[m].reset_index(drop=True), d["X"][m]


# ------------------------------------------------------- figure 1: the data ----

def fig_data_space():
    coh, X = load_primary()
    age = coh.scan_age.values
    preterm = coh.preterm.values.astype(bool)
    Z, _, _ = standardise(X.reshape(len(X), -1))
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var = (S ** 2 / (S ** 2).sum())[:2] * 100
    age_norm = Normalize(age.min(), age.max())

    fig, axes = plt.subplots(2, 2, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(hspace=0.42, wspace=0.32, bottom=0.2, top=0.95)

    # A: the population in topological space, by age
    ax = axes[0, 0]
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=age, cmap=fs.SEQUENTIAL_VISIBLE, norm=age_norm,
                    s=16, linewidths=0.3, edgecolors="white")
    fs.panel_title(ax, "A", "Topological space")
    ax.set_xlabel(f"PC1 ({var[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}%)")

    # B: the same projection by prematurity
    ax = axes[0, 1]
    for flag, colour in ((False, fs.TERM_COLOUR), (True, fs.PRETERM_COLOUR)):
        s = preterm == flag
        ax.scatter(pcs[s, 0], pcs[s, 1], color=colour, s=16, linewidths=0.3,
                   edgecolors="white", alpha=0.9)
    fs.panel_title(ax, "B", "Term vs preterm")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # C: each measure, averaged over regions, against age
    ax = axes[1, 0]
    order = np.argsort(age)
    for j, meas in enumerate(MEASURES):
        y = X[:, :, j].mean(1)
        y = (y - y.mean()) / y.std()
        ax.scatter(age, y, s=6, color=fs.NEUTRAL_GREY, alpha=0.18, linewidths=0)
        sm = pd.Series(y[order]).rolling(60, center=True, min_periods=15).mean()
        ax.plot(age[order], sm, color=fs.MEASURE_COLOURS[meas], lw=2.0, zorder=3)
    ax.set_ylim(-3.6, 2.4)
    fs.panel_title(ax, "C", "Measures vs age")
    ax.set_xlabel("postmenstrual age (weeks)"); ax.set_ylabel("z-scored value")

    # D: the density confound (SUPPLEMENT S2)
    ax = axes[1, 1]
    dens = coh.density_thr.values * 100
    ax.scatter(age, dens, c=age, cmap=fs.SEQUENTIAL_VISIBLE, norm=age_norm, s=16,
               linewidths=0.3, edgecolors="white")
    z = np.polyfit(age, dens, 1)
    xs = np.linspace(age.min(), age.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=fs.TEXT_PRIMARY, lw=2.0)
    r = np.corrcoef(age, dens)[0, 1]
    ax.annotate(f"r = {r:+.2f}", (0.97, 0.95), xycoords="axes fraction", ha="right",
                va="top", fontsize=fs.NOTE_SIZE)
    fs.panel_title(ax, "D", "Density vs age")
    ax.set_xlabel("postmenstrual age (weeks)"); ax.set_ylabel("network density (%)")

    # shared legend and colour bar at the bottom
    handles = [Patch(color=fs.TERM_COLOUR, label=f"term (n = {(~preterm).sum()})"),
               Patch(color=fs.PRETERM_COLOUR, label=f"preterm (n = {preterm.sum()})")] + \
              [Patch(color=fs.MEASURE_COLOURS[m], label=MEASURE_LABEL[m]) for m in MEASURES]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.08, 0.035), ncol=3,
               handlelength=1.2, columnspacing=1.2)
    fs.horizontal_colorbar(fig, sc, [0.66, 0.075, 0.26, 0.015], "postmenstrual age (weeks)")
    return fs.save_figure(fig, OUT / "fig1_data_space")


# -------------------------------------------------- figure 2: model selection ----

def _summary():
    f = paths.RESULTS / "sweep_summary.csv"
    if not f.exists():
        print("  (no sweep_summary.csv — run scripts/collect_sweep.py first)"); return None
    df = pd.read_csv(f)
    if "gain_per_infant" not in df:
        print("  (sweep_summary.csv predates the baseline — rerun scripts/collect_sweep.py)")
        return None
    return df


def _points(ax, x, y, conv, colour, marker):
    """Filled markers for converged fits, open ones for fits that did not converge."""
    conv = np.asarray(conv, bool)
    ax.scatter(x[conv], y[conv], marker=marker, s=40, color=colour, zorder=3,
               edgecolors="white", linewidths=0.5)
    ax.scatter(x[~conv], y[~conv], marker=marker, s=40, facecolors="white", zorder=3,
               edgecolors=colour, linewidths=1.4)


def fig_model_selection():
    df = _summary()
    if df is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGURE_SIZES["double_column"])
    fig.subplots_adjust(wspace=0.3, bottom=0.36, top=0.88)

    # A: score against q, one line per K
    ax = axes[0]
    for K, g in df.groupby("K"):
        g = g.sort_values("q")
        ax.plot(g.q, g.gain_per_infant, color=fs.K_COLOURS[K], zorder=2)
        _points(ax, g.q.values, g.gain_per_infant.values, g.converged.values,
                fs.K_COLOURS[K], fs.K_MARKERS[K])
    fs.panel_title(ax, "A", "Score vs number of axes")
    ax.set_xlabel("number of axes q"); ax.set_ylabel("gain per infant")
    handles = [Line2D([], [], color=fs.K_COLOURS[K], marker=fs.K_MARKERS[K], label=f"K = {K}")
               for K in sorted(df.K.unique())]
    handles.append(Line2D([], [], color=fs.TEXT_PRIMARY, marker="o", ls="none",
                          markerfacecolor="white", label="not converged"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3,
              handlelength=1.6, columnspacing=1.0)

    # B: score against K, one line per q (q is ordered, so a sequential scale)
    ax = axes[1]
    qs = sorted(df.q.unique())
    shades = dict(zip(qs, fs.sequential_shades(len(qs))))
    for q in qs:
        g = df[df.q == q].sort_values("K")
        ax.plot(g.K, g.gain_per_infant, color=shades[q], zorder=2)
        _points(ax, g.K.values, g.gain_per_infant.values, g.converged.values, shades[q], "o")
    fs.panel_title(ax, "B", "Score vs number of groups")
    ax.set_xlabel("number of groups K"); ax.set_ylabel("gain per infant")
    ax.set_xticks(sorted(df.K.unique()))
    ax.legend(handles=[Patch(color=shades[q], label=f"q = {q}") for q in qs],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, handlelength=1.2,
              columnspacing=1.0)

    n_inf = int(df.n_heldout.iloc[0])
    fig.text(0.07, -0.04, f"Gain per infant: held-out log predictive density (5-fold CV, "
             f"n = {n_inf}) above a baseline that knows only each measure's mean and SD. "
             "Higher is better; a gap of 1 means the data are e ≈ 2.7× more plausible.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, wrap=True)
    return fs.save_figure(fig, OUT / "fig2_model_selection")


def fig_group_collapse():
    df = _summary()
    if df is None:
        return None
    sub = df[df.K > 1]
    FLOOR = 1e-3        # a vanished group is ~1e-47; the floor keeps the axis readable
    share = sub.min_group_share.values * 100
    collapsed = share < FLOOR
    drawn = np.maximum(share, FLOOR)

    fig, ax = plt.subplots(figsize=fs.FIGURE_SIZES["single_column"])
    ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.35, 400)
    ax.set_xlim(sub.K.min() - 0.5, sub.K.max() + 0.5)
    fs.reference_line(ax, 2)
    ax.scatter(sub.K[~collapsed], drawn[~collapsed], color=fs.COLORS["blue"], s=40,
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.scatter(sub.K[collapsed], drawn[collapsed], marker="v", s=55,
               color=fs.COLORS["vermillion"], edgecolors="white", linewidths=0.5, zorder=3)
    # one label per K and outcome: the points sit on top of one another
    for (K, col), g in sub.assign(col=collapsed).groupby(["K", "col"]):
        if not col and (g.min_group_share.max() * 100) >= 5:
            continue
        y = FLOOR if col else g.min_group_share.max() * 100
        ax.annotate("q " + ",".join(str(v) for v in sorted(g.q)), (K, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=fs.VALUE_SIZE)
    ax.annotate("2%", (sub.K.max() + 0.45, 2), xytext=(0, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=fs.VALUE_SIZE, color=fs.TEXT_MUTED)
    ax.set_xticks(sorted(sub.K.unique()))
    ax.set_xlabel("number of groups K"); ax.set_ylabel("smallest group (%)")
    ax.set_title("Did any group collapse?", pad=12)
    ax.legend(handles=[Patch(color=fs.COLORS["blue"], label="all groups kept"),
                       Patch(color=fs.COLORS["vermillion"], label="a group vanished")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return fs.save_figure(fig, OUT / "fig2b_group_collapse")


if __name__ == "__main__":
    fs.apply_style()
    for fn in (fig_data_space, fig_model_selection, fig_group_collapse):
        p = fn()
        if p:
            print("wrote " + ", ".join(str(x) for x in p))
