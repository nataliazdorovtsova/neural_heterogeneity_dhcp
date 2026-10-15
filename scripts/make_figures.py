"""Figures, styled after the figure in the research proposal.

    python scripts/make_figures.py

fig1_data_space.png   the proposal's schematic panels, drawn from the real data
fig2_model_selection.png   what the sweep is telling us
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paths
import figstyle as fs
from features import standardise

OUT = paths.FIGURES
OUT.mkdir(parents=True, exist_ok=True)
MEASURE_LABEL = ["strength", "log clustering", "communicability"]


def load_primary():
    coh = pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv")
    d = np.load(paths.RESULTS / "features" / "perbin25_mst.npz")
    X = d["X"]
    m = coh.in_primary.values
    return coh[m].reset_index(drop=True), X[m]


# ------------------------------------------------------- figure 1: the data ----

def fig_data_space():
    coh, X = load_primary()
    age = coh.scan_age.values
    preterm = coh.preterm.values
    Z, _, _ = standardise(X.reshape(len(X), -1))

    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var = (S ** 2 / (S ** 2).sum())[:2] * 100

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.6))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, bottom=0.17, top=0.9)

    # A — population in topological space, coloured by age
    ax = axes[0, 0]; fs.tidy(ax)
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=age, cmap=fs.AGE_RAMP, s=13,
                    linewidths=0.4, edgecolors=fs.SURFACE, alpha=0.95)
    fs.panel_letter(ax, "A", "Population in topological space")
    ax.set_xlabel(f"PC1 ({var[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}%)")

    # B — the same projection by preterm status: is the structure just prematurity?
    ax = axes[0, 1]; fs.tidy(ax)
    for flag, colour, lab in [(False, fs.CLUSTER[0], "term"), (True, fs.CLUSTER[1], "preterm")]:
        s = preterm == flag
        ax.scatter(pcs[s, 0], pcs[s, 1], c=colour, s=13, linewidths=0.4,
                   edgecolors=fs.SURFACE, alpha=0.9, label=f"{lab} (n={s.sum()})")
    ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.2)
    fs.panel_letter(ax, "B", "Is that structure just prematurity?")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # C — each measure averaged across regions, against age (the proposal's summary view)
    ax = axes[1, 0]; fs.tidy(ax)
    order = np.argsort(age)
    curves = []
    for j, lab in enumerate(MEASURE_LABEL):
        y = X[:, :, j].mean(1)
        y = (y - y.mean()) / y.std()
        ax.scatter(age, y, s=7, c=fs.NEUTRAL, alpha=0.15, linewidths=0)
        sm = pd.Series(y[order]).rolling(60, center=True, min_periods=15).mean()
        ax.plot(age[order], sm, color=fs.SERIES[j], lw=2.2, zorder=3)
        curves.append((lab, sm.iloc[-1], fs.SERIES[j]))
    # Place the three direct labels in vertical order with a guaranteed minimum
    # gap, so they can never print on top of one another.
    curves.sort(key=lambda c: -c[1])
    span = 0.42
    for i, (lab, yend, colour) in enumerate(curves):
        ax.annotate(lab, (age.max(), yend), xytext=(8, (1 - i) * span * 34),
                    textcoords="offset points", fontsize=7.5, color=colour,
                    va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.7,
                                    shrinkA=0, shrinkB=2, alpha=0.5))
    ax.set_xlim(26, 48.5)
    ax.set_ylim(-3.6, 2.4)
    fs.panel_letter(ax, "C", "Each measure against age (region-averaged, z-scored)")
    ax.set_xlabel("postmenstrual age (weeks)"); ax.set_ylabel("standardised value")

    # D — the density confound (SUPPLEMENT S2)
    ax = axes[1, 1]; fs.tidy(ax)
    ax.scatter(age, coh.density_thr.values * 100, c=age, cmap=fs.AGE_RAMP, s=13,
               linewidths=0.4, edgecolors=fs.SURFACE, alpha=0.95)
    z = np.polyfit(age, coh.density_thr.values * 100, 1)
    xs = np.linspace(age.min(), age.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=fs.INK_2, lw=1.6, ls=(0, (4, 2)))
    r = np.corrcoef(age, coh.density_thr.values)[0, 1]
    ax.annotate(f"r = {r:+.3f}", (0.96, 0.93), xycoords="axes fraction",
                ha="right", fontsize=8.5, color=fs.INK)
    fs.panel_letter(ax, "D", "Density still tracks age after thresholding")
    ax.set_xlabel("postmenstrual age (weeks)"); ax.set_ylabel("network density (%)")

    fs.age_colorbar(fig, sc)
    fig.suptitle("Neonatal connectome features across the perinatal period",
                 x=0.085, y=0.975, ha="left", fontsize=11, color=fs.INK)
    fig.text(0.085, 0.945, f"dHCP, n = {len(coh)} infants (earliest scan each), AAL90, "
             "per-bin 25% + MST", ha="left", fontsize=8, color=fs.INK_2)
    p = OUT / "fig1_data_space.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# -------------------------------------------------- figure 2: model selection ----

GAIN_LABEL = "gain per infant over baseline (log units)"


def _points(ax, x, y, conv, color, size=30):
    """Filled markers for converged fits, open ones for fits still moving."""
    conv = np.asarray(conv, bool)
    ax.scatter(x[conv], y[conv], s=size, color=color, zorder=3,
               linewidths=0.5, edgecolors=fs.SURFACE)
    ax.scatter(x[~conv], y[~conv], s=size, facecolors=fs.SURFACE, zorder=3,
               edgecolors=color, linewidths=1.4)


def fig_model_selection():
    f = paths.RESULTS / "sweep_summary.csv"
    if not f.exists():
        print("  (no sweep_summary.csv — run scripts/collect_sweep.py first)"); return None
    df = pd.read_csv(f)
    if "gain_per_infant" not in df:
        print("  (sweep_summary.csv predates the baseline — rerun scripts/collect_sweep.py)"); return None

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    fig.subplots_adjust(wspace=0.42, bottom=0.2, top=0.8)

    # A — score against rank, one line per K
    ax = axes[0]; fs.tidy(ax)
    for i, (K, g) in enumerate(df.groupby("K")):
        g = g.sort_values("q")
        ax.plot(g.q, g.gain_per_infant, color=fs.SERIES[i % 5], lw=2.0, zorder=2)
        _points(ax, g.q.values, g.gain_per_infant.values, g.converged.values, fs.SERIES[i % 5])
        ax.annotate(f"K={K}", (g.q.iloc[-1], g.gain_per_infant.iloc[-1]),
                    xytext=(5, 0), textcoords="offset points", fontsize=8,
                    color=fs.SERIES[i % 5], va="center", fontweight="bold")
    ax.scatter([], [], s=30, c=fs.INK_2, label="converged")
    ax.scatter([], [], s=30, facecolors=fs.SURFACE, edgecolors=fs.INK_2, linewidths=1.4,
               label="not yet converged")
    ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.3)
    fs.panel_letter(ax, "A", "Score against rank q")
    ax.set_xlim(-1.5, df.q.max() * 1.22)      # room for the direct labels
    ax.set_xlabel("latent dimension q"); ax.set_ylabel(GAIN_LABEL)

    # B — score against group count, one line per q
    ax = axes[1]; fs.tidy(ax)
    qs = sorted(df.q.unique())
    for i, q in enumerate(qs):
        g = df[df.q == q].sort_values("K")
        if len(g) < 2:
            continue
        shade = fs.AGE_RAMP(0.15 + 0.75 * i / max(len(qs) - 1, 1))
        # Join only consecutive K: with the grid still filling, a line drawn
        # straight from K=1 to K=3 would imply a K=2 result that does not exist.
        Kv, yv = g.K.values, g.gain_per_infant.values
        start = 0
        for b in range(1, len(Kv) + 1):
            if b == len(Kv) or Kv[b] != Kv[b - 1] + 1:
                if b - start > 1:
                    ax.plot(Kv[start:b], yv[start:b], color=shade, lw=2.0, zorder=2)
                start = b
        _points(ax, Kv, yv, g.converged.values, shade)
        # Label at K=1, where the lines are well separated; at large K they
        # converge and labels there stack on top of each other.
        ax.annotate(f"q={q}", (g.K.iloc[0], g.gain_per_infant.iloc[0]),
                    xytext=(-7, 0), textcoords="offset points", fontsize=7.5,
                    color=shade, va="center", ha="right", fontweight="bold")
    fs.panel_letter(ax, "B", "Score against group count K")
    ax.set_xlabel("number of groups K"); ax.set_ylabel(GAIN_LABEL)
    ax.set_xticks(sorted(df.K.unique()))
    ax.set_xlim(df.K.min() - 0.75, df.K.max() + 0.3)

    # C — collapse diagnostic
    ax = axes[2]; fs.tidy(ax)
    sub = df[df.K > 1]
    FLOOR = 1e-3        # a fully collapsed group is ~1e-47; without a floor one
                        # point would stretch the axis over forty decades
    if len(sub):
        share = sub.min_group_share.values * 100
        collapsed = share < FLOOR
        drawn = np.maximum(share, FLOOR)
        ax.scatter(sub.K[~collapsed], drawn[~collapsed], c=fs.SERIES[0], s=48,
                   linewidths=0.5, edgecolors=fs.SURFACE, zorder=3)
        if collapsed.any():
            ax.scatter(sub.K[collapsed], drawn[collapsed], marker="v", s=70,
                       c=fs.SERIES[1], linewidths=0.5, edgecolors=fs.SURFACE, zorder=3)
        # Label only the points that carry information: anything collapsed or
        # within sight of the 2% line. Labelling every retained point stacks
        # them illegibly at the top of the panel.
        # Retained points near the line sit almost on top of one another, so
        # each K gets one combined label.
        def _label(rows, y, color):
            for K, g in rows.groupby("K"):
                ax.annotate("q " + ",".join(str(v) for v in sorted(g.q)), (K, y(g)),
                            xytext=(0, 9), textcoords="offset points", fontsize=7,
                            ha="center", va="bottom", color=color)
        _label(sub[collapsed], lambda g: FLOOR, fs.SERIES[1])
        _label(sub[~collapsed & (drawn < 5)],
               lambda g: float(g.min_group_share.max() * 100), fs.INK_2)
        ok = drawn[~collapsed & (drawn >= 5)]
        if len(ok):
            ax.annotate(f"{len(ok)} specifications kept every group\n"
                        f"(smallest {ok.min():.0f}–{ok.max():.0f}% of infants)",
                        (0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
                        fontsize=7.5, color=fs.INK_2)
        ax.axhline(2, color=fs.INK_MUTED, lw=1.3, ls=(0, (4, 2)), zorder=1)
        ax.annotate("2% — a group has collapsed below this", (0.03, 2),
                    xytext=(0, -11), textcoords="offset points",
                    xycoords=("axes fraction", "data"), ha="left",
                    fontsize=7.5, color=fs.INK_2)
        ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.35, 400)
        ax.set_xticks(sorted(sub.K.unique()))
        ax.set_xlim(sub.K.min() - 0.4, sub.K.max() + 0.75)
        ax.scatter([], [], c=fs.SERIES[0], s=48, label="smallest group retained")
        ax.scatter([], [], c=fs.SERIES[1], s=70, marker="v", label="group vanished")
        ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.3), fontsize=7.5,
                  handletextpad=0.4)
    fs.panel_letter(ax, "C", "Did any group collapse?")
    ax.set_xlabel("number of groups K"); ax.set_ylabel("smallest group share (%)")

    n_done = len(df)
    fig.suptitle("Model selection: how many groups, and how many axes of variation?",
                 x=0.062, y=0.975, ha="left", fontsize=11, color=fs.INK)
    n_inf = int(df.n_heldout.iloc[0])
    fig.text(0.062, 0.925,
             f"{n_done} of 35 specifications · held-out score per infant (5-fold CV, n = {n_inf}), "
             "relative to a baseline that knows only each measure's mean and SD · higher is better · "
             "a gap of 1 = data e ≈ 2.7× more plausible",
             ha="left", fontsize=8, color=fs.INK_2)
    p = OUT / "fig2_model_selection.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    fs.apply()
    for fn in (fig_data_space, fig_model_selection):
        p = fn()
        if p:
            print(f"wrote {p}")
