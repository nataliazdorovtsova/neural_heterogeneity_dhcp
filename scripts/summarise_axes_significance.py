"""Are the axes bigger than noise, and do they replicate in independent halves?

    python scripts/summarise_axes_significance.py

Reads the permutation null (scripts/permutation_null.py) and the split-half
replications (scripts/split_half.py) for the one-group, q = 15 model.

Parallel analysis: axis j is bigger than noise if its variance exceeds the
95th percentile of the j-th largest axis variance across permuted datasets.

Split-half: for each full-data axis, |cosine| between its matched axes in two
independently fitted halves. For scale, the same matching applied to random
loading vectors gives the "chance" level, simulated here.

Writes results/significance/axes_summary.json and
figures/fig8_axis_significance.{png,pdf}.
"""

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paths
import figstyle as fs
from scripts.fit_final import principal
from scripts.split_half import match, N_MATCH

SIG = paths.RESULTS / "significance"
Q = 15


def chance_cos(n_sim=200, D=270, rng=np.random.default_rng(5)):
    """|cos| of Hungarian-matched axes when both sets are random directions."""
    out = []
    for _ in range(n_sim):
        A = rng.standard_normal((D, Q)); Bm = rng.standard_normal((D, Q))
        _, c = match(A, Bm, N_MATCH)
        out.append(c)
    return np.array(out)


def main():
    full = np.load(paths.RESULTS / "final" / f"K1_q{Q}.npz")
    _, S2, _, _ = principal(full["W"][0], full["psi"][0])
    total = S2.sum() + full["psi"][0].sum()

    perm = [np.load(f) for f in sorted((SIG / "permutation").glob("chunk*.npz"))]
    null = np.concatenate([p["var"] for p in perm])                 # (n_perm, Q)
    null_conv = np.concatenate([p["converged"] for p in perm])
    thr = np.percentile(null, 95, axis=0)
    beats = S2 > thr
    n_sig = int(np.argmin(beats)) if not beats.all() else Q          # leading run

    splits = [np.load(f) for f in sorted((SIG / "split_half").glob("split*.npz"))]
    cos_ab = np.array([s["cos_ab"] for s in splits])                 # (n_split, N_MATCH)
    chance = chance_cos()

    summary = dict(
        q=Q, n_permutations=int(len(null)), permutations_converged=int(null_conv.sum()),
        axis_variance=S2.round(3).tolist(), axis_share_pct=(100 * S2 / total).round(2).tolist(),
        null_95th=thr.round(3).tolist(), null_max_share_pct=float(100 * null.max() / total),
        axes_beating_null=int(beats.sum()), leading_axes_beating_null=n_sig,
        n_splits=int(len(cos_ab)), split_half_median=np.median(cos_ab, 0).round(3).tolist(),
        split_half_min=cos_ab.min(0).round(3).tolist(),
        chance_cos_95th=np.percentile(chance, 95, axis=0).round(3).tolist(),
    )
    (SIG / "axes_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGURE_SIZES["double_column"])
    fig.subplots_adjust(wspace=0.3, bottom=0.3, top=0.88)

    ax = axes[0]
    x = np.arange(1, Q + 1)
    share, null_share = 100 * S2 / total, 100 * null / total
    ax.set_xlim(0.3, Q + 0.7); ax.set_ylim(0, share.max() * 1.25)
    fs.rounded_bars(ax, x, share, color=[fs.Q_COLOURS[Q] if b else fs.NEUTRAL_LIGHT_BLUE
                                         for b in beats])
    ax.plot(x, np.percentile(null_share, 95, axis=0), color=fs.COLORS["vermillion"], marker="s",
            markersize=5, zorder=4)
    ax.set_xticks(x[::2])
    ax.set_xlabel("axis (largest first)"); ax.set_ylabel("% of variation")
    fs.panel_title(ax, "A", "Bigger than noise?")
    ax.legend(handles=[Patch(color=fs.Q_COLOURS[Q], label="beats the null"),
                       Patch(color=fs.NEUTRAL_LIGHT_BLUE, label="does not"),
                       Line2D([], [], color=fs.COLORS["vermillion"], marker="s",
                              label="null, 95th percentile")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    ax = axes[1]
    xs = np.arange(1, N_MATCH + 1)
    for j in range(N_MATCH):
        ax.scatter(np.full(len(cos_ab), xs[j]) + np.linspace(-0.15, 0.15, len(cos_ab)),
                   cos_ab[:, j], s=22, color=fs.COLORS["blue"], edgecolors="white",
                   linewidths=0.4, zorder=3)
    ax.plot(xs, np.median(cos_ab, 0), color=fs.TEXT_PRIMARY, lw=2.0, marker="D", markersize=5,
            zorder=4)
    ax.plot(xs, np.percentile(chance, 95, axis=0), color=fs.NEUTRAL_GREY, ls="--", lw=1.5)
    fs.reference_line(ax, 0.9)
    ax.set_ylim(0, 1.05); ax.set_xticks(xs)
    ax.set_xlabel("axis of the full-data model"); ax.set_ylabel("similarity (|cos|)")
    fs.panel_title(ax, "B", "Same axes in each half?")
    ax.legend(handles=[Line2D([], [], color=fs.COLORS["blue"], marker="o", ls="none",
                              label="one random split"),
                       Line2D([], [], color=fs.TEXT_PRIMARY, marker="D", label="median"),
                       Line2D([], [], color=fs.NEUTRAL_GREY, ls="--", label="chance (95th pct)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.text(0.07, -0.03, f"One-group model, q = {Q}. A: {len(null)} permutations, each "
             "coordinate shuffled independently after removing the age trajectory.\n"
             f"B: {len(cos_ab)} random halves stratified by age and prematurity; "
             "each half fitted on its own.", fontsize=fs.NOTE_SIZE,
             color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "fig8_axis_significance")


if __name__ == "__main__":
    main()
