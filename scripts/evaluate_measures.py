"""Is communicability shaky, and would another nodal measure do better?

    python scripts/evaluate_measures.py

For the 569 training infants, every regional measure is computed on both the
primary (per-bin 25% + MST) and the unthresholded networks, and scored on:

  robustness   for each region, the correlation across infants between its
               values in the two network variants, after removing the age
               trend from each (median over regions). A measure that depends
               on preprocessing choices ranks infants differently in the two.
  redundancy   median |correlation| with strength and with clustering in the
               primary variant, after removing age. High = little new
               information.
  shape        median |skewness| and the fraction of exact zeros.

Candidates beyond the current three: weighted betweenness centrality (lengths
1/w; Brandes via bctpy), nodal efficiency (mean inverse shortest-path length)
and eigenvector centrality. Correlations are Spearman, so skewed measures are
not penalised for their scale.

It also reports the communality of each current measure in the selected model:
the share of each coordinate's variance carried by the shared axes rather than
the region-specific term psi.

Writes results/measures/measure_evaluation.json and
figures/fig10_measure_evaluation.{png,pdf}.
"""

import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import bct
import paths
import figstyle as fs
from features import MEASURES
from model import splines
from scripts.fit_final import principal

OUT = paths.RESULTS / "measures"
NEW = ("betweenness", "nodal_efficiency", "eigenvector")
LABEL = {"strength": "strength", "log_clustering": "clustering", "communicability":
         "communicability", "betweenness": "betweenness", "nodal_efficiency":
         "nodal efficiency", "eigenvector": "eigenvector centrality"}


def new_measures(W):
    W = np.array(W, dtype=float)                                 # copy: the stack is memory-mapped read-only
    np.fill_diagonal(W, 0)
    L = np.where(W > 0, 1.0 / np.where(W > 0, W, 1), 0)        # lengths for path measures
    n = len(W)
    btw = bct.betweenness_wei(L) / ((n - 1) * (n - 2))
    D = bct.distance_wei(L)[0]
    with np.errstate(divide="ignore"):
        inv = np.where(np.isfinite(D) & (D > 0), 1.0 / D, 0.0)
    eff = inv.sum(1) / (n - 1)
    eig = bct.eigenvector_centrality_und(W)
    return np.stack([btw, eff, eig], -1)                         # (90, 3)


def compute_new(stack, idx):
    A = np.load(stack, mmap_mode="r")
    return np.array([new_measures(A[:, :, i]) for i in idx])     # (n, 90, 3)


def residualise(Y, B):
    """Remove the age trend from each column (penalised spline, as in the model)."""
    P = splines.difference_penalty(B.shape[1])
    coef = np.linalg.solve(B.T @ B + 1.0 * P, B.T @ Y)
    return Y - B @ coef


def spearman_cols(a, b):
    ra, rb = stats.rankdata(a, axis=0), stats.rankdata(b, axis=0)
    ra = (ra - ra.mean(0)) / ra.std(0); rb = (rb - rb.mean(0)) / rb.std(0)
    return (ra * rb).mean(0)


def main():
    import sys
    if "--figure-only" in sys.argv:
        r = json.loads((OUT / "measure_evaluation.json").read_text())
        fs.apply_style()
        figure(r["measures"], r["n"], r["q"])
        return
    t0 = time.time()
    coh = pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv")
    tr = coh[coh.in_primary & (coh.split == "train")]
    idx = tr.subject_index.values
    B = splines.basis(tr.scan_age.values)

    var = {}
    for name, stack, feat in (("primary", paths.STACK, "perbin25_mst"),
                              ("unthresholded", paths.STACK_UNTHRESH, "unthresholded")):
        old = np.load(paths.RESULTS / "features" / f"{feat}.npz")["X"][idx]   # (n, 90, 3)
        new = compute_new(stack, idx)
        var[name] = {m: old[:, :, j] for j, m in enumerate(MEASURES)}
        var[name].update({m: new[:, :, j] for j, m in enumerate(NEW)})
        print(f"{name}: measures computed ({time.time() - t0:.0f}s)", flush=True)

    res = {v: {m: residualise(x, B) for m, x in d.items()} for v, d in var.items()}
    out = {}
    for m in MEASURES + NEW:
        x, y = var["primary"][m], res["primary"][m]
        rob = spearman_cols(res["primary"][m], res["unthresholded"][m])
        rs = np.abs(spearman_cols(y, res["primary"]["strength"]))
        rc = np.abs(spearman_cols(y, res["primary"]["log_clustering"]))
        out[m] = dict(
            robustness_median=float(np.nanmedian(rob)), robustness_q25=float(np.nanpercentile(rob, 25)),
            with_strength=None if m == "strength" else float(np.nanmedian(rs)),
            with_clustering=None if m == "log_clustering" else float(np.nanmedian(rc)),
            median_abs_skew=float(np.nanmedian(np.abs(stats.skew(x, axis=0)))),
            frac_zero=float((x == 0).mean()),
            age_r2_median=float(np.nanmedian(1 - y.var(0) / np.maximum(x.var(0), 1e-300))))

    # communality of the current measures in the selected model
    sel = json.loads((paths.RESULTS / "selection.json").read_text())["selected"]
    d = np.load(paths.RESULTS / "final" / f"K1_q{sel['q']}.npz")
    L, _, _, _ = principal(d["W"][0], d["psi"][0])
    h2 = (L ** 2).sum(1) / ((L ** 2).sum(1) + d["psi"][0])
    for j, m in enumerate(MEASURES):
        out[m]["communality_median"] = float(np.median(h2[j::3]))
        out[m]["communality_top3_median"] = float(np.median(
            ((L[:, :3] ** 2).sum(1) / ((L ** 2).sum(1) + d["psi"][0]))[j::3]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measure_evaluation.json").write_text(json.dumps(dict(n=len(tr), q=sel["q"], measures=out), indent=2))
    print(json.dumps(out, indent=2))
    fs.apply_style()
    figure(out, len(tr), sel["q"])
    print(f"done in {time.time() - t0:.0f}s")


def figure(out, n, q):
    allm = list(MEASURES) + list(NEW)
    colours = [fs.MEASURE_COLOURS.get(m, c) for m, c in
               zip(allm, [None] * 3 + [fs.COLORS["sky_blue"], fs.COLORS["blue"], fs.COLORS["vermillion"]])]
    fig, axes = plt.subplots(2, 2, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(hspace=0.75, wspace=0.35, bottom=0.14, top=0.95)
    x = np.arange(len(allm))
    names = [LABEL[m] for m in allm]

    ax = axes[0, 0]
    ax.set_xlim(-0.6, len(allm) - 0.4); ax.set_ylim(0, 1.1)
    fs.rounded_bars(ax, x, [out[m]["robustness_median"] for m in allm], color=colours,
                    value_fmt="{:.2f}")
    ax.set_xticks(x, names, rotation=40, ha="right")
    ax.set_ylabel("thresholded vs unthresholded")
    fs.panel_title(ax, "A", "Robust to preprocessing?")

    ax = axes[0, 1]
    for k, (key, mk, lab) in enumerate((("with_strength", "o", "vs strength"),
                                        ("with_clustering", "s", "vs clustering"))):
        xs = [i for i, m in enumerate(allm) if out[m][key] is not None]
        ys = [out[allm[i]][key] for i in xs]
        ax.scatter(np.array(xs) + (k - 0.5) * 0.2, ys, marker=mk, s=45, zorder=3,
                   color=[colours[i] for i in xs], edgecolors=fs.TEXT_PRIMARY, linewidths=0.6,
                   label=lab)
    ax.set_ylim(0, 1.05); ax.set_xlim(-0.6, len(allm) - 0.4)
    ax.set_xticks(x, names, rotation=40, ha="right")
    ax.set_ylabel("|correlation|")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color="white",
                              markeredgecolor=fs.TEXT_PRIMARY, label="vs strength"),
                       Line2D([], [], marker="s", ls="none", color="white",
                              markeredgecolor=fs.TEXT_PRIMARY, label="vs clustering")],
              loc="upper right")
    fs.panel_title(ax, "B", "Redundant with the others?")

    ax = axes[1, 0]
    xm = np.arange(3)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.0)
    fs.rounded_bars(ax, xm, [out[m]["communality_median"] for m in MEASURES],
                    color=colours[:3], value_fmt="{:.2f}")
    ax.set_xticks(xm, names[:3], rotation=40, ha="right")
    ax.set_ylabel("shared variance (communality)")
    fs.panel_title(ax, "C", "Captured by the axes?")

    ax = axes[1, 1]
    ax.set_xlim(-0.6, len(allm) - 0.4)
    sk = [out[m]["median_abs_skew"] for m in allm]
    ax.set_ylim(0, max(sk) * 1.25)
    fs.rounded_bars(ax, x, sk, color=colours, value_fmt="{:.1f}")
    ax.set_xticks(x, names, rotation=40, ha="right")
    ax.set_ylabel("median |skewness|")
    fs.panel_title(ax, "D", "Distribution shape")

    fig.text(0.07, 0.0, f"{n} training infants. A, B: Spearman correlations across infants per "
             "region after removing the age trend; median over 90 regions.\n"
             f"C: share of each coordinate's variance carried by the shared axes in the selected "
             f"model (K = 1, q = {q}); median over regions.", fontsize=fs.NOTE_SIZE,
             color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "fig10_measure_evaluation")


if __name__ == "__main__":
    main()
