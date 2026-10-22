"""How much of the axis structure is "real", and what is the tail made of?

    python scripts/characterise_tail.py

Two questions about the selected model (K = 1, q = 70).

1. What do the axes encode? The covariance an axis block implies between pairs of
   coordinates (region x measure) is split by the anatomical relation of the pair:
     within region   the same region's different measures
     mirror          a region and its homologue in the other hemisphere
     neighbour       regions that touch in the connectome atlas (not mirrors)
     distant         everything else
   For each block of axes (1-3, 4-8, 9-20, 21-70), the share of its squared
   implied correlation falling in each category is compared with the share of
   pairs in that category (what an evenly spread block would give). A block that
   mostly encodes local, anatomically expected coupling is enriched for the first
   three categories. The age-detrended data's own correlation matrix is the
   reference.

2. Does the three-axis conclusion hold at q = 70? Split-half replication of the
   q = 70 model (scripts/split_half.py --q 70): per-axis similarity and subspace
   overlap per block, against chance.

Writes results/significance/tail_summary.json and
figures/fig9_tail_structure.{png,pdf}.
"""

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

import paths
import atlas
import figstyle as fs
from features import standardise
from model import splines
from scripts.run_spec import load, age_for, CLOCK
from scripts.fit_final import principal

Q = 70
BLOCKS = {"1–3": slice(0, 3), "4–8": slice(3, 8), "9–20": slice(8, 20), "21–70": slice(20, 70)}
CATS = ("within region", "mirror", "neighbour", "distant")
CAT_COLOURS = [fs.COLORS["vermillion"], fs.COLORS["orange"], fs.COLORS["sky_blue"],
               fs.NEUTRAL_GREY]


def region_adjacency():
    lab, _ = atlas.load_atlas()
    A = np.zeros((91, 91), bool)
    for ax in range(3):
        for step in (1, 2):                          # 0.5 mm voxels: touching within 1 mm
            a = np.take(lab, range(0, lab.shape[ax] - step), axis=ax)
            b = np.take(lab, range(step, lab.shape[ax]), axis=ax)
            m = (a != b) & (a > 0) & (b > 0)
            A[a[m], b[m]] = True
    A = A | A.T
    return A[1:, 1:]


def pair_categories(adj):
    D = 270
    r = np.arange(D) // 3
    ri, rj = np.meshgrid(r, r, indexing="ij")
    cat = np.full((D, D), 3)
    cat[adj[ri, rj]] = 2
    cat[(ri // 2 == rj // 2) & (ri != rj)] = 1       # AAL pairs L/R as 2k, 2k+1
    cat[ri == rj] = 0
    iu = np.triu_indices(D, 1)
    return cat[iu], iu


def shares(R, cat, iu):
    e = R[iu] ** 2
    return np.array([e[cat == c].sum() for c in range(4)]) / e.sum()


def split_half_summary():
    files = sorted((paths.RESULTS / "significance" / f"split_half_q{Q}").glob("split*.npz"))
    if not files:
        return None
    sp = [np.load(f) for f in files]
    cos = np.array([s["cos_ab"] for s in sp])
    blocks = {}
    for name, sl in BLOCKS.items():
        k = sl.stop - sl.start
        vals = []
        for s in sp:
            Qa, _ = np.linalg.qr(s["LA_all"][:, sl]); Qb, _ = np.linalg.qr(s["LB_all"][:, sl])
            vals.append(float(np.mean(np.linalg.svd(Qa.T @ Qb, compute_uv=False) ** 2)))
        blocks[name] = dict(median=float(np.median(vals)), min=float(np.min(vals)), chance=k / 270)
    conv = np.array([s["converged"] for s in sp])
    return dict(n_splits=len(sp), all_converged=bool(conv.all()),
                cos_median=np.median(cos, 0).tolist(), cos_min=cos.min(0).tolist(),
                blocks=blocks)


def main():
    d = np.load(paths.RESULTS / "final" / f"K1_q{Q}.npz")
    L, S2, _, _ = principal(d["W"][0], d["psi"][0])
    sd = np.sqrt((L ** 2).sum(1) + d["psi"][0])
    cat, iu = pair_categories(region_adjacency())
    chance = np.bincount(cat, minlength=4) / len(cat)

    coh, Xraw = load()
    X, *_ = standardise(Xraw)
    resid = X - splines.basis(age_for(coh, CLOCK)) @ d["beta"][0]
    data_share = shares(np.corrcoef(resid.T), cat, iu)

    comp = {}
    for name, sl in BLOCKS.items():
        R = (L[:, sl] @ L[:, sl].T) / np.outer(sd, sd)
        comp[name] = shares(R, cat, iu)
    total = S2.sum() + d["psi"][0].sum()
    block_var = {n: float(100 * S2[sl].sum() / total) for n, sl in BLOCKS.items()}

    summary = dict(
        q=Q, categories=CATS, pairs_per_category=np.bincount(cat, minlength=4).tolist(),
        chance_share=chance.round(4).tolist(), data_share=data_share.round(4).tolist(),
        block_share={n: v.round(4).tolist() for n, v in comp.items()},
        block_enrichment={n: (v / chance).round(2).tolist() for n, v in comp.items()},
        block_local_share={n: float(v[:3].sum()) for n, v in comp.items()},
        chance_local_share=float(chance[:3].sum()),
        block_variance_pct=block_var,
        split_half_q70=split_half_summary(),
    )
    out = paths.RESULTS / "significance" / "tail_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    fs.apply_style()
    figure(summary)


def figure(s):
    fig, axes = plt.subplots(2, 2, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(hspace=0.62, wspace=0.32, bottom=0.17, top=0.95)

    # A: composition of each block's implied correlation, vs chance and the data
    ax = axes[0, 0]
    cols = ["chance", "data"] + list(BLOCKS)
    comps = [s["chance_share"], s["data_share"]] + [s["block_share"][b] for b in BLOCKS]
    ax.set_xlim(-0.6, len(cols) - 0.4); ax.set_ylim(0, 1.05)
    for i, c in enumerate(comps):
        # the full bar carries the rounded top; lower segments are drawn over it
        fs.rounded_bars(ax, [i], [1.0], width=0.7, color=CAT_COLOURS[3])
        cum = 0.0
        for k in range(3):
            ax.add_patch(Rectangle((i - 0.35, cum), 0.7, c[k], facecolor=CAT_COLOURS[k],
                                   edgecolor=fs.TEXT_PRIMARY, linewidth=0.6, zorder=3))
            cum += c[k]
    ax.set_xticks(range(len(cols)), ["chance", "data"] + [f"axes\n{b}" for b in BLOCKS])
    ax.set_ylabel("share of squared correlation")
    fs.panel_title(ax, "A", "What the axes encode")

    # B: enrichment of local coupling per block
    ax = axes[0, 1]
    x = np.arange(len(BLOCKS))
    for k in range(3):
        y = [s["block_enrichment"][b][k] for b in BLOCKS]
        ax.plot(x, y, color=CAT_COLOURS[k], marker="osD"[k], markersize=7, lw=2.0)
    ax.set_yscale("log")
    fs.reference_line(ax, 1.0)
    ax.set_xticks(x, [f"axes {b}" for b in BLOCKS])
    ax.set_ylabel("enrichment over chance")
    fs.panel_title(ax, "B", "How local is each block?")

    sh = s["split_half_q70"]
    ax = axes[1, 0]
    if sh:
        n = len(sh["cos_median"])
        xs = np.arange(1, n + 1)
        ax.plot(xs, sh["cos_median"], color=fs.TEXT_PRIMARY, marker="D", lw=2.0)
        ax.plot(xs, sh["cos_min"], color=fs.NEUTRAL_GREY, marker="o", ls="--")
        fs.reference_line(ax, 0.7)
        ax.set_ylim(0, 1.05); ax.set_xticks(xs)
    ax.set_xlabel("axis of the q = 70 model"); ax.set_ylabel("similarity (|cos|)")
    fs.panel_title(ax, "C", "Replication, axis by axis")

    ax = axes[1, 1]
    if sh:
        xb = np.arange(len(BLOCKS))
        ax.set_xlim(-0.6, len(BLOCKS) - 0.4); ax.set_ylim(0, 1.05)
        fs.rounded_bars(ax, xb, [sh["blocks"][b]["median"] for b in BLOCKS],
                        color=fs.Q_COLOURS[25], value_fmt="{:.2f}")
        ax.scatter(xb, [sh["blocks"][b]["chance"] for b in BLOCKS], marker="_", s=600,
                   color=fs.COLORS["vermillion"], linewidths=2.5, zorder=4)
        ax.set_xticks(xb, [f"axes {b}" for b in BLOCKS])
    ax.set_ylabel("subspace overlap")
    fs.panel_title(ax, "D", "Replication, by block")

    handles = [Patch(color=c, label=l) for c, l in zip(CAT_COLOURS, CATS)] + [
        Line2D([], [], color=fs.TEXT_PRIMARY, marker="D", label="C: median of 10 splits"),
        Line2D([], [], color=fs.NEUTRAL_GREY, marker="o", ls="--", label="C: worst split"),
        Line2D([], [], color=fs.COLORS["vermillion"], lw=2.5, label="D: chance")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.035), ncol=4,
               handlelength=1.4, columnspacing=1.0)
    fig.text(0.06, 0.02, "Selected model, K = 1, q = 70. A, B: pairs of coordinates (region × "
             "measure) grouped by anatomical relation; neighbours touch in the connectome atlas.\n"
             "C, D: 10 random halves (stratified by age and prematurity), each fitted on its own; "
             "D: mean cos² of principal angles between the halves' subspaces.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "fig9_tail_structure")


if __name__ == "__main__":
    main()
