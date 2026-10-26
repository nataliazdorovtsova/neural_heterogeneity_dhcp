"""Does axis 1 survive without Onnela's per-infant max-weight normalisation?

    python scripts/check_clustering_norm.py

AUDIT.md finding A: Onnela clustering divides by each infant's maximum edge
weight, which shifts every region's log clustering by -log(max W). Axis 1
("clustering everywhere") correlates -0.62 with the age-adjusted maximum
weight. This compares the primary model with the same model fitted to
clustering computed without that normalisation (feature set perbin25_mst_cl,
scripts/extract_alt_features.py), on the same infants:

  pattern     axes matched to the primary axes (|cos|, all 270 coordinates)
  infants     correlation of matched axis scores
  max weight  correlation of axis scores with the age-adjusted log max weight
  groups      K = 1 vs K = 2 at q = 15 and 25 (paired per-infant difference)
  replicable  split-half similarity of the new model's axes 1-3

Writes results/significance/clustering_norm.json and
figures/fig14_clustering_norm.{png,pdf}.
"""

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import paths
import figstyle as fs
from model import splines
from scripts.run_spec import load
from scripts.compare_variant_axes import axes_of, compare
from scripts.compare_measure_sets import k1_vs_k2, split_half
from scripts.split_half import match

V = "perbin25_mst_cl"
NEW = fs.COLORS["sky_blue"]
OLD = fs.COLORS["blue"]


def age_adjusted_log_max_weight(coh):
    A = np.load(paths.STACK, mmap_mode="r")
    y = np.log(np.array([A[:, :, i].max() for i in coh.subject_index.values]))
    B = splines.basis(coh.scan_age.values)
    P = splines.difference_penalty(B.shape[1])
    return y - B @ np.linalg.solve(B.T @ B + P, B.T @ y)


def main():
    coh, _ = load()
    mw = age_adjusted_log_max_weight(coh)
    out = {}
    for q in (15, 70):
        f = paths.RESULTS / "final" / f"K1_q{q}_{V}.npz"
        if not f.exists():
            continue
        r = compare(q, V, coh)
        a = axes_of(paths.RESULTS / "final" / f"K1_q{q}.npz")
        b = axes_of(f)
        idx, _ = match(a["L"], b["L"], 5)
        r["score_corr"] = [float(abs(np.corrcoef(a["scores"][:, j], b["scores"][:, idx[j]])[0, 1]))
                           for j in range(5)]
        r["max_weight_corr_primary"] = [float(np.corrcoef(a["scores"][:, j], mw)[0, 1]) for j in range(5)]
        # signed to the primary axis, so the two rows are comparable
        sg = np.sign((a["L"][:, :5] * b["L"][:, idx]).sum(0))
        r["max_weight_corr_new"] = [float(sg[j] * np.corrcoef(b["scores"][:, idx[j]], mw)[0, 1])
                                    for j in range(5)]
        # the new model's own largest axes and their max-weight correlation
        r["new_own_axes"] = dict(share=b["share"][:5].round(2).tolist(),
                                 max_weight_corr=[float(np.corrcoef(b["scores"][:, j], mw)[0, 1])
                                                  for j in range(5)])
        out[f"q{q}"] = r
    out["K1_vs_K2"] = k1_vs_k2(V)
    out["K1_vs_K2_primary"] = {r["q"]: dict(z=r["z"], per_infant=r["diff"] / 569, se=r["se"])
                               for r in json.loads((paths.RESULTS / "selection.json").read_text())["K1_vs_K2"]
                               if r["q"] in (15, 25)}
    out["split_half"] = split_half(f"split_half_q15_{V}")
    (paths.RESULTS / "significance" / "clustering_norm.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    fs.apply_style()
    figure(out)


def figure(out):
    r = out.get("q70") or out["q15"]
    fig, axes = plt.subplots(2, 2, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(hspace=0.6, wspace=0.32, bottom=0.17, top=0.95)
    x = np.arange(1, 6)

    ax = axes[0, 0]
    for qk, m, ls in (("q15", "o", "--"), ("q70", "s", "-")):
        if qk in out:
            ax.plot(x, out[qk]["cos"][:5], color=fs.Q_COLOURS[15 if qk == "q15" else 25], marker=m,
                    ls=ls, label=f"pattern, {qk.replace('q', 'q = ')}")
    ax.plot(x, r["score_corr"], color=fs.TEXT_PRIMARY, marker="D", label=f"same infants (q = {r['q']})")
    fs.reference_line(ax, 0.9)
    ax.set_ylim(0, 1.05); ax.set_xticks(x)
    ax.set_xlabel("axis of the primary model"); ax.set_ylabel("similarity")
    ax.legend(loc="lower left")
    fs.panel_title(ax, "A", "Same axes?")

    ax = axes[0, 1]
    ax.set_xlim(0.3, 5.7)
    ax.set_ylim(0, max(max(r["share_primary"][:5]), max(r["share_variant"][:5])) * 1.3)
    fs.rounded_bars(ax, x - 0.2, r["share_primary"][:5], width=0.38, color=OLD)
    fs.rounded_bars(ax, x + 0.2, r["share_variant"][:5], width=0.38, color=NEW)
    ax.set_xticks(x)
    ax.set_xlabel("axis (primary order)"); ax.set_ylabel("% of variation beyond age")
    fs.panel_title(ax, "B", "Axis sizes")

    ax = axes[1, 0]
    ax.set_xlim(0.3, 5.7); ax.set_ylim(-0.8, 0.8)
    fs.rounded_bars(ax, x - 0.2, r["max_weight_corr_primary"], width=0.38, color=OLD)
    fs.rounded_bars(ax, x + 0.2, r["max_weight_corr_new"], width=0.38, color=NEW)
    fs.reference_line(ax, 0)
    ax.set_xticks(x)
    ax.set_xlabel("axis (primary order)"); ax.set_ylabel("r with max edge weight (age-adj.)")
    fs.panel_title(ax, "C", "Tied to the max weight?")

    ax = axes[1, 1]
    qs = (15, 25)
    xs = np.arange(len(qs))
    for off, kk, col in ((-0.2, out["K1_vs_K2_primary"], OLD), (0.2, out["K1_vs_K2"], NEW)):
        v = [kk.get(q, kk.get(str(q))) for q in qs]
        if all(v):
            vals = [e["per_infant"] for e in v]
            errs = [1.96 * e["se"] / 569 for e in v]
            fs.rounded_bars(ax, xs + off, vals, width=0.38, color=col)
            ax.errorbar(xs + off, vals, yerr=errs, fmt="none", ecolor=fs.TEXT_PRIMARY,
                        elinewidth=0.8, capsize=3, zorder=4)
    ax.set_xlim(-0.6, len(qs) - 0.4)
    fs.reference_line(ax, 0)
    ax.set_xticks(xs, [f"q = {q}" for q in qs])
    ax.set_ylabel("K=1 minus K=2, per infant")
    fs.panel_title(ax, "D", "Still one group?")

    fig.legend(handles=[Patch(color=OLD, label="primary clustering (Onnela, max-normalised)"),
                        Patch(color=NEW, label="clustering without max normalisation")],
               loc="lower left", bbox_to_anchor=(0.06, 0.035), ncol=1)
    fig.text(0.06, 0.02, "One-group model refitted with the only change being the clustering "
             "normalisation (same infants, 20 starts, 1000 sweeps).\nB, C: axes paired to the "
             f"primary axes; q = {r['q']}. D: within each feature set, 95% intervals.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "fig14_clustering_norm")


if __name__ == "__main__":
    main()
