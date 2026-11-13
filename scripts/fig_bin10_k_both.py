"""One group vs two, with and without the 7 orbitofrontal-low infants (step 2).

    python scripts/fig_bin10_k_both.py      # after compare_models for bin10 and bin10_no7

Reads results/{bin10,bin10_no7}/selection.json; writes figures/bin10/b5b_k_both.
"""

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paths
import figstyle as fs

RUNS = {"bin10": ("all training infants", fs.COLORS["blue"], "o"),
        "bin10_no7": ("without the 7 low infants", fs.COLORS["vermillion"], "s")}


def main():
    fs.apply_style()
    fig, ax = plt.subplots(figsize=fs.FIGURE_SIZES["single_column"])
    handles = []
    for k, (run, (label, col, mk)) in enumerate(RUNS.items()):
        s = json.loads((paths.PROJECT / "results" / run / "selection.json").read_text())
        kk = s["K1_vs_K2"]; n = s["n_infants"]
        q = np.array([r["q"] for r in kk]) + (k - 0.5) * 0.25
        v = np.array([r["diff"] for r in kk]) / n
        e = 1.96 * np.array([r["se"] for r in kk]) / n
        ax.errorbar(q, v, yerr=e, fmt=mk, color=col, ecolor=col, elinewidth=1, capsize=3,
                    ms=7, mec=fs.TEXT_PRIMARY, mew=0.5, zorder=3)
        handles.append(Line2D([], [], color=col, marker=mk, ls="", ms=7, label=f"{label} (n = {n})"))
    fs.reference_line(ax, 0)
    ax.set_xlabel("number of axes q"); ax.set_ylabel("1 group minus 2, per infant")
    ax.text(0.98, 0.04, "above 0: one group predicts better", transform=ax.transAxes, ha="right",
            fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    fs.save_figure(fig, paths.PROJECT / "figures" / "bin10" / "b5b_k_both")


if __name__ == "__main__":
    main()
