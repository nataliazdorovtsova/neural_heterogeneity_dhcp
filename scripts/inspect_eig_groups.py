"""What separates the two groups when eigenvector centrality is a measure?

    python scripts/inspect_eig_groups.py

With strength + clustering + eigenvector centrality, two groups beat one at
q = 15 (SUPPLEMENT S27). Two explanations compete:

  more axes needed   the set has extra dimensions, and groups stand in for
                     within-group covariance the model lacks (as for the
                     primary set at q <= 5)
  hub switching      eigenvector centrality can jump between regions when two
                     compete for the network's leading eigenvector, giving
                     near-discrete values that a mixture can fit as "groups"

This fits the K = 2, q = 15 model's output (scripts/fit_final.py --K 2 --q 15
--variant perbin25_mst_eig) and asks:

  1. how many infants are in each group, and how cleanly are they assigned?
  2. which measures and regions differ between the group centres (at 40 weeks
     PMA, in SD units)?
  3. is membership explained by which region has the highest eigenvector
     centrality in each infant (the dominant hub)?
  4. do the regions with bimodal eigenvector centrality carry the group
     differences?
  5. does membership split infants along the large eigenvector axis of the
     one-group model (eig axis 1), and are those scores bimodal?
  6. do the groups differ in age, prematurity, sex, motion or density?

Writes results/significance/eig_groups.json and
figures/fig13_eig_groups.{png,pdf}.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import adjusted_rand_score

import paths
import atlas
import figstyle as fs
from model import splines
from scripts.run_spec import load
from scripts.fit_final import principal

V = "perbin25_mst_eig"
MEAS = ("strength", "clustering", "eigenvector")


def bimodality(x):
    """Sarle's bimodality coefficient per column; > 0.555 suggests bimodality."""
    n = len(x)
    g, k = stats.skew(x, axis=0), stats.kurtosis(x, axis=0)
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def cramers_v(a, b):
    t = pd.crosstab(a, b).values
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def main():
    coh, Xraw = load(V)
    d2 = np.load(paths.RESULTS / "final" / f"K2_q15_{V}.npz")
    d1 = np.load(paths.RESULTS / "final" / f"K1_q15_{V}.npz")
    resp = d2["resp"]                                           # (N, 2)
    z = resp.argmax(1)
    sizes = np.bincount(z, minlength=2)
    hard = float((resp.max(1) > 0.9).mean())

    # 2. centre difference at 40 weeks PMA, in standardised units
    B40 = splines.basis(np.array([40.0]))[0]
    centres = np.einsum("m,kmd->kd", B40, d2["beta"])          # (2, 270)
    diff = centres[1] - centres[0]
    e = diff ** 2
    by_measure = (np.array([e[m::3].sum() for m in range(3)]) / e.sum()).round(3)
    top = np.argsort(-np.abs(diff))[:10]

    # 3. dominant hub (raw eigenvector centrality) vs group
    ev = Xraw.reshape(len(Xraw), 90, 3)[:, :, 2]
    hub = ev.argmax(1)
    hub_counts = pd.Series(hub).value_counts()
    v_hub = cramers_v(hub, z)
    ari_hub = adjusted_rand_score(hub, z)
    top_hubs = hub_counts.index[:6]
    hub_table = {atlas.NAMES[h]: [int(((hub == h) & (z == k)).sum()) for k in range(2)] for h in top_hubs}

    # 4. bimodal regions vs group difference on eigenvector coordinates
    bc = bimodality(ev)
    ev_diff = np.abs(diff[2::3])
    r_bc = float(stats.spearmanr(bc, ev_diff)[0])

    # 4b. which hemisphere holds the eigenvector centrality (AAL: even index = left)
    left = ev[:, 0::2].sum(1) / ev.sum(1)

    # 5. the one-group model's eigenvector axis
    L1, S2, Vt, sign = principal(d1["W"][0], d1["psi"][0])
    s1 = (d1["scores"][:, 0, :] @ Vt.T * sign)[:, 0]
    bc_axis = float(bimodality(s1[:, None])[0])
    r_axis = float(abs(stats.pointbiserialr(z, s1)[0]))

    # 6. covariates by group
    cov = {"PMA at scan": coh.scan_age.values, "GA at birth": coh.birth_age.values,
           "age since birth": coh.pna.values, "head motion": coh.qc_dmri_eddy_rel_motion.values,
           "network density": coh.density_thr.values}
    covs = {}
    for name, v in cov.items():
        ok = np.isfinite(v)
        a, b = v[ok & (z == 0)], v[ok & (z == 1)]
        covs[name] = dict(mean=[float(a.mean()), float(b.mean())],
                          p=float(stats.mannwhitneyu(a, b).pvalue))
    for name, v in {"preterm": coh.preterm.values.astype(bool),
                    "female": (coh.sex == "female").values}.items():
        t = pd.crosstab(v, z).values
        covs[name] = dict(share=[float(v[z == k].mean()) for k in range(2)],
                          p=float(stats.chi2_contingency(t)[1]))

    out = dict(
        sizes=sizes.tolist(), assigned_with_p_over_0_9=hard,
        centre_difference_share_by_measure=dict(zip(MEAS, by_measure.tolist())),
        top_differences=[dict(region=atlas.NAMES[i // 3], measure=MEAS[i % 3], diff_sd=float(diff[i]))
                         for i in top],
        dominant_hub=dict(n_distinct=int(len(hub_counts)), cramers_v=v_hub, ari=ari_hub,
                          table_top6=hub_table),
        bimodality=dict(regions_bc_over_0_555=float((bc > 0.555).mean()),
                        spearman_bc_vs_group_difference=r_bc),
        eig_axis1=dict(bimodality_coefficient=bc_axis, abs_corr_with_group=r_axis),
        left_hemisphere_share=dict(mean_by_group=[float(left[z == k].mean()) for k in range(2)],
                                   abs_corr_with_group=float(abs(stats.pointbiserialr(z, left)[0])),
                                   range=[float(left.min()), float(left.max())]),
        covariates=covs,
    )
    (paths.RESULTS / "significance" / "eig_groups.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    fs.apply_style()
    figure(out, s1, z, left)


def figure(out, s1, z, left):
    gcol = [fs.COLORS["blue"], fs.COLORS["vermillion"]]
    fig, axes = plt.subplots(2, 2, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(hspace=0.6, wspace=0.32, bottom=0.17, top=0.95)

    ax = axes[0, 0]
    bins = np.linspace(s1.min(), s1.max(), 35)
    for k in range(2):
        ax.hist(s1[z == k], bins=bins, color=gcol[k], alpha=0.75, edgecolor=fs.TEXT_PRIMARY,
                linewidth=0.4)
    ax.set_xlabel("eigenvector-axis score"); ax.set_ylabel("infants")
    fs.panel_title(ax, "A", "Groups along the big axis")

    ax = axes[0, 1]
    tab = out["dominant_hub"]["table_top6"]
    names = list(tab)
    x = np.arange(len(names))
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, max(sum(v) for v in tab.values()) * 1.15)
    tot = np.array([sum(tab[n]) for n in names], float)
    fs.rounded_bars(ax, x, tot, width=0.7, color=gcol[1])
    for i, n in enumerate(names):
        ax.add_patch(plt.Rectangle((i - 0.35, 0), 0.7, tab[n][0], facecolor=gcol[0],
                                   edgecolor=fs.TEXT_PRIMARY, linewidth=0.6, zorder=3))
    ax.set_xticks(x, [n.replace("_", " ") for n in names], rotation=40, ha="right")
    ax.set_ylabel("infants whose top hub it is")
    fs.panel_title(ax, "B", "Dominant hub by group")

    ax = axes[1, 0]
    shares = [out["centre_difference_share_by_measure"][m] for m in MEAS]
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.05)
    fs.rounded_bars(ax, np.arange(3), shares, color=[fs.COLORS["orange"], fs.COLORS["bluish_green"],
                                                     fs.COLORS["vermillion"]], value_fmt="{:.2f}")
    ax.set_xticks(np.arange(3), MEAS)
    ax.set_ylabel("share of squared centre difference")
    fs.panel_title(ax, "C", "What differs between groups")

    ax = axes[1, 1]
    bins = np.linspace(0, 1, 26)
    for k in range(2):
        ax.hist(left[z == k], bins=bins, color=gcol[k], alpha=0.75, edgecolor=fs.TEXT_PRIMARY,
                linewidth=0.4)
    fs.reference_line(ax, 0.5, axis="x")
    ax.set_xlabel("share of eigenvector centrality in left hemisphere"); ax.set_ylabel("infants")
    fs.panel_title(ax, "D", "Which hemisphere leads?")

    fig.legend(handles=[Patch(color=gcol[k], label=f"group {k + 1} (n = {out['sizes'][k]})")
                        for k in range(2)], loc="lower left", bbox_to_anchor=(0.06, 0.035), ncol=2)
    hv = out["dominant_hub"]
    fig.text(0.06, 0.02, "Strength + clustering + eigenvector centrality, K = 2, q = 15, all 569 "
             "training infants.\nB: Cramér's V between dominant hub and group = "
             f"{hv['cramers_v']:.2f}. C: group-centre differences at 40 weeks PMA.\n"
             "D: dashed line = an even split between hemispheres.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "fig13_eig_groups")


if __name__ == "__main__":
    main()
