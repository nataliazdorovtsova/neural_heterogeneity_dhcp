"""Do the axes survive a different preprocessing choice?

    python scripts/compare_variant_axes.py --variant unthresholded --q 15 70

Compares the one-group model fitted to the primary networks (per-bin 25% +
MST) with the same model fitted to another feature set computed on the same
infants, e.g. the unthresholded networks (scripts/fit_final.py --variant).
Coordinates mean the same thing in both (the same region x measure), so axes
are compared directly:

  match        each primary axis 1-8 is paired with one axis of the other fit
               (Hungarian assignment on |cosine|); |cos| near 1 = same pattern
  subspace     mean cos^2 of principal angles for axes 1-3 and 4-8
  size         share of variation beyond age, per matched axis
  measures     share of each axis's squared loadings on strength, clustering
               and communicability
  correlates   correlation of matched axis scores with network density (of the
               thresholded network) and head motion, signed to the primary axis

Writes results/significance/variant_{variant}.json and
figures/fig11_preprocessing_check.{png,pdf}.
"""

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import paths
import figstyle as fs
from features import MEASURES
from scripts.run_spec import load
from scripts.fit_final import principal
from scripts.split_half import match

N = 8
MEASURE_LABEL = {"strength": "strength", "log_clustering": "clustering",
                 "communicability": "communicability"}


def axes_of(path):
    d = np.load(path)
    L, S2, Vt, sign = principal(d["W"][0], d["psi"][0])
    scores = (d["scores"][:, 0, :] @ Vt.T) * sign
    total = S2.sum() + d["psi"][0].sum()
    return dict(L=L, scores=scores, share=100 * S2 / total, ids=d["dhcp_id"])


def overlap(A, B):
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    return float(np.mean(np.linalg.svd(Qa.T @ Qb, compute_uv=False) ** 2))


def composition(L):
    e = L ** 2
    return np.array([e[m::3].sum(0) for m in range(3)]) / e.sum(0)        # (3, q)


def compare(q, variant, coh):
    ref = axes_of(paths.RESULTS / "final" / f"K1_q{q}.npz")
    alt = axes_of(paths.RESULTS / "final" / f"K1_q{q}_{variant}.npz")
    assert (ref["ids"] == alt["ids"]).all()
    idx, cos = match(ref["L"], alt["L"], N)
    La = alt["L"][:, idx]
    sgn = np.sign((ref["L"][:, :N] * La).sum(0))
    La, Sa = La * sgn, alt["scores"][:, idx] * sgn
    cov = {"network density": coh.density_thr.values, "head motion": coh.qc_dmri_eddy_rel_motion.values}
    corr = {}
    for name, v in cov.items():
        ok = np.isfinite(v)
        corr[name] = dict(
            primary=[float(np.corrcoef(ref["scores"][ok, j], v[ok])[0, 1]) for j in range(N)],
            variant=[float(np.corrcoef(Sa[ok, j], v[ok])[0, 1]) for j in range(N)])
    return dict(
        q=q, matched_axis=(idx + 1).tolist(), cos=cos.round(3).tolist(),
        subspace_1_3=overlap(ref["L"][:, :3], alt["L"][:, :3]),
        subspace_4_8=overlap(ref["L"][:, 3:8], alt["L"][:, 3:8]),
        share_primary=ref["share"][:N].round(2).tolist(),
        share_variant=alt["share"][idx].round(2).tolist(),
        composition_primary=composition(ref["L"][:, :N]).round(3).tolist(),
        composition_variant=composition(La).round(3).tolist(),
        correlates=corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="unthresholded")
    ap.add_argument("--q", type=int, nargs="+", default=[15, 70])
    a = ap.parse_args()
    coh, _ = load()
    out = {f"q{q}": compare(q, a.variant, coh) for q in a.q}
    f = paths.RESULTS / "significance" / f"variant_{a.variant}.json"
    f.write_text(json.dumps(dict(variant=a.variant, **out), indent=2))
    print(json.dumps(out, indent=2))
    fs.apply_style()
    figure(out, a.variant)


def figure(out, variant):
    qs = sorted(out, key=lambda k: int(k[1:]))
    main_q = qs[-1]
    r = out[main_q]
    fig, axes = plt.subplots(2, 2, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(hspace=0.55, wspace=0.32, bottom=0.14, top=0.95)
    x = np.arange(1, N + 1)

    ax = axes[0, 0]
    for k, (qk, m, ls) in enumerate(zip(qs, "os", ("--", "-"))):
        q = out[qk]["q"]
        ax.plot(x, out[qk]["cos"], color=fs.Q_COLOURS.get(q, fs.Q_COLOURS[25]), marker=m, ls=ls,
                markersize=6, label=f"q = {q}")
    fs.reference_line(ax, 0.9)
    ax.set_ylim(0, 1.05); ax.set_xticks(x)
    ax.set_xlabel("axis of the primary model"); ax.set_ylabel("similarity (|cos|)")
    ax.legend(loc="lower left")
    fs.panel_title(ax, "A", "Same axes?")

    ax = axes[0, 1]
    ax.set_xlim(0.3, N + 0.7); ax.set_ylim(0, max(max(r["share_primary"]), max(r["share_variant"])) * 1.3)
    fs.rounded_bars(ax, x - 0.2, r["share_primary"], width=0.38, color=fs.COLORS["blue"])
    fs.rounded_bars(ax, x + 0.2, r["share_variant"], width=0.38, color=fs.COLORS["orange"])
    ax.set_xticks(x)
    ax.set_xlabel("axis (primary order)"); ax.set_ylabel("% of variation beyond age")
    fs.panel_title(ax, "B", "Axis sizes")

    ax = axes[1, 0]
    cols = [fs.MEASURE_COLOURS[m] for m in MEASURES]
    xs = np.arange(3)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.05)
    for off, comp in ((-0.2, np.array(r["composition_primary"])), (0.2, np.array(r["composition_variant"]))):
        fs.rounded_bars(ax, xs + off, [1.0] * 3, width=0.36, color=cols[2])
        for j in range(3):
            b = 0.0
            for m in range(2):
                ax.add_patch(plt.Rectangle((xs[j] + off - 0.18, b), 0.36, comp[m, j], facecolor=cols[m],
                                           edgecolor=fs.TEXT_PRIMARY, linewidth=0.6, zorder=3))
                b += comp[m, j]
    ax.set_xticks(xs, [f"axis {j + 1}\nprimary | unthr." for j in range(3)])
    ax.set_ylabel("share of squared loadings")
    fs.panel_title(ax, "C", "What each axis is made of")

    ax = axes[1, 1]
    cr = r["correlates"]["network density"]
    xs = np.arange(3)
    ax.set_xlim(-0.6, 2.6)
    lim = max(0.35, np.max(np.abs(cr["primary"][:3] + cr["variant"][:3])) * 1.3)
    ax.set_ylim(-lim, lim)
    fs.rounded_bars(ax, xs - 0.2, cr["primary"][:3], width=0.38, color=fs.COLORS["blue"])
    fs.rounded_bars(ax, xs + 0.2, cr["variant"][:3], width=0.38, color=fs.COLORS["orange"])
    fs.reference_line(ax, 0)
    ax.set_xticks(xs, [f"axis {j + 1}" for j in range(3)])
    ax.set_ylabel("correlation with network density")
    fs.panel_title(ax, "D", "Density link")

    handles = [Patch(color=fs.COLORS["blue"], label="primary (per-bin 25% + MST)"),
               Patch(color=fs.COLORS["orange"], label=f"{variant}")] + \
              [Patch(color=c, label=MEASURE_LABEL[m]) for c, m in zip(cols, MEASURES)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.035), ncol=3)
    fig.text(0.06, 0.02, f"One-group model refitted to {variant} networks (same infants, standardisation, "
             f"20 starts, 1000 sweeps). B–D: q = {r['q']}; axes paired by Hungarian matching on |cos|.\n"
             "Network density is that of the thresholded primary network, a per-infant covariate.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "fig11_preprocessing_check")


if __name__ == "__main__":
    main()
