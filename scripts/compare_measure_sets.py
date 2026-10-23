"""Do the conclusions depend on communicability? Compare measure sets.

    python scripts/compare_measure_sets.py

Sets (scripts/extract_alt_features.py), each fitted on the same infants:
  primary   strength + clustering + communicability
  eig       strength + clustering + eigenvector centrality
  sc        strength + clustering only

Held-out scores cannot be compared across sets (the data predicted differ), so
each check compares conclusions:

  1. one group or two?   K = 1 vs K = 2 at q = 15 and 25 within each set:
                         paired per-infant difference, SE and z (as in S16)
  2. same axes?          axes of the one-group q = 15 model compared with the
                         primary axes on the coordinates every set shares
                         (strength and clustering of each region), Hungarian-
                         matched on |cosine|
  3. reproducible?       split-half similarity of axes 1-3 within each set
                         (scripts/split_half.py --variant)

Also reports axis sizes and what each set's leading axes are made of.

Writes results/significance/measure_sets.json and
figures/fig12_measure_sets.{png,pdf}.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import paths
import figstyle as fs
from scripts.fit_final import principal
from scripts.split_half import match

PROV, PERINFANT = paths.RESULTS / "provenance", paths.RESULTS / "perinfant"
SETS = {"eig": "perbin25_mst_eig", "sc": "perbin25_mst_sc"}
LABEL = {"primary": "str + clu + communicability", "eig": "str + clu + eigenvector",
         "sc": "str + clu only"}
COLOUR = {"primary": fs.COLORS["blue"], "eig": fs.COLORS["vermillion"], "sc": fs.COLORS["bluish_green"]}
Q = 15
N = 8


def shared_rows(n_measures):
    """Row indices of strength and clustering (region-major layout)."""
    return np.array([n_measures * r + m for r in range(90) for m in (0, 1)])


def axes(path):
    d = np.load(path)
    L, S2, _, _ = principal(d["W"][0], d["psi"][0])
    return L, 100 * S2 / (S2.sum() + d["psi"][0].sum())


def k1_vs_k2(variant):
    """Paired K=1 minus K=2 per q, from per-infant held-out scores of one feature set."""
    recs = {}
    for f in sorted(PERINFANT.glob("K*_q*.npz")):
        c = json.loads((PROV / f"{f.stem}.json").read_text())["config"]
        if c.get("variant") != variant or c.get("n_iter") != 1000:
            continue
        z = np.load(f)
        recs[(c["K"], c["q"])] = pd.Series(z["lpd"], index=z["dhcp_id"].astype(str))
    out = {}
    for q in (15, 25):
        if (1, q) in recs and (2, q) in recs:
            d = (recs[(1, q)] - recs[(2, q)].reindex(recs[(1, q)].index)).values
            se = np.sqrt(len(d)) * d.std(ddof=1)
            out[q] = dict(diff=float(d.sum()), se=float(se), z=float(d.sum() / se),
                          per_infant=float(d.mean()), n=len(d))
    return out


def split_half(variant_dir, idx=(0, 1, 2)):
    """Split-half |cos| for the set's own axes `idx` (0-based, within the 8 tracked)."""
    files = sorted((paths.RESULTS / "significance" / variant_dir).glob("split*.npz"))
    if not files:
        return None
    c = np.array([np.load(f)["cos_ab"] for f in files])
    idx = [i for i in idx if i < c.shape[1]]
    return dict(n=len(files), axes=[i + 1 for i in idx], median=np.median(c[:, idx], 0).round(3).tolist(),
                min=c[:, idx].min(0).round(3).tolist())


def main():
    Lp, sp = axes(paths.RESULTS / "final" / f"K1_q{Q}.npz")
    sel = json.loads((paths.RESULTS / "selection.json").read_text())
    prim_kk = {r["q"]: dict(diff=r["diff"], se=r["se"], z=r["z"], per_infant=r["diff"] / 569)
               for r in sel["K1_vs_K2"] if r["q"] in (15, 25)}
    out = dict(primary=dict(K1_vs_K2=prim_kk, share=sp[:N].round(2).tolist(),
                            split_half=split_half("split_half")))
    for key, variant in SETS.items():
        f = paths.RESULTS / "final" / f"K1_q{Q}_{variant}.npz"
        if not f.exists():
            continue
        L, s = axes(f)
        nm = L.shape[0] // 90
        A, Bm = Lp[shared_rows(3)], L[shared_rows(nm)]
        idx, cos = match(A, Bm, N)
        comp = np.array([(L[m::nm, idx[:3]] ** 2).sum(0) for m in range(nm)]) / (L[:, idx[:3]] ** 2).sum(0)
        out[key] = dict(
            variant=variant, measures=nm, matched_axis=(idx + 1).tolist(),
            cos_shared=cos.round(3).tolist(), share=s[idx].round(2).tolist(),
            composition_axes_1_3=comp.round(3).tolist(), K1_vs_K2=k1_vs_k2(variant),
            # reproducibility of this set's own top three axes, and of the axes
            # that match primary axes 1-3 (which may sit lower in its order)
            split_half_own=split_half(f"split_half_q{Q}_{variant}"),
            split_half=split_half(f"split_half_q{Q}_{variant}", tuple(idx[:3])))
    (paths.RESULTS / "significance" / "measure_sets.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    fs.apply_style()
    figure(out)


def figure(out):
    keys = [k for k in ("primary", "eig", "sc") if k in out]
    fig, axes_ = plt.subplots(2, 2, figsize=fs.FIGURE_SIZES["full_page"])
    fig.subplots_adjust(hspace=0.6, wspace=0.32, bottom=0.17, top=0.95)
    x = np.arange(1, N + 1)

    ax = axes_[0, 0]
    for k, m in zip(("eig", "sc"), "oD"):
        if k in out:
            ax.plot(x, out[k]["cos_shared"], color=COLOUR[k], marker=m, markersize=6)
    fs.reference_line(ax, 0.9)
    ax.set_ylim(0, 1.05); ax.set_xticks(x)
    ax.set_xlabel("axis of the primary model"); ax.set_ylabel("similarity on shared measures")
    fs.panel_title(ax, "A", "Same axes?")

    ax = axes_[0, 1]
    w = 0.8 / len(keys)
    ax.set_xlim(0.3, N + 0.7)
    ax.set_ylim(0, max(max(out[k]["share"]) for k in keys) * 1.3)
    for i, k in enumerate(keys):
        fs.rounded_bars(ax, x - 0.4 + w * (i + 0.5), out[k]["share"], width=w, color=COLOUR[k])
    ax.set_xticks(x)
    ax.set_xlabel("axis (primary order)"); ax.set_ylabel("% of variation beyond age")
    fs.panel_title(ax, "B", "Axis sizes")

    ax = axes_[1, 0]
    xs = np.arange(3)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.05)
    for i, k in enumerate(keys):
        sh = out[k]["split_half"]
        if sh:
            fs.rounded_bars(ax, xs - 0.4 + w * (i + 0.5), sh["median"], width=w, color=COLOUR[k])
    fs.reference_line(ax, 0.7)
    ax.set_xticks(xs, [f"axis {j + 1}" for j in range(3)])
    ax.set_ylabel("split-half similarity (median)")
    ax.set_xlabel("axis of the primary model (matched)")
    fs.panel_title(ax, "C", "Reproducible?")

    ax = axes_[1, 1]
    qs = (15, 25)
    xs = np.arange(len(qs))
    vals, errs, cols, pos = [], [], [], []
    for i, k in enumerate(keys):
        kk = out[k]["K1_vs_K2"]
        for j, q in enumerate(qs):
            r = kk.get(q) or kk.get(str(q))          # int keys in memory, str after JSON
            if r:
                n = r.get("n", 569)
                vals.append(r["diff"] / n); errs.append(1.96 * r["se"] / n)
                cols.append(COLOUR[k]); pos.append(xs[j] - 0.4 + w * (i + 0.5))
    if vals:
        lim = max(np.abs(vals) + np.array(errs)) * 1.2
        ax.set_xlim(-0.6, len(qs) - 0.4); ax.set_ylim(-lim, lim)
        fs.rounded_bars(ax, pos, vals, width=w, color=cols)
        ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=fs.TEXT_PRIMARY, elinewidth=0.8,
                    capsize=3, zorder=4)
    fs.reference_line(ax, 0)
    ax.set_xticks(xs, [f"q = {q}" for q in qs])
    ax.set_ylabel("K=1 minus K=2, per infant")
    fs.panel_title(ax, "D", "Still one group?")

    fig.legend(handles=[Patch(color=COLOUR[k], label=LABEL[k]) for k in keys],
               loc="lower left", bbox_to_anchor=(0.06, 0.035), ncol=3)
    fig.text(0.06, 0.02, f"One-group model, q = {Q}, same 569 infants. A–C: each set's axes are "
             "matched to the primary axes\non the strength and clustering coordinates every set "
             "shares. D: paired held-out difference within each set\n(positive favours one group), "
             "95% intervals; held-out scores are not comparable across sets.", fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    fs.save_figure(fig, paths.FIGURES / "fig12_measure_sets")


if __name__ == "__main__":
    main()
