"""Paired model comparison on per-infant held-out scores, and model selection.

    python scripts/compare_models.py

Every infant is held out exactly once in cross-validation, so two models can be
compared on the same 569 infants: d_i = lpd_i(model 1) - lpd_i(model 2). The
total difference has standard error sqrt(n) * sd(d) (Vehtari, Gelman & Gabry,
2017), which says whether a difference is larger than the infant-to-infant
noise in the comparison.

Selection (the one-standard-error rule; Hastie, Tibshirani & Friedman, 2009):
among models whose total held-out score is within one SE of the best model's
(SE of the paired difference to the best), choose the one with the fewest
parameters.

Uses only records fitted with --n-iter 1000 (the 2026-10-16 rerun), so every
candidate had the same optimisation budget.

Writes results/selection.json, results/model_comparison.csv and
figures/fig7_model_comparison.{png,pdf}.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import paths
import figstyle as fs
from features import standardise
from model import splines
from scripts.run_spec import load

PERINFANT = paths.RESULTS / "perinfant"
PROV = paths.RESULTS / "provenance"
N_ITER = 1000
D = 270


def n_params(K, q, M=splines.M_DEFAULT):
    """Free parameters: trajectories, loadings (less rotation), uniquenesses, weights."""
    return K * (M * D + D * q - q * (q - 1) // 2 + D) + (K - 1)


def baseline_pointwise():
    """Per-infant held-out log density of the null model (as in collect_sweep)."""
    coh, X = load()
    out = pd.Series(0.0, index=coh.dhcp_id.astype(str).values)
    for f in np.unique(coh.fold.values):
        m = coh.fold.values == f
        _, Zo, *_ = standardise(X[~m], X[m])
        out[coh.dhcp_id.astype(str).values[m]] = -0.5 * (np.log(2 * np.pi) + Zo ** 2).sum(1)
    return out


def load_records():
    rows = []
    for f in sorted(PERINFANT.glob("K*_q*.npz")):
        d = json.loads((PROV / f"{f.stem}.json").read_text())
        c = d["config"]
        if c.get("n_iter") != N_ITER or c.get("variant", "perbin25_mst") != "perbin25_mst":
            continue                         # same budget, primary feature set only
        z = np.load(f)
        rows.append(dict(K=c["K"], q=c["q"], file=f.stem, converged=d["results"]["all_converged"],
                         lpd=pd.Series(z["lpd"], index=z["dhcp_id"].astype(str))))
    df = pd.DataFrame(rows).sort_values("file").drop_duplicates(["K", "q"], keep="last")
    return df.sort_values(["K", "q"]).reset_index(drop=True)


def paired(a, b):
    d = (a - b.reindex(a.index)).values
    return d.sum(), np.sqrt(len(d)) * d.std(ddof=1)


def main():
    df = load_records()
    if df.empty:
        raise SystemExit("no per-infant records with n_iter = 1000")
    base = baseline_pointwise()
    n = len(base)
    assert all(len(l) == n and set(l.index) == set(base.index) for l in df.lpd), \
        "records scored different infants"

    df["total"] = [l.sum() for l in df.lpd]
    df["gain_per_infant"] = [(l - base.reindex(l.index)).mean() for l in df.lpd]
    df["n_params"] = [n_params(K, q) for K, q in zip(df.K, df.q)]
    best = df.loc[df.total.idxmax()]
    diffs = [paired(best.lpd, l) for l in df.lpd]
    df["diff_to_best"] = [d for d, _ in diffs]
    df["se_to_best"] = [s for _, s in diffs]
    df["within_1se"] = df.diff_to_best <= df.se_to_best
    sel = df[df.within_1se].sort_values("n_params").iloc[0]

    # K = 1 vs K = 2 at each q
    kk = []
    for q in sorted(set(df.q[df.K == 1]) & set(df.q[df.K == 2])):
        a = df[(df.K == 1) & (df.q == q)].iloc[0]; b = df[(df.K == 2) & (df.q == q)].iloc[0]
        d, se = paired(a.lpd, b.lpd)
        kk.append(dict(q=q, diff=d, se=se, z=d / se))
    kk = pd.DataFrame(kk)

    # runner-up for the validation comparison: the best model with a different K
    other = df[df.K != sel.K]
    runner = other.loc[other.total.idxmax()] if len(other) else None

    df.drop(columns="lpd").to_csv(paths.RESULTS / "model_comparison.csv", index=False)
    q1 = sorted(df.q[df.K == 1])
    sel_json = dict(
        rule="one-standard-error: fewest parameters within 1 SE (paired, per infant) of the best",
        n_infants=n, n_iter=N_ITER,
        best=dict(K=int(best.K), q=int(best.q), total=float(best.total),
                  gain_per_infant=float(best.gain_per_infant)),
        selected=dict(K=int(sel.K), q=int(sel.q), total=float(sel.total),
                      gain_per_infant=float(sel.gain_per_infant),
                      diff_to_best=float(sel.diff_to_best), se_to_best=float(sel.se_to_best)),
        runner_up=None if runner is None else dict(K=int(runner.K), q=int(runner.q),
                                                   total=float(runner.total)),
        q_grid_bracketed=bool(int(best.q) < max(q1)) if best.K == 1 else True,
        K1_vs_K2=kk.to_dict(orient="records"),
    )
    (paths.RESULTS / "selection.json").write_text(json.dumps(sel_json, indent=2))

    pd.set_option("display.width", 140)
    print(df.drop(columns=["lpd", "file"]).to_string(index=False,
          float_format=lambda x: f"{x:,.1f}"))
    print("\nK=1 minus K=2 at each q (positive favours one group):")
    print(kk.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print(f"\nbest: K={best.K}, q={best.q}; selected by the 1-SE rule: K={sel.K}, q={sel.q} "
          f"({sel.diff_to_best:,.0f} below the best, SE {sel.se_to_best:,.0f})")
    if not sel_json["q_grid_bracketed"]:
        print("NOTE: the best model is at the largest q tested; the q grid is not bracketed")

    fs.apply_style()
    fig_comparison(df, kk, sel, n)


def fig_comparison(df, kk, sel, n):
    fig, axes = plt.subplots(1, 2, figsize=fs.FIGURE_SIZES["double_column"])
    fig.subplots_adjust(wspace=0.32, bottom=0.3, top=0.88)

    ax = axes[0]
    for K, g in df.groupby("K"):
        g = g.sort_values("q")
        ax.errorbar(g.q, g.gain_per_infant, yerr=g.se_to_best / n, color=fs.K_COLOURS[K],
                    marker=fs.K_MARKERS[K], ecolor=fs.TEXT_PRIMARY, elinewidth=0.8, capsize=3,
                    label=f"K = {K}", zorder=3)
    ax.annotate("selected", (sel.q, sel.gain_per_infant), xytext=(0, -26),
                textcoords="offset points", ha="center", fontsize=fs.NOTE_SIZE,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("number of axes q"); ax.set_ylabel("gain per infant")
    fs.panel_title(ax, "A", "Held-out score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    ax = axes[1]
    x = np.arange(len(kk))
    v = kk["diff"].values / n
    lim = np.max(np.abs(v) + 2 * kk.se.values / n) * 1.15
    ax.set_xlim(-0.6, len(kk) - 0.4); ax.set_ylim(-lim, lim)
    fs.rounded_bars(ax, x, v, width=0.65,
                    color=[fs.K_COLOURS[1] if d > 0 else fs.K_COLOURS[2] for d in v])
    ax.errorbar(x, v, yerr=1.96 * kk.se.values / n, fmt="none", ecolor=fs.TEXT_PRIMARY,
                elinewidth=0.8, capsize=4, zorder=4)
    fs.reference_line(ax, 0)
    ax.set_xticks(x, [str(q) for q in kk.q])
    ax.set_xlabel("number of axes q"); ax.set_ylabel("K=1 minus K=2, per infant")
    fs.panel_title(ax, "B", "One group vs two")
    ax.legend(handles=[Patch(color=fs.K_COLOURS[1], label="favours K = 1"),
                       Patch(color=fs.K_COLOURS[2], label="favours K = 2")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.text(0.07, -0.02, f"Paired comparison on the same {n} held-out infants. A: error bars "
             "= 1 SE of the difference to the best model. B: error bars = 95% interval.\n"
             "Selection rule: fewest parameters within 1 SE of the best.",
             fontsize=fs.NOTE_SIZE, color=fs.TEXT_SECONDARY, va="top")
    return fs.save_figure(fig, paths.FIGURES / "fig7_model_comparison")


if __name__ == "__main__":
    main()
