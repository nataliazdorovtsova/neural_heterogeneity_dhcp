"""Gather sweep provenance records into the profiles the methods require.

    python scripts/collect_sweep.py

Reports held-out log predictive density over q at fixed K and over K at fixed q,
flags specifications that failed to converge or collapsed a group, and writes
results/sweep_summary.csv.

Scores are also expressed as a gain per held-out infant over a null baseline
(see baseline_lpd), which is what the model-selection figure plots.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import paths
from features import standardise

PROV = paths.RESULTS / "provenance"


def baseline_lpd():
    """Held-out log density of the null model, summed over the CV folds.

    The null model treats each of the 270 coordinates as an independent normal
    with the fit fold's mean and SD: no age, no groups, no axes. It uses the same
    infants, folds and standardisation as run_spec, so (spec - baseline) / n is
    the gain per held-out infant that the model's structure buys.
    """
    from scripts.run_spec import load
    coh, X = load()
    folds = coh.fold.values
    total = 0.0
    for f in np.unique(folds):
        _, Zo, *_ = standardise(X[folds != f], X[folds == f])
        total += float(-0.5 * (np.log(2 * np.pi) + Zo ** 2).sum())
    return total, len(coh)


def main():
    rows = []
    for f in sorted(PROV.glob("K*_q*_M*.json")):
        d = json.loads(f.read_text())
        c, r = d["config"], d["results"]
        if c.get("variant", "perbin25_mst") != "perbin25_mst":
            continue                         # other feature sets are not part of selection
        rows.append(dict(
            K=c["K"], q=c["q"], clock=c["clock"], sharing=c["covariance_sharing"],
            heldout_lpd=r["total_heldout_lpd"],
            per_infant=r["mean_heldout_lpd_per_infant"],
            n_heldout=sum(x["n_out"] for x in r["per_fold"]),
            converged=r["all_converged"],
            min_group_share=min(x["min_group_share"] for x in r["per_fold"]),
            usable_restarts=r["successful_restarts"],
            seconds=r["seconds"], digest=d["spec_digest"][:12], file=f.name))
    if not rows:
        print(f"no sweep records in {PROV}"); return

    # A spec may have been run more than once (benchmarks, reruns); keep the
    # most recent record for each, which is the last by filename timestamp.
    df = (pd.DataFrame(rows)
            .sort_values("file")
            .drop_duplicates(subset=["K", "q", "clock", "sharing"], keep="last")
            .sort_values(["K", "q"]).reset_index(drop=True))
    base, n = baseline_lpd()
    assert (df.n_heldout == n).all(), "spec and baseline scored different infants"
    df["baseline_lpd"] = base
    df["gain_per_infant"] = (df.heldout_lpd - base) / n
    print(f"null baseline: {base:,.1f} over {n} held-out infants "
          f"({base / n:.1f} per infant, {base / n / 270:.3f} per coordinate)")
    df.to_csv(paths.RESULTS / "sweep_summary.csv", index=False)
    print(f"{len(df)} specifications\n")

    piv = df.pivot(index="q", columns="K", values="heldout_lpd")
    print("=== held-out log predictive density (higher is better) ===")
    print(piv.to_string(float_format=lambda x: f"{x:,.0f}"))

    best = df.loc[df.heldout_lpd.idxmax()]
    print(f"\nbest: K={int(best.K)}, q={int(best.q)}  "
          f"({best.heldout_lpd:,.1f}; {best.per_infant:.1f} per held-out infant, "
          f"+{best.gain_per_infant:.1f} per infant over the null baseline)")

    print("\n=== best q at each K ===")
    for K, g in df.groupby("K"):
        b = g.loc[g.heldout_lpd.idxmax()]
        print(f"  K={K}: q={int(b.q):<3} lpd {b.heldout_lpd:>12,.0f}")
    print("\n=== best K at each q ===")
    for q, g in df.groupby("q"):
        b = g.loc[g.heldout_lpd.idxmax()]
        print(f"  q={q:<3}: K={int(b.K)}  lpd {b.heldout_lpd:>12,.0f}")

    bad = df[(~df.converged) | (df.min_group_share < 0.02)]
    if len(bad):
        print("\n=== flagged (not converged, or a group collapsed below 2%) ===")
        print(bad[["K", "q", "converged", "min_group_share", "usable_restarts"]]
              .to_string(index=False))
    else:
        print("\nall specifications converged with no collapsed groups")

    rng = df.heldout_lpd.max() - df.heldout_lpd.min()
    near = (df.heldout_lpd > df.heldout_lpd.max() - 0.01 * abs(df.heldout_lpd.max())).sum()
    print(f"\nprofile spread {rng:,.0f} lpd; {near} specifications within 1% of the best")
    print("(a flat profile is weak evidence for a particular rank or group count)")
    # seconds are per-task wall time; each task held its own CPU allocation
    # (8 CPUs for the sweep), so core-hours are this times the CPUs per task.
    print(f"\ntotal compute {df.seconds.sum()/3600:.1f} task wall-hours across {len(df)} specs "
          f"(x CPUs per task for core-hours)")


if __name__ == "__main__":
    main()
