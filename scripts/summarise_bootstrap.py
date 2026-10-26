"""Bootstrap intervals for the selected model's axes.

    python scripts/summarise_bootstrap.py

Reads the replicates from scripts/bootstrap.py. For every axis and every
region x measure loading it reports the 95% percentile interval. A loading is
"significant" when its interval excludes zero; the brain figures colour only
those. Axis sizes get intervals too.

Percentile intervals are pointwise: across 270 loadings per axis, about 5% of
truly-zero loadings would be flagged by chance. The figures therefore show
the pattern, and the tables carry the intervals.

Writes results/significance/bootstrap_summary.npz and
results/significance/bootstrap_loadings.csv.
"""

import json

import numpy as np
import pandas as pd

import paths
import atlas
from features import MEASURES
from scripts.fit_final import principal

SIG = paths.RESULTS / "significance"


def main():
    import sys
    sub = sys.argv[1] if len(sys.argv) > 1 else "bootstrap"
    files = sorted((SIG / sub).glob("chunk*.npz")) + sorted((SIG / sub).glob("rep*.npz"))
    chunks = [np.load(f) for f in files]
    q = int(chunks[0]["q"])
    Lb = np.concatenate([c["loadings"] for c in chunks])          # (n_boot, 270, q)
    S2b = np.concatenate([c["S2"] for c in chunks])
    conv = np.concatenate([c["converged"] for c in chunks])

    d = np.load(paths.RESULTS / "final" / f"K1_q{q}.npz")
    L, S2, _, _ = principal(d["W"][0], d["psi"][0])
    total = S2.sum() + d["psi"][0].sum()

    # Axis stability as the methods plan specifies (section 4): the principal
    # angle between each leading axis and its counterpart in each replicate
    # (best match among the replicate's 10 largest axes, Hungarian on |cos|),
    # and the largest principal angle between the axis 1-3 subspaces.
    from scipy.optimize import linear_sum_assignment
    ref = L[:, :5] / np.linalg.norm(L[:, :5], axis=0)
    ang = np.empty((len(Lb), 5)); blk = np.empty(len(Lb))
    for b in range(len(Lb)):
        U = np.linalg.svd(Lb[b], full_matrices=False)[0][:, :10]
        cosm = np.abs(ref.T @ U)
        rows, cols = linear_sum_assignment(-cosm)
        ang[b] = np.degrees(np.arccos(np.clip(cosm[rows, cols], 0, 1)))
        sv = np.linalg.svd(np.linalg.qr(L[:, :3])[0].T @ U[:, :3], compute_uv=False)
        blk[b] = np.degrees(np.arccos(np.clip(sv.min(), 0, 1)))
    angles = dict(axis_median_deg=np.median(ang, 0).round(1).tolist(),
                  axis_95th_deg=np.percentile(ang, 95, 0).round(1).tolist(),
                  block_1_3_max_angle_median_deg=round(float(np.median(blk)), 1),
                  block_1_3_max_angle_95th_deg=round(float(np.percentile(blk, 95)), 1))

    lo, hi = np.percentile(Lb, [2.5, 97.5], axis=0)
    sig = (lo > 0) | (hi < 0)
    share_lo, share_hi = np.percentile(100 * S2b / total, [2.5, 97.5], axis=0)

    np.savez(SIG / "bootstrap_summary.npz", q=q, L=L, lo=lo, hi=hi, significant=sig,
             share=100 * S2 / total, share_lo=share_lo, share_hi=share_hi,
             n_boot=len(Lb), n_converged=int(conv.sum()))
    rows = []
    for r in range(90):
        for m, meas in enumerate(MEASURES):
            i = 3 * r + m
            for j in range(min(q, 8)):
                rows.append(dict(region=atlas.NAMES[r], lobe=atlas.LOBE[r], measure=meas,
                                 axis=j + 1, loading=L[i, j], ci_low=lo[i, j], ci_high=hi[i, j],
                                 significant=bool(sig[i, j])))
    pd.DataFrame(rows).to_csv(SIG / "bootstrap_loadings.csv", index=False)

    summary = dict(q=q, n_boot=len(Lb), n_converged=int(conv.sum()),
                   share=(100 * S2 / total).round(2).tolist()[:8],
                   share_ci=[[round(a, 2), round(b, 2)] for a, b in zip(share_lo[:8], share_hi[:8])],
                   significant_loadings_per_axis=sig.sum(0).tolist()[:8],
                   principal_angles=angles)
    (SIG / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
