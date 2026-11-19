"""Weighted lobe communicability on a dense connectome stack, age-adjusted
for PLS (DECISIONS D24).

    NH_RUN=w25 python scripts/build_dense_lobes.py --stack S.npy \\
        --participants participants_with_index.csv [--tag w25] [--binarise]

Weighted edges (Duncan, 2026-11-19). --binarise (every retained edge = 1,
topology only) is kept as an option for a later analysis.

1. Stack (90, 90, n) of weighted networks, third axis matched to the cohort
   by session through the participants file (subject_index, session).
2. Per region: weighted communicability, strength-normalised (Crofts &
   Higham 2009; features.communicability). It is invariant to a global
   rescaling of an infant's weights (checked below). Regions are averaged
   into the 12 lobe x hemisphere groups (build_bin10.lobe_means).
3. Age adjustment, fitted in the training infants without the D20 seven
   (scripts/compare_age_spread.py): a restricted cubic spline mean in PMA
   per lobe, and a log SD of M0 (constant), M1 (one age scale shared by the
   12 lobes, as D21) or M2 (one per lobe). The model is chosen by held-out
   log predictive density (5 family-grouped folds), with the one-SE rule
   (Natalia, 2026-11-19; M1 on the bin10 values). z = (x - mean(PMA)) /
   SD(PMA) for all primary infants.

Checks printed: edge count and density per session and their correlation
with PMA; isolated nodes; scale invariance; age correlation of the raw and
adjusted lobe values; correlation of the adjusted values with total weight
(after PMA); SD of z by PMA bin (should be about 1); the D20 seven.

Writes results/features/<tag>_comm_lobe.npz (raw lobe values, all sessions)
and results/<run>/positions_<tag>.csv (dhcp_id, split, scan_age, z_<lobe>,
low7) for scripts/run_pls.py --positions positions_<tag>.csv.
"""

import argparse
import json

import numpy as np
import pandas as pd

import paths
from features import communicability
from scripts.build_bin10 import GROUP_NAMES, lobe_means, IU, N

PMA_BINS = [26, 34, 38, 40, 42, 46]


def rcs_knots(x):
    return np.percentile(x, [5, 35, 65, 95])


def rcs_basis(x, k):
    x = np.asarray(x, float)
    s = (k[-1] - k[0]) ** 2
    cols = [np.ones_like(x), x]
    for j in range(len(k) - 2):
        cols.append((np.maximum(x - k[j], 0) ** 3
                     - np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])) / s)
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--participants", required=True)
    ap.add_argument("--tag", default="dense25")
    ap.add_argument("--binarise", action="store_true")
    a = ap.parse_args()

    coh = pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv")
    part = pd.read_csv(a.participants)
    S = np.load(a.stack, mmap_mode="r")
    assert S.shape[:2] == (N, N) and S.shape[2] == len(part), (S.shape, len(part))
    idx = coh.merge(part[["session", "subject_index"]], on="session", how="left",
                    suffixes=("", "_stack")).subject_index_stack
    missing = coh.session[idx.isna()].tolist()
    print(f"sessions in cohort {len(coh)}, in stack {len(part)}, cohort sessions missing from stack: {len(missing)}")
    assert not coh.in_primary[idx.isna()].any(), f"primary sessions missing from the stack: {missing[:10]}"

    n = len(coh)
    comm = np.full((n, N), np.nan)
    n_edges = np.full(n, np.nan)
    isolated = np.zeros(n, int)
    total_w = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(idx[i]):
            continue
        W = np.asarray(S[:, :, int(idx[i])], float)
        assert np.allclose(W, W.T) and (np.diag(W) == 0).all() and (W >= 0).all()
        n_edges[i] = (W[IU] > 0).sum()
        isolated[i] = int((W.sum(1) == 0).sum())
        total_w[i] = W[IU].sum()
        A = (W > 0).astype(float) if a.binarise else W
        comm[i] = communicability(A)
        if i == 0:
            assert np.allclose(communicability(3.7 * A), comm[i]), "communicability is not scale-invariant"
    X = lobe_means(comm)

    prim = coh.in_primary.values.astype(bool)
    low7 = set(pd.read_csv(paths.EXCLUDE).dhcp_id.astype(str)) if paths.EXCLUDE.exists() else set()
    is_low7 = coh.dhcp_id.astype(str).isin(low7).values
    fit_rows = prim & (coh.split == "train").values & ~is_low7
    t = coh.scan_age.values
    from scripts.compare_age_spread import select, fit as fit_spread
    Xf = X[fit_rows]
    mu, sd = Xf.mean(0), Xf.std(0)
    sel, chosen = select((Xf - mu) / sd, t[fit_rows], coh.dhcp_id.astype(str)[fit_rows].str[:7].values)
    zfun, _, info = fit_spread((Xf - mu) / sd, t[fit_rows], chosen)
    Zs = np.full_like(X, np.nan)
    ok = ~np.isnan(X).any(1)
    Zs[ok] = zfun((X[ok] - mu) / sd, t[ok])
    params = dict(chosen=chosen, **sel, shared_g=info["g"])
    B = rcs_basis(t, rcs_knots(t[fit_rows]))      # PMA spline for the total-weight check

    out = paths.PROJECT / "results" / "features"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.tag}_comm_lobe.npz", X=X, features=GROUP_NAMES, node=comm,
                        measure=("binary" if a.binarise else "weighted") + " communicability (degree/strength-normalised)", dhcp_id=coh.dhcp_id.values,
                        session=coh.session.values, stack=str(a.stack))
    pos = pd.DataFrame(dict(dhcp_id=coh.dhcp_id.astype(str), split=coh.split, scan_age=t))
    for d, g in enumerate(GROUP_NAMES):
        pos[f"z_{g}"] = Zs[:, d]
    pos["low7"] = is_low7
    pos = pos[prim].reset_index(drop=True)
    pos.to_csv(paths.RESULTS / f"positions_{a.tag}.csv", index=False)

    p = prim & ~np.isnan(n_edges)
    tr = fit_rows
    sd_bins = pd.DataFrame(Zs[tr]).groupby(pd.cut(t[tr], PMA_BINS, right=False), observed=False).std()
    summ = dict(
        stack=str(a.stack), binarise=a.binarise, n_sessions=int(p.sum()),
        edges=dict(min=int(np.nanmin(n_edges[p])), max=int(np.nanmax(n_edges[p])),
                   median=float(np.nanmedian(n_edges[p]))),
        density_median=float(np.nanmedian(n_edges[p]) / len(IU[0])),
        r_edges_pma=float(np.corrcoef(n_edges[p], t[p])[0, 1]) if np.nanstd(n_edges[p]) > 0 else 0.0,
        r_total_weight_pma=float(np.corrcoef(total_w[p], t[p])[0, 1]),
        sessions_with_isolated_nodes=int((isolated[p] > 0).sum()),
        r_raw_lobe_pma={g: round(float(np.corrcoef(X[p, d], t[p])[0, 1]), 3) for d, g in enumerate(GROUP_NAMES)},
        r_z_lobe_pma={g: round(float(np.corrcoef(Zs[p, d], t[p])[0, 1]), 3) for d, g in enumerate(GROUP_NAMES)},
        r_raw_lobe_total_weight={g: round(float(np.corrcoef(X[p, d], total_w[p])[0, 1]), 3)
                                 for d, g in enumerate(GROUP_NAMES)},
        # does total weight still go with the age-adjusted values? (total
        # weight residualised on the same PMA spline)
        r_z_lobe_total_weight_given_pma={
            g: round(float(np.corrcoef(Zs[p, d], total_w[p] - B[p] @ np.linalg.lstsq(B[p], total_w[p], rcond=None)[0])[0, 1]), 3)
            for d, g in enumerate(GROUP_NAMES)},
        z_sd_by_pma_bin_train={str(b): row.round(2).tolist() for b, row in sd_bins.iterrows()},
        low7_min_z=pos[pos.low7][[f"z_{g}" for g in GROUP_NAMES]].min().round(2).to_dict(),
        n_abs_z_gt4=int((np.abs(Zs[p & ~is_low7]) > 4).any(1).sum()),
        age_model=params)
    (paths.RESULTS / f"positions_{a.tag}_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "age_model"}, indent=1, default=float))
    print("age model:", json.dumps({k: params[k] for k in ("chosen", "vs_best", "shared_g")}, default=float))


if __name__ == "__main__":
    main()
