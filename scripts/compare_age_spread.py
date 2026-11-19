"""Which age model for the spread of the lobe values? Held-out comparison.

    NH_RUN=bin10_no7 python scripts/compare_age_spread.py [--features bin10_comm_avg_lobe]
                                                          [--positions positions_scaled.csv]

Lobe values x_d (12 lobes), training infants without the D20 seven.
Mean per lobe: restricted cubic spline in PMA (4 knots). Spread models:
  M0  constant spread         log SD_d(t) = c_d                    (12 spread parameters)
  M1  one shared age scale    log SD_d(t) = c_d + g1 u + g2 u^2    (12 + 2; as D21)
  M2  one age scale per lobe  log SD_d(t) = c_d + g1d u + g2d u^2  (36)
with u = (PMA - 40) / 5. Gaussian maximum likelihood, lobes independent.

Score: 5-fold cross-validation with whole families in one fold (twins
together; the cohort folds split 21 twin pairs). Held-out log predictive
density summed over lobes, per infant; differences between models with the
SE of the paired per-infant difference. One-SE rule: the simplest model
within one SE of the best.

Also: correlation of the z-scores from M0/M1/M2 (fitted on all training
infants) with each other and with the factor-model lobe z-scores of the
positions file. Writes results/<run>/age_spread_<features>.json.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import paths
from scripts.build_bin10 import GROUP_NAMES
from scripts.build_dense_lobes import rcs_knots, rcs_basis

MODELS = ("M0", "M1", "M2")


def fit(X, t, model):
    """X (n, D). Returns a function z(X, t) and a function logpdf(X, t) -> (n,)."""
    n, D = X.shape
    k = rcs_knots(t)
    B = rcs_basis(t, k)
    u = (t - 40) / 5
    U = np.column_stack([u, u ** 2])
    nb = B.shape[1]
    beta0 = np.linalg.lstsq(B, X, rcond=None)[0]                  # (nb, D)
    c0 = np.log((X - B @ beta0).std(0))
    ng = {"M0": 0, "M1": 2, "M2": 2 * D}[model]

    def unpack(th):
        beta = th[:nb * D].reshape(nb, D)
        c = th[nb * D:nb * D + D]
        g = th[nb * D + D:]
        G = np.zeros((2, D)) if model == "M0" else (np.repeat(g[:, None], D, 1) if model == "M1"
                                                     else g.reshape(2, D))
        return beta, c, G

    def nll_grad(th):
        beta, c, G = unpack(th)
        ls = c + U @ G                                            # (n, D) log SD
        w = np.exp(-2 * ls)
        e = X - B @ beta
        f = np.sum(ls + 0.5 * e ** 2 * w)
        dls = 1 - e ** 2 * w                                      # d f / d ls
        gb = -(B.T @ (e * w)).ravel()
        gc = dls.sum(0)
        gG = U.T @ dls                                            # (2, D)
        gg = [] if model == "M0" else (gG.sum(1) if model == "M1" else gG.ravel())
        return f, np.r_[gb, gc, gg]

    th0 = np.r_[beta0.ravel(), c0, np.zeros(ng)]
    res = minimize(nll_grad, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000))
    assert res.success, res.message
    beta, c, G = unpack(res.x)

    def parts(Xn, tn):
        Bn = rcs_basis(tn, k)
        un = (tn - 40) / 5
        ls = c + np.column_stack([un, un ** 2]) @ G
        return (Xn - Bn @ beta) / np.exp(ls), ls

    def z(Xn, tn):
        return parts(Xn, tn)[0]

    def logpdf(Xn, tn):
        zz, ls = parts(Xn, tn)
        return np.sum(-0.5 * np.log(2 * np.pi) - ls - 0.5 * zz ** 2, axis=1)

    return z, logpdf, dict(g=G[:, :1].ravel().round(4).tolist() if model == "M1" else None)


def family_folds(fam, k=5, seed=0):
    u = np.unique(fam)
    rng = np.random.default_rng(seed)
    f_of = dict(zip(u, rng.permutation(len(u)) % k))
    return np.array([f_of[f] for f in fam])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="bin10_comm_avg_lobe")
    ap.add_argument("--positions", default="positions_scaled.csv")
    a = ap.parse_args()
    coh = pd.read_csv(paths.PROJECT / "results" / "cohort" / "cohort.csv")
    X = np.load(paths.PROJECT / "results" / "features" / f"{a.features}.npz")["X"]
    low7 = set(pd.read_csv(paths.EXCLUDE).dhcp_id.astype(str)) if paths.EXCLUDE.exists() else set()
    keep = (coh.in_primary & (coh.split == "train") & ~coh.dhcp_id.astype(str).isin(low7)).values
    c, Xt = coh[keep].reset_index(drop=True), X[keep]
    Xt = (Xt - Xt.mean(0)) / Xt.std(0)            # units only; does not change the comparison
    t = c.scan_age.values
    fold = family_folds(c.dhcp_id.str[:7].values)

    lpd = {m: np.empty(len(c)) for m in MODELS}
    for f in range(5):
        tr, te = fold != f, fold == f
        for m in MODELS:
            _, logpdf, _ = fit(Xt[tr], t[tr], m)
            lpd[m][te] = logpdf(Xt[te], t[te])
    tot = {m: float(lpd[m].sum()) for m in MODELS}
    best = max(tot, key=tot.get)
    diff = {m: dict(vs_best=round(tot[m] - tot[best], 1),
                    se=round(float(np.std(lpd[m] - lpd[best], ddof=1) * np.sqrt(len(c))), 1)) for m in MODELS}
    chosen = next(m for m in MODELS if tot[best] - tot[m] <= diff[m]["se"])
    pairs = {f"{m1}-{m2}": dict(diff=round(tot[m2] - tot[m1], 1),
                                se=round(float(np.std(lpd[m2] - lpd[m1], ddof=1) * np.sqrt(len(c))), 1))
             for m1, m2 in (("M0", "M1"), ("M1", "M2"), ("M0", "M2"))}

    zs, info = {}, {}
    for m in MODELS:
        z, _, inf = fit(Xt, t, m)
        zs[m], info[m] = z(Xt, t), inf
    corr = {f"{m1}-{m2}": round(float(np.min([np.corrcoef(zs[m1][:, d], zs[m2][:, d])[0, 1]
                                             for d in range(12)])), 3)
            for m1, m2 in (("M0", "M1"), ("M1", "M2"), ("M0", "M2"))}
    pos = pd.read_csv(paths.RESULTS / a.positions)
    pos["dhcp_id"] = pos.dhcp_id.astype(str)
    zf = c[["dhcp_id"]].astype(str).merge(pos, on="dhcp_id", how="left")[[f"z_{g}" for g in GROUP_NAMES]].values
    ok = ~np.isnan(zf).any(1)
    corr_factor = {m: dict(min=round(float(np.min([np.corrcoef(zs[m][ok, d], zf[ok, d])[0, 1] for d in range(12)])), 3),
                           median=round(float(np.median([np.corrcoef(zs[m][ok, d], zf[ok, d])[0, 1]
                                                         for d in range(12)])), 3)) for m in MODELS}
    out = dict(features=a.features, n=int(len(c)), n_families=int(len(np.unique(c.dhcp_id.str[:7]))),
               heldout_lpd=tot, vs_best=diff, pairwise=pairs, best=best, chosen_one_se=chosen,
               shared_scale_g=info["M1"]["g"],
               sd_ratio_30_vs_40_M1=round(float(np.exp(-2 * info["M1"]["g"][0] + 4 * info["M1"]["g"][1])), 3),
               min_lobe_corr_between_models=corr, corr_with_factor_model_z=corr_factor)
    (paths.RESULTS / f"age_spread_{a.features}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
