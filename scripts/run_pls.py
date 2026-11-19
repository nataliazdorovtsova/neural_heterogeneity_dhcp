"""PLS correlation between the 12 lobe communicability values and (1) polygenic
scores, (2) 18-month behaviour (Duncan, 2026-11-19). model/pls.py has the
method.

    NH_RUN=bin10_no7 python scripts/run_pls.py [--positions positions_scaled.csv]
                                               [--n-perm 10000] [--n-boot 10000]

Lobe block: the 12 age-adjusted lobe z-scores (positions file, D21), entered
directly (no factor model).

PLS 1, polygenic scores (n = 411, EUR ancestry, D9):
  prs_condition  main: one score per condition (12; Natalia, 2026-11-19).
                 The 27 psychiatric / neurodevelopmental scores cover 12
                 conditions; several have near-copies (ADHD r = 0.94, eight
                 autism variants) that PLS would over-weight. Per condition:
                 the overall-diagnosis score, as chosen in D22 where D22 has
                 one; otherwise the only score.
  prs_psych      the 27 psychiatric / neurodevelopmental scores (sensitivity)
  prs_all48      all 48 scores (sensitivity)
  Both blocks are residualised on PC1-10, sex, GA at birth, rcs(PMA at scan).
PLS 2, behaviour: the 7 CBCL syndrome t-scores + 3 ECBQ scales, infants with
  all 10. Both blocks residualised on sex, GA at birth, rcs(PMA at scan), age
  at CBCL and at ECBQ. behaviour_rank: the same with each behaviour scale
  rank-transformed first (CBCL floor at 50).

Each analysis: 10,000 family-block permutations, 10,000 family bootstraps
(Procrustes-aligned), and a train -> validation replication (saliences from
the training infants, latent correlation in the validation infants,
10,000 permutations). Residualisation is done within each sample used.

Summary statistics only. Writes results/<run>/pls/.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import paths
from model import pls as P
from scripts.run_bin10_step4 import LOBES, rcs
from scripts.run_bin10_step5 import load as load_prs, NEURO, BRAIN
from scripts.run_bin10_step6 import load as load_behaviour, CBCL, ECBQ

OUT = paths.RESULTS / "pls"
PER_CONDITION = ["adhd2022", "asd2019", "an2019", "anx2024", "bip2024_noUKBB", "insomnia2017",
                 "mdd2025_noUKBB", "ocd2024", "panic2019", "ptsd2019", "scz2022", "sud2023"]


def residualise(A, C):
    A = np.asarray(A, float)
    return A - C @ np.linalg.lstsq(C, A, rcond=None)[0]


def prs_covariates(d):
    return np.column_stack([np.ones(len(d)), d[[f"PC{i}" for i in range(1, 11)]].values, d.male, d.ga_birth,
                            rcs(d.scan_age.values)])


def behaviour_covariates(d):
    return np.column_stack([np.ones(len(d)), d.male, d.ga_birth, rcs(d.scan_age.values), d.age_cbcl_clean,
                            d.age_ecbq_clean])


def run_one(name, d, ycols, covfun, rank, n_perm, n_boot):
    def blocks(sub):
        C = covfun(sub)
        Y = sub[ycols].values.astype(float)
        if rank:
            Y = np.apply_along_axis(stats.rankdata, 0, Y)
        return residualise(sub[LOBES].values, C), residualise(Y, C)

    X, Y = blocks(d)
    r = P.pls(X, Y, d.family.values, n_perm=n_perm, n_boot=n_boot)
    tr, te = d[d.split == "train"], d[d.split != "train"]
    Xtr, Ytr = blocks(tr)
    Xte, Yte = blocks(te)
    rep = P.replicate(Xtr, Ytr, Xte, Yte, te.family.values, n_perm=n_perm)

    L = len(r["s"])
    lv = pd.DataFrame(dict(lv=np.arange(1, L + 1), singular_value=r["s"], cov_share=r["cov_share"],
                           p_perm=r["p_perm"], s_null_95=r["s_perm_95"], r_validation=rep["r"],
                           p_validation=rep["p"]))
    rows = []
    for block, names, U, bsr, rs, ci in (("lobes", LOBES, r["U"], r["bsr_x"], r["rx"], r["rx_ci"]),
                                          ("y", ycols, r["V"], r["bsr_y"], r["ry"], r["ry_ci"])):
        for i, v in enumerate(names):
            for l in range(min(L, 3)):
                rows.append(dict(block=block, variable=v, lv=l + 1, salience=U[i, l], bsr=bsr[i, l],
                                 r_structure=rs[i, l], r_lo=ci[0, i, l], r_hi=ci[1, i, l]))
    sal = pd.DataFrame(rows)
    lv.to_csv(OUT / f"{name}_lvs.csv", index=False)
    sal.to_csv(OUT / f"{name}_saliences.csv", index=False)
    lv1 = sal[sal.lv == 1]
    stable = lv1[lv1.bsr.abs() >= 2.58].sort_values("bsr", key=np.abs, ascending=False)
    return dict(name=name, n=r["n"], n_train=len(tr), n_validation=len(te), n_y=len(ycols),
                lvs=lv.head(5).round(4).to_dict("records"),
                n_lv_p05=int((r["p_perm"] < 0.05).sum()),
                lv1_stable_bsr258=stable[["block", "variable", "bsr", "r_structure"]].round(3).to_dict("records"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="positions_scaled.csv")
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert a.positions == "positions_scaled.csv", "load() helpers read positions_scaled.csv"

    d, scores = load_prs()
    psych = [s for s in scores if s not in NEURO + BRAIN]
    assert set(PER_CONDITION) <= set(psych) and len(psych) == 27
    out = [run_one("prs_condition", d, PER_CONDITION, prs_covariates, False, a.n_perm, a.n_boot),
           run_one("prs_psych", d, psych, prs_covariates, False, a.n_perm, a.n_boot),
           run_one("prs_all48", d, scores, prs_covariates, False, a.n_perm, a.n_boot)]

    b = load_behaviour()
    b = b.dropna(subset=CBCL + ECBQ + LOBES + ["age_cbcl_clean", "age_ecbq_clean", "ga_birth", "scan_age"])
    b = b.reset_index(drop=True)
    for name, rank in (("behaviour", False), ("behaviour_rank", True)):
        out.append(run_one(name, b, CBCL + ECBQ, behaviour_covariates, rank, a.n_perm, a.n_boot))

    summ = dict(run=paths.RUN, positions=a.positions, n_perm=a.n_perm, n_boot=a.n_boot,
                psych_scores=psych, analyses=out)
    (OUT / "pls_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
