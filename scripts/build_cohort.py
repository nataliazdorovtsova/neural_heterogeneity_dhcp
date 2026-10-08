"""Stage 1: build the analysis cohort and freeze the data splits.

    python scripts/build_cohort.py

Writes to results/cohort/ (tracked in git — these must never silently change):
    cohort.csv            one row per session, with covariates and split labels
    validation_split.csv  infant-level held-out set, scored once at the very end
    cv_folds.csv          infant-level 5-fold assignment for model selection
    cascade.txt           the exclusion cascade, for the paper's Data section

Splits are assigned at INFANT level and inherited by sessions, so no infant can
appear in two folds regardless of which cohort is used (DECISIONS.md D3).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

import paths

SEED = 20261008          # fixed: these splits are frozen
VALIDATION_FRAC = 0.15
N_FOLDS = 5

OUT = paths.RESULTS / "cohort"
OUT.mkdir(parents=True, exist_ok=True)


def load_sessions() -> pd.DataFrame:
    """753 sessions, with demographics and QC merged on."""
    p = pd.read_csv(paths.PARTICIPANTS)
    dem = pd.read_csv(paths.DEMOGRAPHICS)
    dem.columns = dem.columns.str.strip()

    p["ses"] = p.session.astype(str).str.replace("^ses-", "", regex=True)
    dem["ses"] = dem.session_id.astype(str)

    keep = ["participant_id", "ses", "birth_weight", "head_circumference_scan",
            "radiology_score", "qc_dmri_eddy_rel_motion",
            "qc_dmri_eddy_abs_motion", "qc_dmri_eddy_percent_outlier_slices"]
    keep = [c for c in keep if c in dem.columns]
    df = p.merge(dem[keep], left_on=["dhcp_id", "ses"],
                 right_on=["participant_id", "ses"], how="left")

    df["pna"] = df.scan_age - df.birth_age
    df["preterm"] = df.birth_age < 37
    df = df.rename(columns={"density": "density_unthresh"})
    return df


def add_thresholded_density(df: pd.DataFrame) -> pd.DataFrame:
    """Post-threshold density, computed from the stack.

    The `density` column shipped with the participants file is the PRE-threshold
    figure (~0.854). The real one has to be measured (DECISIONS.md D4).
    """
    A = np.load(paths.STACK, mmap_mode="r")
    assert A.shape[2] == len(df), "stack and participants file disagree"
    df["density_thr"] = [(np.asarray(A[:, :, i]) > 0).sum() / (90 * 89)
                         for i in range(A.shape[2])]
    return df


def add_outcome_availability(df: pd.DataFrame) -> pd.DataFrame:
    strip = lambda s: s.astype(str).str.replace("^sub-", "", regex=True)
    cbcl = strip(pd.read_csv(paths.CBCL).src_id)
    ecbq = strip(pd.read_csv(paths.ECBQ).src_id)
    prs = pd.read_csv(paths.PRS).participant_id.astype(str)
    pcs = pd.read_csv(paths.ANCESTRY_PCS, sep=r"\s+", header=None)[0].astype(str)

    df["has_cbcl"] = df.dhcp_id.isin(cbcl)
    df["has_ecbq"] = df.dhcp_id.isin(ecbq)
    df["has_prs"] = df.dhcp_id.isin(prs)
    df["has_ancestry_pcs"] = df.dhcp_id.isin(pcs)
    return df


def select_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Earliest scan per infant (DECISIONS.md D3)."""
    df["in_primary"] = False
    df.loc[df.groupby("dhcp_id").scan_age.idxmin(), "in_primary"] = True
    return df


def stratum(infants: pd.DataFrame) -> pd.Series:
    """PMA quartile crossed with preterm status."""
    band = pd.qcut(infants.scan_age, 4, labels=False)
    return band.astype(str) + "_" + infants.preterm.astype(str)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Infant-level validation hold-out and CV folds; sessions inherit."""
    infants = df[df.in_primary].copy()
    strat = stratum(infants)

    train_ids, val_ids = train_test_split(
        infants.dhcp_id.values, test_size=VALIDATION_FRAC,
        stratify=strat.values, random_state=SEED)

    tr = infants[infants.dhcp_id.isin(train_ids)]
    folds = pd.Series(-1, index=tr.dhcp_id.values, name="fold")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, test_idx) in enumerate(skf.split(tr, stratum(tr))):
        folds.iloc[test_idx] = k

    df["split"] = np.where(df.dhcp_id.isin(val_ids), "validation", "train")
    df["fold"] = df.dhcp_id.map(folds).fillna(-1).astype(int)   # -1 = validation
    return df


def write_cascade(df: pd.DataFrame) -> str:
    n_inf = df.dhcp_id.nunique()
    prim = df[df.in_primary]
    lines = [
        "Exclusion cascade",
        "=================",
        "",
        "Upstream (ab15's pipeline, from her first year report — not ours):",
        "   887  imaging sessions released",
        "   886  T2w passed QC",
        "   758  dMRI passed QC",
        "   673  infants with both required modalities (110 missing one)",
        "   756  connectome matrices built",
        "    -3  participants with isolated nodes surviving mask dilation",
        f"   753  connectomes delivered to us  (observed: {len(df)})",
        "",
        "Ours:",
        "   no motion or QC exclusions applied (DECISIONS.md D8)",
        f"   {len(df)} sessions from {n_inf} unique infants ({len(df)-n_inf} repeat scans)",
        f"   {len(prim)}  primary cohort: earliest scan per infant",
        "",
        "Primary cohort composition:",
        f"   preterm (GA<37)   {int(prim.preterm.sum())}",
        f"   term              {int((~prim.preterm).sum())}",
        f"   PMA  {prim.scan_age.min():.2f} - {prim.scan_age.max():.2f} weeks",
        f"   PNA  {prim.pna.min():.2f} - {prim.pna.max():.2f} weeks",
        f"   corr(PMA, PNA)    {prim.scan_age.corr(prim.pna):+.3f}",
        "",
        "Splits (infant level, frozen; seed %d):" % SEED,
        f"   validation  {prim[prim.split=='validation'].shape[0]}",
        f"   model fitting/selection  {prim[prim.split=='train'].shape[0]} in {N_FOLDS} folds",
        "",
        "Outcome availability in the primary cohort:",
        f"   CBCL            {int(prim.has_cbcl.sum())}",
        f"   ECBQ            {int(prim.has_ecbq.sum())}",
        f"   PRS             {int(prim.has_prs.sum())}",
        f"   PRS + anc. PCs  {int((prim.has_prs & prim.has_ancestry_pcs).sum())}",
        f"   PRS + PCs + CBCL {int((prim.has_prs & prim.has_ancestry_pcs & prim.has_cbcl).sum())}",
    ]
    return "\n".join(lines)


def main():
    df = load_sessions()
    df = add_thresholded_density(df)
    df = add_outcome_availability(df)
    df = select_primary(df)
    df = assign_splits(df)

    # --- gates -------------------------------------------------------------
    prim = df[df.in_primary]
    assert len(prim) == df.dhcp_id.nunique() == 670, "primary cohort is not one row per infant"
    leak = df.groupby("dhcp_id").fold.nunique()
    assert (leak == 1).all(), "an infant appears in more than one fold"
    assert df.groupby("dhcp_id").split.nunique().eq(1).all(), "an infant spans both splits"
    assert (df.subject_index.values == np.arange(len(df))).all(), "stack index misaligned"

    # exclusions must not be age-dependent; we apply none, so this is a no-op
    # check that the SPLIT is not age-dependent instead
    from scipy import stats
    val = prim[prim.split == "validation"].scan_age
    tr = prim[prim.split == "train"].scan_age
    ks = stats.ks_2samp(val, tr)
    assert ks.pvalue > 0.05, f"validation split is age-biased (KS p={ks.pvalue:.3g})"

    cols = ["dhcp_id", "session", "subject_index", "in_primary", "split", "fold",
            "scan_age", "birth_age", "pna", "preterm", "sex",
            "birth_weight", "head_circumference_scan", "radiology_score",
            "qc_dmri_eddy_rel_motion", "density_unthresh", "density_thr",
            "has_cbcl", "has_ecbq", "has_prs", "has_ancestry_pcs"]
    df[[c for c in cols if c in df.columns]].to_csv(OUT / "cohort.csv", index=False)
    prim[prim.split == "validation"][["dhcp_id"]].to_csv(OUT / "validation_split.csv", index=False)
    prim[prim.split == "train"][["dhcp_id", "fold"]].to_csv(OUT / "cv_folds.csv", index=False)

    cascade = write_cascade(df)
    (OUT / "cascade.txt").write_text(cascade + "\n")
    print(cascade)
    print(f"\nvalidation vs training PMA: KS p = {ks.pvalue:.3f} (want > 0.05)")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
