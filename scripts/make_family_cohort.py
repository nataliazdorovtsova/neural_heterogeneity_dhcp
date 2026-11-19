"""A cohort for a new run with twins (and the one triplet set) kept together
in the validation split and in the cross-validation folds (Natalia,
2026-11-19).

    NH_RUN=<run> python scripts/make_family_cohort.py

The shared results/cohort/ (D3) splits by infant: 9 of the 41 multi-infant
families are split between training and validation, and 21 training
families are split across folds. It stays unchanged, because the earlier
runs use it. This script writes results/<run>/cohort/ (a real folder, not
the shared symlink) with the minimum change:
- split: a split family goes to the side that holds most of its primary
  members; a tie goes to the side of its alphabetically first member;
- folds: every training family takes the fold of its alphabetically first
  member (in the new split).
Family = the first 7 characters of dhcp_id. Rows, their order and all other
columns are as in the shared cohort, so the feature files still match.
Repeat scans inherit their infant's split and fold.
"""

import json

import numpy as np
import pandas as pd

import paths

SRC = paths.PROJECT / "results" / "cohort"


def main():
    out = paths.RESULTS / "cohort"
    assert out.resolve() != SRC.resolve(), "this run uses the shared cohort; give it its own folder first"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "cohort.csv").exists(), f"{out / 'cohort.csv'} exists; not overwriting"
    c = pd.read_csv(SRC / "cohort.csv")
    c["dhcp_id"] = c.dhcp_id.astype(str)
    p = c[c.in_primary].copy()
    p["family"] = p.dhcp_id.str[:7]

    split, fold = dict(zip(p.dhcp_id, p.split)), dict(zip(p.dhcp_id, p.fold))
    moved_split, moved_fold = [], []
    for fam, g in p.groupby("family"):
        if g.split.nunique() > 1:
            n = g.split.value_counts()
            side = n.idxmax() if n.iloc[0] != n.iloc[-1] else g.sort_values("dhcp_id").split.iloc[0]
            for i in g.dhcp_id:
                if split[i] != side:
                    moved_split.append(dict(dhcp_id=i, family=fam, old=split[i], new=side))
                    split[i] = side
    p["split"] = p.dhcp_id.map(split)
    tr = p[p.split == "train"].sort_values("dhcp_id")
    for fam, g in tr.groupby("family"):
        f0 = fold[g.dhcp_id.iloc[0]]
        if f0 == -1:                       # first member came from validation: use the next one with a fold
            f0 = next(fold[i] for i in g.dhcp_id if fold[i] != -1)
        for i in g.dhcp_id:
            if fold[i] != f0:
                moved_fold.append(dict(dhcp_id=i, family=fam, old=int(fold[i]), new=int(f0)))
                fold[i] = f0
    for i in p.dhcp_id[p.split == "validation"]:
        fold[i] = -1

    c["split"] = c.dhcp_id.map(split)
    c["fold"] = c.dhcp_id.map(fold).astype(int)
    assert c.split.notna().all()
    q = c[c.in_primary].assign(family=lambda d: d.dhcp_id.str[:7])
    assert (q.groupby("family").split.nunique() == 1).all()
    assert (q[q.split == "train"].groupby("family").fold.nunique() == 1).all()
    assert ((q.split == "validation") == (q.fold == -1)).all()

    c.to_csv(out / "cohort.csv", index=False)
    q[q.split == "train"][["dhcp_id", "fold"]].to_csv(out / "cv_folds.csv", index=False)
    q[q.split == "validation"][["dhcp_id"]].to_csv(out / "validation_split.csv", index=False)
    summ = dict(source=str(SRC / "cohort.csv"), n_primary=int(len(q)),
                n_train=int((q.split == "train").sum()), n_validation=int((q.split == "validation").sum()),
                fold_sizes=q[q.split == "train"].fold.value_counts().sort_index().to_dict(),
                moved_split=moved_split, moved_fold=moved_fold,
                preterm_share=dict(train=float(q[q.split == "train"].preterm.mean()),
                                   validation=float(q[q.split == "validation"].preterm.mean())))
    (out / "family_regrouping.json").write_text(json.dumps(summ, indent=2, default=int))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("moved_split", "moved_fold")}, indent=1, default=int))
    print(f"moved between train/validation: {len(moved_split)} infants; between folds: {len(moved_fold)} infants")


if __name__ == "__main__":
    main()
