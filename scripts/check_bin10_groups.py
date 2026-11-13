"""Who is in the small group of the two-group model? (run bin10, step 2 check)

    NH_RUN=bin10 python scripts/check_bin10_groups.py

With all infants, K = 2 at q = 4 scores best in CV, but its smaller group holds
only 1-9% of infants in each fold. This assigns the training infants of the
full-training K = 2 fit to its groups and asks whether the small group is the
7 orbitofrontal-low infants (D19), other outliers, or a real subgroup
(age, prematurity, motion, radiology score).

Writes results/bin10/group_check.json.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import paths
from model import mfa, splines
from scripts.build_bin10 import GROUP_NAMES
from scripts.run_spec import load, CLOCK, age_for
from scripts.validate_once import fitted


def main():
    sel = json.loads((paths.RESULTS / "selection.json").read_text())
    K, q = (sel["best"]["K"], sel["best"]["q"]) if sel["best"]["K"] == 2 else (2, sel["runner_up"]["q"])
    coh, X = load()
    d, p = fitted(K, q)
    Z = (X - d["mu"]) / d["sd"]
    B = splines.basis(age_for(coh, CLOCK))
    R = np.asarray(mfa.responsibilities(Z, B, p))
    g = R.argmax(1)
    small = int(np.argmin(np.bincount(g, minlength=K)))
    S = g == small
    flagged = set(pd.read_csv(paths.PROJECT / "results" / "bin10_no7" / "excluded_infants.csv")
                  .dhcp_id.astype(str))
    is7 = coh.dhcp_id.astype(str).isin(flagged).values

    # how far each infant sits from the one-group age curve, per lobe (K = 1 fit)
    d1, p1 = fitted(1, sel["selected"]["q"])
    mu1 = B @ np.asarray(p1.beta[0])
    z1 = (X - d1["mu"]) / d1["sd"] - mu1
    out = dict(K=K, q=q, n=len(coh), n_small=int(S.sum()), small_share=float(S.mean()),
               mean_responsibility_small=float(R[S, small].mean()) if S.any() else None,
               n7_in_small=int((S & is7).sum()), n7_total=int(is7.sum()),
               lobe_mean_z_small=dict(zip(GROUP_NAMES, np.round(z1[S].mean(0), 2).tolist())),
               lobe_mean_z_rest=dict(zip(GROUP_NAMES, np.round(z1[~S].mean(0), 2).tolist())),
               max_abs_z_small_median=float(np.median(np.abs(z1[S]).max(1))),
               max_abs_z_rest_median=float(np.median(np.abs(z1[~S]).max(1))))
    cmp = {}
    for col in ("scan_age", "birth_age", "qc_dmri_eddy_rel_motion", "radiology_score"):
        v = coh[col].values.astype(float)
        a, b = v[S & np.isfinite(v)], v[~S & np.isfinite(v)]
        cmp[col] = dict(small=float(np.median(a)), rest=float(np.median(b)),
                        p=float(mannwhitneyu(a, b).pvalue))
    cmp["preterm_share"] = dict(small=float(coh.preterm.values[S].mean()),
                                rest=float(coh.preterm.values[~S].mean()))
    out["small_vs_rest"] = cmp
    (paths.RESULTS / "group_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
