"""Score the selected model once on the reserved validation infants (plan, section 3).

    python scripts/validate_once.py

The 101 validation infants have not been used for anything so far: not for
fitting, standardisation, selection or figures. They are scored exactly once,
after selection, with models fitted on all 569 training infants and
standardised with the training moments:

  - the selected model (results/selection.json, "selected")
  - the strongest model with a different number of groups ("runner_up")
  - the null baseline (each measure's training mean and SD only)

If the selected model's advantage in cross-validation was real rather than an
artefact of the selection, it should also beat the runner-up and the baseline
on these new infants. The script refuses to run twice unless given --force, so
that the result cannot be quietly re-scored.

Writes results/validation/validation_score.json.
"""

import argparse
import json

import numpy as np
import pandas as pd

import paths
import provenance as prov
from model import mfa, splines
from scripts.run_spec import VARIANT, CLOCK

OUT = paths.RESULTS / "validation"


def load_validation():
    coh = pd.read_csv(paths.RESULTS / "cohort" / "cohort.csv")
    feat = np.load(paths.RESULTS / "features" / f"{VARIANT}.npz")
    X = feat["X"].reshape(len(feat["X"]), -1)
    m = coh.in_primary.values & (coh.split == "validation").values
    return coh[m].reset_index(drop=True), X[m]


def fitted(K, q):
    d = np.load(paths.RESULTS / "final" / f"K{K}_q{q}.npz")
    return d, mfa.Params(log_pi=d["log_pi"], beta=d["beta"], W=d["W"], psi=d["psi"],
                         lam=d["lam"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    target = OUT / "validation_score.json"
    if target.exists() and not a.force:
        raise SystemExit(f"{target} exists: the validation set has already been scored once")

    sel = json.loads((paths.RESULTS / "selection.json").read_text())
    coh, Xraw = load_validation()
    assert len(coh) == 101, f"expected 101 validation infants, found {len(coh)}"
    t = coh.scan_age.values if CLOCK == "pma" else coh.pna.values
    B = splines.basis(t)
    n_clipped = int(((t < splines.LO) | (t > splines.HI)).sum())

    scores = {}
    models = {"selected": sel["selected"], "runner_up": sel["runner_up"]}
    ref = None
    for role, spec in models.items():
        if spec is None:
            continue
        d, p = fitted(spec["K"], spec["q"])
        if ref is None:
            ref = (d["mu"], d["sd"])
        assert np.allclose(d["mu"], ref[0]) and np.allclose(d["sd"], ref[1])
        X = (Xraw - d["mu"]) / d["sd"]                     # training moments only
        scores[role] = dict(K=spec["K"], q=spec["q"],
                            lpd=np.asarray(mfa.pointwise_log_density(X, B, p)))
    Z = (Xraw - ref[0]) / ref[1]
    null = -0.5 * (np.log(2 * np.pi) + Z ** 2).sum(1)

    n = len(coh)
    out = dict(n=n, ages_outside_basis=n_clipped, null_total=float(null.sum()))
    for role, s in scores.items():
        d = s["lpd"] - null
        out[role] = dict(K=s["K"], q=s["q"], total=float(s["lpd"].sum()),
                         gain_per_infant=float(d.mean()),
                         gain_se_per_infant=float(d.std(ddof=1) / np.sqrt(n)))
    if "runner_up" in scores:
        d = scores["selected"]["lpd"] - scores["runner_up"]["lpd"]
        out["selected_minus_runner_up"] = dict(
            total=float(d.sum()), se=float(np.sqrt(n) * d.std(ddof=1)),
            per_infant=float(d.mean()), infants_favouring_selected=int((d > 0).sum()))
    # The same quantity in cross-validation, for comparison
    cv = sel["selected"]["gain_per_infant"]
    out["cv_gain_per_infant_selected"] = cv

    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    prov.snapshot(name="validation_score", config=dict(selection=sel["selected"],
                  runner_up=sel["runner_up"], scored_once=True),
                  results=out, inputs=[paths.RESULTS / "selection.json",
                                       paths.RESULTS / "cohort" / "cohort.csv"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
