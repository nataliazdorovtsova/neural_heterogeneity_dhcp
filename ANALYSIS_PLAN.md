# Analysis plan

Ordered so that each stage is checkable before the next depends on it.
Decisions behind these choices are in `DECISIONS.md`; the governing documents
are transcribed in `notes/`.

---

## Stage 0 — Blocked, chase in parallel

| Item | Needed for | Status |
|---|---|---|
| Fixed-density rebuild of the stack (D4) | Stage 2 onward | asked ab15 — **the only open blocker** |
| ~~Ancestry principal components~~ | ~~Stage 7~~ | **resolved** — `paths.ANCESTRY_PCS` (D9) |
| ~~Motion exclusion rule~~ | ~~Stage 1~~ | **resolved** — none needed (D8) |

Neither blocks Stages 1–6. Build on the current stack, carry density as a
covariate, and re-run when the rebuild lands — the pipeline should take the
stack as a parameter so this is a re-run, not a rewrite.

---

## Stage 1 — Cohort and the frozen split

1. **Define the cohort.** Earliest scan per `dhcp_id` → 670 infants (D3). Keep
   the 753-session version alongside as the sensitivity cohort.
2. **No motion exclusions** (D8). Motion was handled at source and the dHCP QC
   gate already removed 129 of 887 sessions. Carry `qc_dmri_eddy_rel_motion`
   through as a covariate for the Stage 6 artefact check, not as a filter.
   The exclusion cascade to report is therefore ab15's, not ours.
3. **Freeze the validation split.** 15% (~100 infants), stratified by PMA band
   and preterm status, **grouped on `dhcp_id`**. Write it to
   `results/cohort/validation_split.csv` and never regenerate it.
4. **Freeze the CV folds.** 5 folds, same stratification, same grouping. Also
   written to disk.

**Deliverable:** `scripts/build_cohort.py` → a cohort table with the exclusion
cascade, plus the frozen split and fold assignments.

**Gate:** exclusions must not be strongly age-dependent; splits must contain no
`dhcp_id` in more than one fold.

---

## Stage 2 — Features

5. **Compute the 90 × 3 tables** (D5): strength; Onnela clustering, per-subject
   normalisation; log of degree-normalised communicability, diagonal excluded.
6. **Preprocessing variants**, each a separate feature set: per-bin 25% + MST
   (primary), unthresholded, and binarised versions of each.
7. **Standardise** each of the 270 coordinates using **training-fold moments
   only** (D6). Validation and test coordinates use the training moments.

**Deliverable:** `scripts/extract_features.py` → `(n, 90, 3)` arrays per variant,
plus the standardisation moments.

**Gate:** log-communicability residuals adequately symmetric (§1 asks for this);
no coordinate with zero variance; feature matrices align with the cohort table
by `dhcp_id`.

---

## Stage 3 — Exploration, training set only

8. Pairwise correlations among the three measures across regions — **in
   particular strength vs communicability**, which degree normalisation couples.
   If near-collinear, two of the three measures are one measure.
9. Marginal distributions; low-dimensional projection coloured by PMA and by
   preterm status.
10. **Density sensitivity**: recompute 8–9 with density regressed out, and
    confirm how much of each age trend survives.

**Gate:** nothing here touches the validation set.

---

## Stage 4 — Model

11. **Implement the mixture of factor analysers with P-spline centres.** See the
    note on EM below. Woodbury and the determinant lemma throughout, so
    covariance work stays at q × q.
12. **Fix the basis**: knots spanning **26.5–45.5** weeks (observed range is
    26.71–45.14, so 26–45 extrapolates). Raise M from 8 to ~15–20 if we want a
    genuine P-spline, or document it as a regression spline.
13. **Get the penalty scaling right** (raised in review): the λD₂ᵀD₂ term must carry the ψ_kd
    factor. Standardisation (D6) reduces the damage if this is wrong, but it
    should be right.
14. **Validate on synthetic data before any real fit** — simulate from the
    generative model at known K, q and trajectory shape, and confirm recovery.

**Gate:** parameter recovery on synthetic data; held-out log predictive density
computed two independent ways agrees.

---

## Stage 5 — Selection

15. **Greedy search**, ~32 specifications in the five documented steps, scored by
    held-out log predictive density under the frozen folds.
16. **Preterm diagnostic at every K**: adjusted Rand index between the fitted
    partition and preterm status. If K = 2 simply recovers preterm/term, that is
    a replication of Mousley et al. (2025), not a finding — say so, and ask
    whether structure survives conditioning on GA at birth.
17. **Recovery check**: simulate from the selected fit and confirm the search
    recovers the generating K and q rather than trading one against the other.
18. **Score the selected specification once** on the reserved validation set.

**Gate:** profiles over q and K reported, flat ones called weak.

---

## Stage 6 — Stability and reporting

19. **200 parametric bootstrap refits** of the selected specification →
    pointwise intervals for trajectories, eigenvalues, eigenvectors, group
    differences.
20. **ARI done properly** (raised in review): assign the **original** infants to each
    bootstrap-fitted model by responsibility, then compare partitions. For
    K-recovery, rerun *selection* on ~50 replicates — refitting at fixed K
    cannot test it.
21. **Motion artefact check** (D8): correlate each infant's position along each
    axis of Cₖ with residual motion. An axis tracking motion is scan quality,
    not biology — log-communicability carries r ≈ −0.24 with motion net of age.
22. **Order groups by mixing proportion π**, not by mean-across-coordinates
    (scale-dominated, D6).
23. **Reference profile** for centres under the postnatal/both-clock models
    (raised in review): PMA 40 / GA 40 / PNA 0.
24. Report trajectories at three levels, axes as brain maps with variance
    shares, group differences against age.

---

## Stage 7 — Extensions

24. Group membership and axis positions vs covariates outside the model: sex,
    birth weight, head circumference, deprivation.
25. **18-month outcomes** — CBCL and ECBQ, 522 of 670 infants.
26. **Polygenic scores** — 415 of 670; 341 have both PGS and CBCL. **Read in
    place, never copied** (D1). Blocked on ancestry PCs.

---

## Note: EM or MCMC?

The methods supplement specifies **EM** (§3), and for the sweep it is clearly
right. The greedy search is ~32 specifications × 5 folds × 20 initialisations ≈
**3,200 fits**; NUTS at that volume is not feasible, EM comfortably is. Rough
cost per iteration at N≈570, D=270, q≤25, K≤5 is O(NKDq) + O(KD(M+q)³) ≈ 70
MFlop, so the whole sweep is minutes-to-hours of real compute on 32 cores, not
days. **Benchmark and report actual numbers** rather than restating this — the review
removed runtime promises for good reason.

JAX remains the right substrate: vectorised EM with Woodbury, `jit`-compiled,
using all cores. Nothing about the environment choice changes.

**Where MCMC earns its place** is Stage 6. The parametric bootstrap gives
intervals *conditional on the selected model* and is explicitly acknowledged not
to cover selection uncertainty. Running NUTS on the **selected specification
only** would give genuine posterior intervals for trajectories and Cₖ at a cost
of a handful of fits, not thousands. Worth doing alongside the bootstrap and
comparing — if they disagree materially, that is itself informative.

---

## Suggested first deliverable

**Stages 1 and 2**: the cohort with its exclusion cascade, the frozen split, and
the feature tables. Deterministic, fully specified now, and everything
downstream depends on them. It also produces the numbers the proposal's Data
section still needs.
