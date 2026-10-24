# Code and methods audit — 2026-10-24

A review of the code against the methods plan after the first block of
results (SUPPLEMENT S1–S28), before further analysis. Each finding carries a
severity, the evidence, and the action taken or proposed. Checks were run on
the training infants only.

## 1. Verified correct

| component | check |
|---|---|
| E-step (`model/mfa.py`) | Woodbury quadratic form and log-determinant match the closed forms; validated to 8 d.p. against dense algebra (S11) |
| M-step | spline and loading blocks solved jointly; the score block uses E[θθ′] = G + mm′ (Rubin & Thayer), as the plan specifies; the penalty is scaled by ψ, per the plan's prior (raised in review); ψ includes the score-uncertainty trace term; λ is the Gamma posterior mode with rank M − 2 |
| ECM monotonicity | penalised log likelihood never decreases (S11) |
| Splits | 670 infants (earliest scan each), 569 training / 101 validation, disjoint; 5 folds of 113–114, every infant in exactly one fold, balanced on PMA (39.3–39.7) and preterm share (28–30%) |
| Standardisation | per coordinate, fit-fold moments only; validation scored with training moments |
| Per-infant scores | sum exactly to the fold totals (gate, S11) |
| Connectomes | zero diagonals, so strength and degree are not inflated by self-loops |
| Validation set | scored exactly once (S17); the script refuses a rerun |

## 2. Findings

### A. HIGH — Axis 1 partly tracks each infant's maximum edge weight

Onnela clustering divides every weight by the infant's single largest edge
weight before taking geometric means, so each region's clustering scales as
1 / max W. Log clustering therefore shifts by −log(max W) in all 90 regions at
once, which is exactly a brain-wide "clustering everywhere" axis.

After removing age (max W rises with PMA, r = 0.78):

| | axis 1 | axis 2 | axis 3 |
|---|---|---|---|
| r with age-adjusted log max edge weight, primary networks | **−0.62** | +0.50 | −0.19 |
| same, unthresholded networks | **−0.71** | +0.57 | −0.08 |

So roughly 40–50% of axis 1's between-infant variance follows one number per
infant, set by one edge. This also explains why axis 1 survived, and grew,
without thresholding (S26): the maximum weight varies there too. Axis 2's
correlation probably reflects an overall per-infant weight scale (strength is
in raw weight units), which may be biological (e.g. brain size) or
acquisition-related.

**Action (proposed, task 1):** recompute clustering with a normaliser that is
identical for every infant (e.g. the cohort-wide maximum, or a fixed robust
constant), refit the one-group model, and test whether axis 1 survives. In
parallel, test axis 2 against total edge weight and head circumference, and
decide whether a per-infant global weight scale is signal or nuisance.

### B. MEDIUM — Axes are eigenvectors of WW′, but the plan specifies C = WW′ + Ψ

The plan (§5) reports the eigendecomposition of Cₖ. The code uses the
principal axes of W. Measured difference: directions agree to
|cos| ≥ 0.994 for axes 1–5 at q = 15 and q = 70; variance shares are
~0.2 points higher under C (axis 1: 6.75% → 6.98%). No conclusion changes.

**Action (task 9):** switch reporting to eigenvectors and eigenvalues of C.

### C. LOW — Sign convention differs from the plan

The plan makes each eigenvector's largest-magnitude coordinate positive; the
code makes loadings sum positive. Only axis 4 of the q = 15 model would flip
(none at q = 70). This is cosmetic, but colours and correlate signs depend on it.

**Action (task 9):** adopt the plan's convention when reporting switches to C.

### D. LOW — Zero clustering values become extreme outliers

0.12% of log-clustering values, in 15 regions, sit at the log floor
(log 10⁻⁴), many SDs from the rest. A Gaussian model is sensitive to such
points.

**Action (task 1):** handle together with the clustering normaliser, e.g. a
floor tied to the distribution, or a check that results are unchanged when
these values are winsorised.

### E. LOW — Initialisation removes a spline trend, not the plan's linear trend

The plan removes a linear age effect before k-means; the code removes the
fitted spline trend. This is arguably better (initial groups cannot reflect
curvature in age either) and affects starting values only. The docstring
wrongly said "linear".

**Action (done):** docstring corrected; deviation recorded here.

### F. LOW — Collapsed groups are kept, not discarded and re-initialised

The plan discards and re-initialises solutions whose group collapses. The code
keeps them and flags them (S15: 10 specifications at K ≥ 3). The K = 1 vs
K = 2 conclusion is unaffected, because no K = 2 fit collapsed (smallest group
9–29%), but scores at K ≥ 3 are of effectively smaller models.

**Action (task 6):** implement re-initialisation if K ≥ 3 results are reported
in detail.

### G. LOW — Fixed iteration counts, not a stopping rule

The plan stops at a relative change of 10⁻⁶. The code runs a fixed number of
sweeps and flags non-convergence at the end. All K ≤ 3 specifications in the
final comparison converged, and scores moved < 20 log units between 250 and
1000 sweeps (S15).

**Action:** none needed; documented.

### H. LOW — The permutation null uses 5 starts; observed fits use 20

Under-optimised null fits could understate null axis sizes, which would make
the test anti-conservative. The margin for axes 1–3 is large (6.8% vs a null
maximum of 0.79%), so this matters only for the smallest axes.

**Action:** none needed for current claims.

## 3. Coverage against the methods plan

| plan item | status |
|---|---|
| §1 regional measures, standardisation | done, with documented overrides (D5, D6, D10) |
| §1 15% validation set, scored once | done (S17) |
| §1 refit under alternative preprocessing | unthresholded done (S26); **binarised variants not done** (features exist) |
| §1 PMA/GA distributions of excluded infants | **to check**; the cascade (S1/S9) is ab15's QC |
| §2 model, P-splines, Gamma(0.001, 0.001), rank M − 2 | done; M = 20, knots 26.5–45.5 (S8) |
| §3 EM details, 20 starts, k-means on de-trended residuals | done (findings E–G) |
| §3 5-fold CV by infant, stratified | done |
| §3 greedy step 1: q at K = 1 | done, extended to q = 70 (S16) |
| §3 **step 2: K × covariance sharing** (W and Ψ shared; W shared) | **not done**: only group-specific covariance implemented |
| §3 **step 3: trajectory shapes** (linear; shared shape + offsets) | **not done** |
| §3 step 4: revisit q at the selected K | done (S16) |
| §3 **step 5: developmental clocks** (PMA, PNA, both) | **not done**, and relevant: axes correlate with age since birth (S14) |
| §3 **simulate from the fitted model; check recovery of K and q** | **not done** (only generic synthetic checks, S11) |
| §4 parametric bootstrap | done, 1,000 replicates (S25) |
| §4 **bootstrap intervals for trajectories** | **not done**; replicate trajectories are saved, so this is cheap |
| §4 bootstrap principal angles for leading axes | **not done**; computable from saved replicates |
| §5 axes from the eigendecomposition of C, sign rule | deviation (findings B, C) |
| §5 **trajectories at three levels, brain maps at four ages** | **not done** (research question 2) |
| §5 axes as 90 × 3 maps with variance shares | done (figures 3–6) |
| §5 extensions: covariates | partly (S14) |
| §5 **extensions: 18-month outcomes, polygenic scores** | **not done** |
| ANALYSIS_PLAN item 16: preterm ARI at every K | **not done** (K = 1 selected, so a diagnostic only) |

## 4. Next tasks, in priority order

**Correctness of what we already claim**
1. **Clustering normalisation** (finding A, with D): recompute clustering with
   a cohort-wide normaliser, refit K = 1 at q = 15 and 70, and repeat
   K = 1 vs 2 at q = 15 and 25. Does axis 1 survive? (~50–150 core-hours)
2. **Global weight scale** (finding A): test axis 2 against total edge weight
   and head circumference; decide whether strength should be relative.

**Completing the planned model search** (the plan's greedy steps)

3. Covariance-sharing variants for K ≥ 2: constrained M-steps, synthetic
   validation, then the K × sharing profile.
4. Trajectory shapes: linear vs smooth at K = 1, and shared shape + offsets
   for K ≥ 2.
5. Developmental clocks: PNA and additive PMA + PNA bases, with the reference
   profile PMA 40 / GA 40 / PNA 0.
6. Recovery simulation from the selected fit; re-initialise collapsed groups
   (finding F); preterm ARI for any K ≥ 2 fit.

**Reporting the three research questions**

7. **Question 2, development:** mean trajectories with bootstrap bands, per
   measure, by lobe, and as brain maps at four ages, from the saved replicates.
8. Bootstrap principal angles for axes 1–3, from the saved replicates.
9. Switch axis reporting to the eigendecomposition of C with the plan's sign
   rule (findings B, C); regenerate figures 3–6 and 8.
10. Binarised-network sensitivity refit.

**Extensions**

11. Axis scores against 18-month CBCL and ECBQ.
12. Axis scores against polygenic scores (EUR-only, 415 infants, ancestry PCs
    as covariates).
13. Further covariates: birth weight, head circumference, deprivation.

**Note on the validation set.** It has been scored once, for the current
feature definition. If task 1 changes the features, the reselected model
cannot be honestly validated on the same infants again. Any re-scoring must
be reported as such.
