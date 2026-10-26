# Methodological supplement — empirical justification

Every methodological choice in this project that was settled by looking at the
data rather than by convention, with the numbers behind it. Written to be lifted
into a paper supplement.

`DECISIONS.md` records *what* we decided. This records *why*, with evidence.
Sections are appended as decisions are made; numbers are never edited in place —
if something is superseded, the old entry is marked and the new one appended.

Unless stated otherwise, statistics are computed on the **training set only**
(569 infants, earliest scan each) to avoid informing choices with held-out data.

**Contents**

| § | Topic | Settles |
|---|---|---|
| S1 | Sample structure: sessions vs infants | split grouping, cohort definition |
| S2 | Density is confounded with age and preterm status | thresholding, covariates |
| S3 | The matrix exponential needs degree normalisation | communicability definition |
| S4 | Measures differ in scale by five orders of magnitude | standardisation |
| S5 | Which measure needs the log transform | transforms |
| S6 | The three measures are not redundant | measure set |
| S7 | Residual motion is not a design confound | exclusions |
| S8 | Spline basis size M | M = 20 |
| S9 | Cohort and frozen splits | split validity |
| S10 | Genetic data coverage | extension sample |
| S11 | Model implementation is correct | validity of everything downstream |
| S12 | Figure palettes | colour choices are computed, not eyeballed |
| S13 | Scores as gain per infant over a null baseline | how model scores are reported |
| S14 | The within-population axes (K = 1, q = 15) | size, content, stability across q, correlates |
| S15 | K × q sweep, final (fold totals) | the grid, convergence reruns |
| S16 | Paired per-infant model comparison and selection | K = 1; q not bracketed (best tested q = 70) |
| S17 | Validation set, scored once | the selection generalises |
| S18 | Permutation null (parallel analysis) | every axis is bigger than noise |
| S19 | Split-half replication | which axes reproduce in independent infants |
| S20 | Parametric bootstrap of the selected model | loading and axis-size intervals |
| S21 | Brain surface renders | how the maps are drawn, atlas check |
| S22 | Compute ledger | core-hours for every run |
| S23 | What the axis tail encodes; split-half at q = 70 | why interpretation stops at three axes |
| S24 | Nodal measures: robustness, redundancy, shape | whether to keep communicability; clustering's fragility |
| S25 | Bootstrap and permutation null extended to 1,000 each | final intervals and p-values; supersedes the counts in S18 and S20 |
| S26 | Axes refitted on unthresholded networks | axes 1–3 are not a thresholding artefact |
| S27 | Measure sets: communicability dropped or replaced by eigenvector centrality | conclusions do not depend on communicability; eigenvector centrality changes the group question |
| S28 | What separates the eigenvector-set "groups" | a left/right localisation of eigenvector centrality, not infant subtypes |
| S29 | Axis 2 and the per-infant weight scale | axis 2 is the infant's total connection weight |
| S30 | Developmental trajectories (research question 2) | strength doubles 30→42 weeks; cortical vs subcortical divergence |
| S31 | Axes as eigenvectors of C; bootstrap principal angles | plan-compliant reporting; axes 1–3 stable within 12–21° |

**Tests at a glance** — every check run on the model and its outputs:

| test | question | result | § |
|---|---|---|---|
| Woodbury / determinant lemma vs dense algebra | is the likelihood computed correctly? | agree to 8 d.p. | S11 |
| ECM monotonicity | is the fitting algorithm correct? | no decrease in 120 sweeps | S11 |
| Parameter recovery on simulated data | can the model recover known structure? | ARI 1.000; centres within 5.1% | S11 |
| Selection recovers the generating K | is the selection criterion biased? | K = 2 recovered | S11 |
| Per-infant scores sum to the total | is the per-infant refactor correct? | exact (|Δ| < 10⁻⁶) | S11 |
| Convergence rerun, 250 → 1000 sweeps | were fold scores stable? | lpd changed −152 to +96; no ranking changed | S15 |
| Axis stability across q (5, 15, 25) | do the leading axes depend on q? | axes 1–3 identical (|cos| ≈ 1.00) | S14 |
| Covariate correlations (Bonferroni, 42 tests) | are axes technical or biological? | density, motion and age-since-birth links | S14 |
| Paired per-infant comparison (SE of difference) | which model predicts unseen infants best, beyond noise? | K = 1 at q ≥ 15 (z = 7.6–14.5); K = 2 at q ≤ 5 | S16 |
| One-standard-error rule | simplest adequate model? | K = 1, q = 70 (grid not bracketed) | S16 |
| Reserved validation set (scored once) | does the selection generalise? | gain 104.5 vs 106.1 in CV; beats K = 2 by z ≈ 8 | S17 |
| Permutation null, 100 shuffles | is each axis bigger than noise? | all 15 axes, smallest 0.94% vs null ≤ 0.78% | S18 |
| Split-half replication, 10 splits | do axes reproduce in independent infants? | axes 1–3 yes (0.91, 0.81, 0.74); 4–8 weakly | S19 |
| Parametric bootstrap, 200 replicates | how precise are loadings and axis sizes? | 64–78% of loadings exclude zero (axes 1–8) | S20 |
| Display vs connectome atlas centroids | are regions drawn in the right place? | r = 0.999 / 0.996 / 0.994 | S21 |
| Permutation null, 1,000 shuffles | corrected significance of each axis | all 15 axes p < 0.001, family-wise corrected | S25 |
| Parametric bootstrap, 1,000 replicates | final loading and axis-size intervals | stable vs 200; 0.7–4% of calls changed | S25 |
| Warm-start-only refits (gate) | can bootstrap refits drop the random starts? | no: loadings differ by up to 0.03; not used | S25 |
| Measure set: strength + clustering only | do conclusions need communicability? | no: same axes (0.99 / 0.99 / 0.94); K = 1 (z = 9.5, 18.5) | S27 |
| Measure set: eigenvector centrality instead of communicability | do conclusions survive a different third measure? | axes 1–3 present (0.94 / 0.98 / 0.83), but a new 18% eigenvector axis; K = 2 at q = 15 (z = −4.0), tie at q = 25 | S27 |
| Eigenvector set, K = 1 vs 2 at higher q | does one group win once there are enough axes? | yes: tie at q = 25, then K = 1 at q = 35 / 50 / 70 (z = 5.8 / 13.1 / 19.7) | S28 |
| Eigenvector-set K = 2 groups inspected | are the groups infant subtypes? | no: they split on which hemisphere holds the eigenvector centrality (r = 0.81; dominant hub V = 0.77); no age, prematurity, sex or density difference | S28 |
| Axes vs per-infant weight scale (age-adjusted) | is axis 2 a global scale? | yes: r = 0.89 with total edge weight; axis 1: r = −0.62 with max edge weight | S29 |
| Bootstrap principal angles | how stable are axis directions? | median 12° / 19° / 21° (axes 1–3); subspace 1–3 median 17° | S31 |
| Refit on unthresholded networks | are axes 1–3 a thresholding artefact? | no: same axes (\|cos\| 0.89 / 0.91 / 0.71), same infants (r = 0.88 / 0.97 / 0.82) | S26 |
| Anatomy of the axis tail | what do axes beyond 3 encode? | increasingly local: within-region coupling 33× chance in axes 21–70 | S23 |
| Split-half replication at q = 70 | does the three-axis conclusion hold in the selected model? | axes 1–3: 0.91 / 0.80 / 0.73; tail replicates as blocks only | S23 |
| Measure robustness to thresholding | which measures depend on preprocessing? | clustering 0.22; communicability 0.84; strength 0.92 | S24 |
| Measure redundancy and communality | does communicability add information? | \|r\| = 0.77 with strength; 5% of its variance on axes 1–3 | S24 |

---

## S1 Sample structure: 753 sessions from 670 infants

*Produced by: `scripts/build_cohort.py`. Settles DECISIONS.md D3.*

The delivered stack contains **753 sessions from 670 unique infants — 83 repeat
scans**. The methods supplement's premise that "each of N infants contributes
one usable connectome" does not hold as delivered.

The repeats are not random: **82 of 83 repeat sessions (99%) are preterm
infants**, scanned once early and again near term-equivalent age (median gap
7.3 weeks, maximum 16.7). Within-infant dependence is therefore almost entirely
confined to the preterm group — the same contrast most likely to drive cluster
structure.

Deduplication also affects whether the two developmental clocks are separable:

| cohort | n | corr(PMA, PNA) |
|---|---|---|
| all sessions | 753 | +0.164 |
| **earliest scan per infant** | **670** | **+0.039** |
| latest scan per infant | 670 | +0.233 |

The earliest-scan cohort gives near-orthogonal clocks, so postmenstrual and
postnatal age contributions are separately identifiable. This is the primary
cohort; all 753 sessions are retained as a sensitivity analysis.

**Consequence for inference.** Cross-validation folds and the held-out
validation split are assigned at infant level on `dhcp_id`. Because held-out
predictive density is the sole criterion for selecting K, q, covariance sharing,
trajectory shape and developmental clock, session-level splitting would place
repeat scans of one infant in both training and test, inflating the score and
biasing selection toward more complex models.

---

## S2 Density is confounded with age and preterm status

*Produced by ad-hoc analysis, 2026-10-08. Settles DECISIONS.md D4. **Open: a
fixed-density rebuild has been requested.***

The thresholding rule (per-bin top 25% + MST, Betzel distance bins) retains the
top 25% of each subject's **observed** edges. Pre-threshold density declines
steeply with age, and that confound passes through almost unchanged:

- `corr(retained edges, pre-threshold edges) = +0.999`
- post-threshold density: mean 0.218, SD 0.013, **range 0.178–0.251**
- `corr(density, PMA) = −0.669` (pre-threshold: −0.688)
- preterm 0.2249 vs term 0.2144, t = 11.57, p = 1.3 × 10⁻²⁸

Taking 25% of *possible* rather than observed connections within each bin would
retain 1001 edges for every subject, fixing density by construction.

Effect on the three regional measures (150 sessions):

| measure | r(age) | r(density) | r(age \| density) |
|---|---|---|---|
| strength | +0.891 | −0.734 | +0.770 |
| **clustering** | **+0.133** | −0.418 | **−0.264** |
| log-communicability | +0.067 | −0.036 | +0.059 |

**The age trend in clustering reverses sign once density is controlled.**
Uncorrected, segregation appears to increase mildly across the perinatal period;
net of density it decreases. Any developmental claim about clustering is unsafe
until density is fixed or adjusted.

Note: the `density` column distributed with the participants file is the
*pre*-threshold figure (mean 0.854). Post-threshold density must be computed
from the stack.

---

## S3 The matrix exponential requires degree normalisation

*Settles DECISIONS.md D5.*

The methods supplement defines pairwise communicability as an entry of the
matrix exponential of the adjacency matrix, without normalisation. Applied to
µ-weighted matrices the result is dominated by exp(λ_max):

| | max value |
|---|---|
| `expm(W)`, thresholded stack | 4.95 × 10⁷ |
| `expm(W)`, unthresholded stack | 3.56 × 10⁸ |
| `expm(D^−1/2 W D^−1/2)` | **1.24** |

Independent confirmation comes from ab15's own earlier topology run, which
computed subgraph centrality — the diagonal of `expm(A)` — on raw weights:
mean 1.15 × 10⁹, SD 4.14 × 10⁹, maximum 6.32 × 10¹⁰. A standard deviation 3.6
times the mean, spanning four orders of magnitude, is not a usable per-subject
feature.

We therefore use the degree-normalised form (Crofts & Higham, 2009), taking the
regional mean over off-diagonal entries only, since self-communicability
dominates the row mean. **Crofts & Higham should be added to the reference
list.**

---

## S4 The measures differ in scale by five orders of magnitude

*Settles DECISIONS.md D6, overriding the methods supplement.*

Measured on 100 infants, before any transform:

| measure | mean | SD |
|---|---|---|
| strength | 17.49 | 10.61 |
| clustering | 0.0189 | 0.0124 |
| log-communicability | −4.16 | 0.359 |

**Variance ratio, strength to clustering: 733,000 : 1.**

The supplement declines to standardise on the grounds that factor analysis
estimates a separate residual variance per coordinate. That argument holds for
the *likelihood* but not for the reported quantities:

- "each measure averaged across regions", and the group-ordering rule at 40
  weeks, are numerically dominated by strength alone;
- eigenvectors of Cₖ are not scale-invariant, so the leading axis of within-group
  variation would be essentially the strength block — the segregation/integration
  contrast the study sets out to find could not emerge;
- the shared-precision smoothness variant is not scale-invariant;
- initialising loadings from principal components of unstandardised residuals
  makes the initialisation a strength detector.

Each of the 270 coordinates is therefore z-scored using **training-fold moments
only**. This also reduces the consequences of misplacing the ψ factor in the
penalised spline solve (raised in review), since residual variances become
comparable across coordinates.

---

## S5 The log belongs on clustering, not communicability

*Produced by: `scripts/extract_features.py`. Settles DECISIONS.md D10,
overriding the methods supplement.*

The supplement log-transforms communicability "because its raw distribution is
strongly right-skewed", and leaves clustering untransformed. After adopting
degree normalisation (S3) this is the wrong way round. Median |skew| across the
90 regions, 569 training infants:

| coordinate | median \|skew\| |
|---|---|
| communicability, raw | **0.41** |
| communicability, log-transformed | 0.67 *(the transform makes it worse)* |
| strength, raw | 0.26 |
| **clustering, raw** | **1.51** — 33 of 90 regions exceed \|skew\| 2 |
| clustering, log-transformed | **0.24** |

The supplement's reasoning was sound for *unnormalised* communicability (S3).
Normalisation bounds the measure and removes the skew, eliminating the premise.

**Adopted:** communicability enters untransformed; clustering enters as
log(c + 10⁻⁴). The offset lies below the smallest observed non-zero value
(4.6 × 10⁻⁴), leaving real values undistorted while keeping the 62 exact zeros
(0.12% of values) finite. The offset was chosen from a sweep over 10⁻⁶ to 10⁻³,
across which skew varied only between 0.26 and 0.28.

Resulting median |skew| on the training set:

| | strength | clustering | communicability |
|---|---|---|---|
| as specified in the supplement | 0.26 | 1.51 | 0.67 |
| **as adopted** | **0.26** | **0.28** | **0.41** |

Both transforms remain switchable in `features.regional_features()`.

---

## S6 The three measures are not redundant

*Settles DECISIONS.md D11.*

Degree-normalising communicability (S3) incorporates strength into it, raising
the possibility that two of the three measures carry the same information.
Across-infant correlation computed separately within each of the 90 regions,
training set:

| pair | median r | IQR |
|---|---|---|
| strength vs communicability | +0.595 | [+0.39, +0.74] |
| strength vs clustering | +0.066 | [−0.04, +0.18] |
| clustering vs communicability | −0.029 | [−0.19, +0.15] |

At the level of region-averaged means, r(strength, communicability) = +0.29.
Strength and communicability share moderate variance without approaching
collinearity; clustering is close to orthogonal to both. The three-measure
specification is retained.

---

## S7 Residual motion is not a design confound

*Settles DECISIONS.md D8.*

Motion was corrected at acquisition and reconstruction (SHARD; Christiaens et
al., 2021) and the dHCP quality gate removed 129 of 887 sessions on diffusion
quality. No further exclusions are applied. Residual
`qc_dmri_eddy_rel_motion` on retained sessions (745 of 753 have a QC record):
median 1.56, 95th percentile 3.94, maximum 16.3.

| | r |
|---|---|
| motion vs PMA | −0.141 |
| motion vs GA at birth | −0.173 |
| preterm vs term (median) | 1.61 vs 1.52 |

Motion is **not** meaningfully associated with age or preterm status, so unlike
density (S2) it cannot imitate a developmental trajectory or group structure.
This is what makes the absence of a motion exclusion defensible.

It is not inert, however:

| measure | r(motion) | r(motion \| age) |
|---|---|---|
| strength | −0.143 | +0.046 |
| clustering | +0.099 | +0.121 |
| **communicability** | **−0.266** | **−0.242** |

An association of approximately −0.24 with communicability survives age
adjustment. Because the second research question *is* the characterisation of
between-infant variation, an axis driven by scan quality could be recovered and
misread as biological. Motion is therefore retained as a post-hoc diagnostic:
each infant's position along each axis of Cₖ is correlated with residual motion,
and an axis tracking motion is reported as artefactual.

---

## S8 Spline basis size

*Settles the choice M = 20. A review comment asked whether M is fixed or
selected; it is fixed, on the evidence below.*

The developmental trajectories use cubic B-splines with a second-order
random-walk (P-spline) prior. M is the number of basis functions. Under the
P-spline construction of Eilers & Marx (1996), which the supplement cites, M is
set generously and the penalty controls smoothness; M = 8 as originally proposed
is small enough that the basis itself does the smoothing, making it a regression
spline with mild shrinkage.

Demonstration on region-averaged strength against postmenstrual age, fitted on
569 training infants and scored on the 101 held-out infants:

| setting | effective df | train RMSE | **validation RMSE** | wiggliness |
|---|---|---|---|---|
| M = 8, no penalty *(as proposed)* | 8.0 | 1.4026 | 1.6132 | 0.005 |
| M = 15, no penalty | 15.0 | 1.3982 | 1.6125 | 0.005 |
| M = 30, no penalty | 29.0 | 1.3712 | 1.6229 | **8.55** |
| M = 15, penalty λ = 1 | 9.0 | 1.4012 | 1.6103 | 0.004 |
| M = 15, penalty λ = 100 | 4.0 | 1.4282 | 1.6082 | 0.002 |
| **M = 30, penalty λ = 100** | **6.5** | 1.4083 | **1.6047** | 0.002 |
| M = 30, penalty λ = 10⁴ | 2.7 | 1.4470 | 1.6213 | 0.005 |

Without a penalty, increasing M improves training error, worsens validation
error, and inflates wiggliness by three orders of magnitude. With a penalty, a
30-function basis uses only 6.5 effective degrees of freedom and achieves the
best held-out score.

With the penalty active, predictive performance is insensitive to M:

| M | 10 | 15 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| validation RMSE | 1.627 | 1.608 | 1.604 | 1.605 | 1.607 |

a spread of 1.4%, against an effective degrees of freedom ranging from 3.0 to
8.0. The penalty, not M, is doing the work.

**Two features of this cohort reinforce the choice.** The age distribution is
markedly uneven — 74% of infants are 38 weeks or older and only 59 (8.8%) fall
below 34 weeks — so a single fixed stiffness must compromise between
data-dense and data-sparse regions, whereas a penalty shrinks toward linearity
where support is thin. And the model fits 270 such curves per group, each with
its own smoothness precision λₖd; a small basis leaves that per-coordinate
adaptation little room to act.

**Adopted: M = 20**, knots equally spaced over 26.5–45.5 weeks (the observed
range is 26.71–45.14; the originally proposed 26–45 span extrapolates for two
infants), with λ estimated per coordinate as specified.

For transparency: on the single coordinate tested, M = 8 unpenalised scored
1.6132 against 1.6047 for the best penalised fit — a difference of 0.5%. The
case for M = 20 rests on matching the cited method, on per-coordinate
adaptivity, and on behaviour where data is sparse, not on a large gain in this
one comparison.

---

## S9 Cohort and frozen splits

*Produced by: `scripts/build_cohort.py`; outputs in `results/cohort/`.*

| stage | n |
|---|---|
| imaging sessions released | 887 |
| T2w passing QC | 886 |
| dMRI passing QC | 758 |
| infants with both required modalities | 673 |
| connectome matrices constructed | 756 |
| excluded — isolated nodes surviving mask dilation | 3 |
| **connectomes delivered** | **753** |
| unique infants | 670 |
| **primary cohort (earliest scan per infant)** | **670** |

The cascade above the final two rows is ab15's pipeline, not ours; we apply no
further exclusions (S7).

Primary cohort: 194 preterm, 476 term; PMA 26.71–45.14 weeks; PNA 0–19.43 weeks.

**Splits**, assigned at infant level and frozen (seed 20261008): 101 infants
held out for final validation, 569 allocated to 5 cross-validation folds,
stratified by PMA quartile crossed with preterm status. No `dhcp_id` appears in
more than one fold or spans both splits. Validation and training PMA
distributions do not differ (Kolmogorov–Smirnov p = 0.98), so the hold-out is
not age-biased.

Age distribution of the primary cohort (2-week bands): 2, 7, 17, 33, 66, 50,
114, 201, 150, 30 from 26–28 through 44–46 weeks. Median 40.6.

---

## S10 Genetic data coverage

*Settles DECISIONS.md D9.*

Polygenic scores: 569 infants × 48 scores, no missing values, covering
psychiatric, neurological and imaging-derived phenotypes. Ancestry principal
components are supplied in two sets:

| file | n |
|---|---|
| all genotyped individuals | 766 |
| European-ancestry subset | **569** |
| polygenic score file | **569** |

The exact correspondence indicates the scores were computed for the
European-ancestry subset only; the EUR components are therefore the matching
covariate set. The first five components account for 21.9% of genotypic
variance and the first ten for 31.7%.

Coverage within the primary cohort of 670 infants:

| | n |
|---|---|
| polygenic scores | 415 |
| polygenic scores + ancestry components | 415 |
| 18-month CBCL | 522 |
| ECBQ | 520 |
| **polygenic scores + components + CBCL** | **341** |

Coverage is balanced across preterm status (63% vs 61%). **The genetic arm
covers 62% of the cohort and is restricted by ancestry; findings from it do not
automatically generalise to the full sample, and this should be stated
explicitly.**

---

## S11 Model implementation validated before use

*Produced by: `scripts/validate_model.py`. Gate for Stage 4.*

The mixture of factor analysers with P-spline centres is fitted by ECM, using
the Woodbury identity and matrix determinant lemma so that every covariance
operation stays at q × q. Three checks were run before the model was applied to
real data.

**Woodbury identities.** Log-determinant and quadratic form computed through the
low-rank decomposition agree with direct computation on the dense D × D
covariance to eight decimal places. The q = 0 case reduces correctly to a
diagonal covariance.

**ECM monotonicity.** On data simulated at K = 2, q = 3, the penalised log
likelihood rose from −25,367.6 to −15,353.5 over 120 sweeps, with a largest
single-step change of **+9.7 × 10⁻⁶** — no decrease at any iteration, against a
tolerance of 1.5 × 10⁻². A decrease would indicate an error in the M-step,
most plausibly in the ψ scaling of the spline penalty (raised in review).

**Parameter recovery.** Simulating at K = 3, q = 3 and refitting from eight
random initialisations:

| quantity | recovery |
|---|---|
| partition (adjusted Rand index vs true labels) | **1.000** |
| group centres, mean absolute error | 0.088 — **5.1%** of signal scale |
| within-group covariance C = WW′ + Ψ | 0.067 — 13.6% of scale |

Covariance is compared through C rather than W, because the likelihood
identifies W only up to rotation.

**Model selection recovers the generating K.** Data simulated at K = 2, fitted
at K = 1, 2, 3 and scored by held-out log predictive density on a 20% hold-out:

| K | held-out log predictive density |
|---|---|
| 1 | −5,908.9 |
| **2** | **−3,401.5** |
| 3 | −3,495.3 |

The procedure selects K = 2, penalising both the under- and over-specified
alternatives.

**Per-infant scores** (added 2026-10-16). Model comparison now uses each
infant's held-out log density (`mfa.pointwise_log_density`), not only fold
totals. The E-step was refactored to expose these, and the gate was rerun
before any new fits (job 4199427): all checks above still pass, and the
per-infant scores sum to the total held-out density to within 10⁻⁶.

### Run provenance

Every fit writes a JSON record (`results/provenance/`) holding the
configuration, a SHA-256 digest of it, SHA-256 digests of the input feature and
cohort files, the git commit, library versions, and whether float64 was active.
Thresholds are recorded as value-plus-method pairs rather than bare numbers —
`{"value": 1e-4, "method": "below smallest observed nonzero",
"smallest_nonzero": 4.59e-4}` — so the rule behind each constant remains
recoverable at write-up. This addresses a review comment, which asks for
rules rather than values. The pattern follows mr09's study-provenance records.

---

## S12 Figure palettes were validated, not chosen by eye

*Produced by: `figstyle.py`, checked with the data-viz palette validator.*

Figures follow the style of the proposal figure: warm sequential ramp for
postmenstrual age, light panels, bold panel letters, horizontal colourbar
beneath. Every palette was checked numerically rather than selected by eye.

**Age ramp.** The proposal's own yellow-to-red ramp fails two computable checks
against a near-white surface: its lightest step reaches only 1.26:1 contrast
(2:1 floor) and it spans 70 degrees of hue, so it is not a single-hue sequential
scale. The adopted six-step amber-to-deep-red ramp passes all four ordinal
checks — monotone in lightness, adjacent lightness gaps at or above 0.06,
light end above the contrast floor, hue spread under 40 degrees — while keeping
the warmth of the original.

**Categorical series** (group count K). Slots 1–5 of the validated reference
palette, in fixed order. Light mode passes every gate on the adjacent pairlist:
worst CVD separation ΔE 9.1 (target ≥ 8), worst normal-vision separation ΔE 19.6
(floor ≥ 15). Three slots fall below 3:1 contrast on the light surface, so the
relief rule applies and **every series carries a visible direct label** — colour
never carries identity alone.

**Scatter palettes** use only the first three slots, which clear the stricter
all-pairs gate (CVD ΔE 9.2, normal-vision ΔE 24.0).

Two defects were found by rendering the figures and inspecting them, which the
validator cannot catch because they are geometry rather than colour: overlapping
direct labels where two trajectories converged, and a line drawn from K = 1
straight to K = 3 implying a K = 2 result that did not exist. A third was a
genuine encoding error — two scatter calls on the same panel normalised the same
variable independently, so identical values rendered in different hues. The
collapse panel now labels values directly instead.

## S13 Model scores are reported as a gain per infant over a null baseline

Held-out log predictive densities are large negative sums: the log of a density
over 270 coordinates, summed over 569 held-out infants. Their absolute value
has no direct interpretation. The model-selection figure therefore plots

  gain per infant = (spec held-out lpd − null held-out lpd) / 569

where the null model treats each standardised coordinate as an independent
normal with its fit-fold mean and SD: no age, no groups and no axes. It is
scored on the same folds and infants as every specification
(`collect_sweep.baseline_lpd`).

| quantity | value |
|---|---|
| null held-out lpd (total) | −218,986.1 |
| per infant | −384.9 |
| per coordinate | −1.425 (the expectation for an exact N(0,1) is −1.419) |

A gain difference of 1 between two models means the better one gives a typical
held-out infant's data e ≈ 2.7 times the density. Its gain is the mean log
density ratio, i.e. a geometric mean over infants. On the 250-sweep sweep,
gains ranged from +37.8 (K=1, q=0) to +84.3 (K=1, q=25, not yet converged).
Unconverged fits are drawn with open markers until the 1000-sweep rerun (D12)
replaces them.

An earlier version of this figure had a fourth encoding error. matplotlib reads
a single RGBA tuple passed as `c=` as four data values when a series has exactly
four points, so several K=4 points in panel B took colours from an unrelated
colormap. It is fixed by passing `color=`.

## S14 The within-population axes (K = 1): size, content, stability, correlates

Fitted on all 569 training infants (`scripts/fit_final.py`; 20 restarts, 1000
ECM sweeps; the validation infants were not used). Every fit converged, and all
20 restarts reached the same optimum (spread 0.0 log-likelihood units):

| model | penalised log lik | wall time (8 CPUs) |
|---|---|---|
| K=1, q=5 | −161,953.9 | 2.8 min |
| K=1, q=15 | −150,325.8 | 17.7 min |
| K=1, q=25 | −140,945.6 | 28.6 min |

Axes are rotated to principal axes (largest first) and signed so that their
loadings sum positive (`scripts/make_axes_figures.py`).

**Size (q = 15).** The share of variation beyond age carried by each axis:
6.7, 4.8, 3.9, 2.8, 2.3, 2.1, 1.9, 1.7, 1.6, 1.4, 1.3, 1.3, 1.2, 1.0 and 0.9%.
The top 3 carry 15%, the top 6 carry 23% and all 15 carry 35%. The remaining
65% is region-specific (the unique variances ψ).

**Stability across q.** Each q = 15 axis was matched to its best-matching axis
in the other models (|cosine|):

| axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| vs q = 25 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.97 |
| vs q = 5 | 1.00 | 1.00 | 1.00 | 0.92 | 0.71 | | | |

Axes 1–3 are the same pattern whatever q is. Adding axes appends smaller
patterns without changing the leading ones.

**Content and correlates (top 6).** r is computed with infants' posterior axis
scores, n = 569. Bonferroni over 42 tests gives |r| ≥ 0.14.

| axis | share | dominant content | notable correlates |
|---|---|---|---|
| 1 | 6.7% | clustering throughout, strongest in insula, putamen and precuneus | network density −0.29, age since birth +0.18 |
| 2 | 4.8% | strength throughout the brain | age since birth −0.31, density −0.26, female −0.23, GA at birth +0.18, motion +0.17 |
| 3 | 3.9% | orbitofrontal strength and communicability vs posterior | motion −0.23, density −0.15 |
| 4 | 2.8% | orbitofrontal/occipital clustering (−) vs pallidum (+) | motion −0.26, age since birth −0.15 |
| 5 | 2.3% | posterior cingulate strength and communicability | age since birth −0.37, GA at birth +0.21, density +0.14 |
| 6 | 2.1% | right olfactory (all three measures) | none |

Notes for interpretation:
- PMA at scan correlates 0.00 with every axis, as it must: age is modelled by
  the trajectories. At fixed PMA, GA at birth and age since birth are the same
  information (PMA = GA + PNA), so axis 5's two correlations are one finding.
  Infants scanned longer after birth (i.e. born earlier) score lower.
- Axis 1 is clustering-dominated and correlates with post-threshold density.
  This is the density confound of S2. Its interpretation waits on the
  fixed-density rebuild.
- Axes 3 and 4 carry the strongest motion correlations. Per D8 they are
  reported with that caveat, not removed.
- Axis 6 is essentially one region and may be parcellation-specific.

## S15 K × q sweep, final (group-specific covariance, PMA clock)

All 35 specifications, 5-fold CV over 569 training infants. The 11 specs that
did not converge or crashed at 250 sweeps were rerun at 1000 (D12). Values are
the total held-out log predictive density (higher is better); the null baseline
is −218,986.1 (S13).

| q \ K | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 0 | −197,438 | −194,353 | −191,489 | −190,595 | −189,157 |
| 1 | −192,677 | −188,555 | −187,193 | −186,929 | −187,104 |
| 2 | −188,013 | −184,484 | −184,638 | −185,770 | −186,442 |
| 5 | −183,442 | −181,678 | −182,582 | −184,606 | −185,254 |
| 10 | −178,979 | −179,054 | −183,521 | −186,046 | −186,810 |
| 15 | −175,867 | −177,982 | −183,459 | −187,054 | −190,494 |
| 25 | **−170,915** | −176,069 | −183,455 | −190,789 | −198,451 |

- **Best: K = 1, q = 25.** Gain over the null +84.5 per infant, vs +75.4 for the
  best two-group model (K = 2, q = 25): a margin of 9.0 log units per held-out
  infant.
- The best K falls as q rises: q ≤ 1 → K = 4–5; q = 2–5 → K = 2; q ≥ 10 → K = 1.
  Groups substitute for within-group covariance, and once q is adequate they
  stop helping.
- Every K = 1, 2 and 3 spec converged. Ten specs drove a group's share to zero:
  K = 3 at q = 10, 15, 25; K = 4 at q = 10, 15, 25; and K = 5 at q = 5, 10, 15, 25.
  Five of them (K = 4, q = 25 and all four K = 5 specs) still did not meet
  the strict convergence criterion at 1000 sweeps. Where a 250-sweep score
  exists (K = 5 at q = 5, 10, 15), the lpd moved by < 20 units, so the flag
  reflects a slowly vanishing group, not an unreliable score.
- 250 → 1000 sweeps changed the reruns' lpd by −152 to +96 (K = 1, q = 25:
  −171,011 → −170,915). No ranking changed.
- **q is still rising at 25 for K = 1.** The q grid has not bracketed the
  optimum (see open items).
- Compute: 52.7 task wall-hours at 8 CPUs per task ≈ 420 core-hours for the
  full grid, including the reruns.

---

## S16 Paired per-infant model comparison and selection

*Produced by: `scripts/run_spec.py --n-iter 1000` (job 4199428) and
`scripts/compare_models.py` (job 4199431). 2026-10-16.*

Fold totals (S15) give five numbers per model; they cannot say whether a
difference exceeds infant-to-infant noise. Every training infant is held out
exactly once in the 5-fold CV, so two models can be compared on the same 569
infants: dᵢ = lpdᵢ(model 1) − lpdᵢ(model 2), with SE(Σdᵢ) = √n · sd(d)
(Vehtari, Gelman & Gabry, 2017).

All K = 1 and K = 2 specifications were refitted at 1000 ECM sweeps (equal
budgets), plus K = 1 at q = 35, 50, 70 because the score was still rising at 25
(S15). K ≥ 3 was not rerun: it lost at every q ≥ 5 and collapsed groups (S15).
Every fold of every specification converged.

| K | q | total held-out lpd | gain per infant | parameters | below best ± SE |
|---|---|---|---|---|---|
| 1 | 0 | −197,437.7 | 37.9 | 5,670 | 38,823 ± 1,729 |
| 1 | 1 | −192,676.7 | 46.2 | 5,940 | 34,062 ± 1,753 |
| 1 | 2 | −188,012.8 | 54.4 | 6,209 | 29,398 ± 1,736 |
| 1 | 5 | −183,441.8 | 62.5 | 7,010 | 24,827 ± 1,729 |
| 1 | 10 | −178,979.4 | 70.3 | 8,325 | 20,365 ± 1,661 |
| 1 | 15 | −175,866.7 | 75.8 | 9,615 | 17,252 ± 1,591 |
| 1 | 25 | −170,915.1 | 84.5 | 12,120 | 12,300 ± 1,470 |
| 1 | 35 | −167,242.5 | 90.9 | 14,525 | 8,628 ± 1,315 |
| 1 | 50 | −162,731.7 | 98.9 | 17,945 | 4,117 ± 923 |
| **1** | **70** | **−158,614.9** | **106.1** | 22,155 | — (best) |
| 2 | 0 | −194,352.6 | 43.3 | 11,341 | 35,738 ± 1,698 |
| 2 | 1 | −188,555.3 | 53.5 | 11,881 | 29,940 ± 1,819 |
| 2 | 2 | −184,484.4 | 60.6 | 12,419 | 25,870 ± 1,871 |
| 2 | 5 | −181,761.0 | 65.4 | 14,021 | 23,146 ± 1,773 |
| 2 | 10 | −179,054.6 | 70.2 | 16,651 | 20,440 ± 1,647 |
| 2 | 15 | −177,920.2 | 72.2 | 19,231 | 19,305 ± 1,486 |
| 2 | 25 | −176,068.5 | 75.4 | 24,241 | 17,454 ± 1,300 |

Parameters = K(MD + Dq − q(q−1)/2 + D) + (K − 1), with M = 20, D = 270.

**One group or two, at each q** (positive favours K = 1):

| q | K1 − K2 | SE | z |
|---|---|---|---|
| 0 | −3,085.1 | 320.1 | −9.6 |
| 1 | −4,121.4 | 477.6 | −8.6 |
| 2 | −3,528.4 | 423.3 | −8.3 |
| 5 | −1,680.8 | 267.9 | −6.3 |
| 10 | +75.2 | 232.7 | +0.3 |
| 15 | +2,053.4 | 271.9 | +7.6 |
| 25 | +5,153.4 | 354.7 | +14.5 |

Two groups win decisively when within-group covariance is too poor (q ≤ 5),
the two tie at q = 10, and one group wins decisively once q ≥ 15. This is
the substitution the methods plan warns of (§3): groups stand in for missing
covariance and stop helping once the covariance is adequate.

**Selection.** The rule was fixed before the results: among models whose
total is within one SE (of the paired difference) of the best, take the one
with the fewest parameters (Hastie, Tibshirani & Friedman, 2009). The best
model, **K = 1, q = 70**, is also the only one within 1 SE: each step in q is
several SEs (q = 50 → 70: +4,117 ± 923, z ≈ 4.5). **The rule therefore selects
the largest q tested, and the q grid is not bracketed.** Held-out density is
still rising at q = 70, though with diminishing increments (+7.2 per infant
from 50 to 70 vs +8.4 from 0 to 1). That suggests the residual covariance has
many small, real dimensions. It does not mean that 70 interpretable axes
exist (see S18–S19).

Figure: `fig7_model_comparison`.

---

## S17 Validation set, scored once

*Produced by: `scripts/validate_once.py` (job 4199435). 2026-10-16. The script
refuses to run again without `--force`; it has not been rerun.*

The 101 reserved infants (ages 29.9–44.4 weeks PMA, all inside the spline
basis) had not been used for fitting, standardisation, selection or figures.
The selected model (K = 1, q = 70) and the best model with a different K
(K = 2, q = 25) were fitted on all 569 training infants (20 restarts, 1000
sweeps; both converged) and scored once, standardised with the training
moments.

| | total lpd | gain per infant over null | |
|---|---|---|---|
| null baseline | −38,968.9 | 0 | |
| **K = 1, q = 70 (selected)** | **−28,416.7** | **104.5 ± 7.9** | CV estimate 106.1 |
| K = 2, q = 25 (runner-up) | −32,022.6 | 68.8 ± 5.3 | CV estimate 75.4 |

Selected minus runner-up: **+3,605.9 ± 445.5** (z ≈ 8.1); 96 of 101 infants
are better predicted by the selected model.

The selected model's gain on new infants (104.5) matches its cross-validated
estimate (106.1), so the selection did not overfit the CV. The one-group
model's advantage over two groups replicates in infants never used for any
decision.

Fitting note: the K = 2 fit had a restart spread of 392 log-likelihood units
across 20 starts, i.e. a multimodal likelihood. The K = 1 fit's 20 starts all
reached the same optimum (spread 0.0).

---

## S18 Permutation null (parallel analysis)

> **Extended to 1,000 permutations in S25.** The conclusion is unchanged, and
> S25 adds corrected p-values.

*Produced by: `scripts/permutation_null.py` (job 4199430) and
`scripts/summarise_axes_significance.py` (job 4199432). Model: K = 1, q = 15.*

Question: would axes this large appear if there were no shared structure? For
each of 100 permutations:
1. the fitted age trajectory was removed from the standardised training data;
2. each of the 270 coordinates' residuals was shuffled independently across
   infants, which destroys every between-coordinate correlation (the only
   thing the axes describe) but keeps each coordinate's distribution and age
   trend;
3. the trajectory was added back and the q = 15 model refitted (5 starts,
   1000 sweeps).

All 100 null fits converged. An axis "beats the null" if its variance exceeds
the 95th percentile of the null variance at the same rank (Horn, 1965).

| | axis 1 | axis 2 | axis 3 | … | axis 15 |
|---|---|---|---|---|---|
| observed share of variation beyond age | 6.75% | 4.75% | 3.95% | … | 0.94% |
| null, 95th percentile | 0.76% | 0.71% | 0.70% | … | 0.51% |

**All 15 axes beat the null.** The largest axis in any of the 100 permutations
carried 0.78%, below the smallest observed axis (0.94%). There is real
correlated structure along every axis. Whether each axis is a reproducible
*pattern* is a separate question (S19).

Figure: `fig8_axis_significance`, panel A.

---

## S19 Split-half replication

*Produced by: `scripts/split_half.py` (jobs 4199429, and 4199764 which also
saves loadings) and `scripts/summarise_axes_significance.py`. Model: K = 1,
q = 15.*

Ten random splits of the 569 training infants into halves (284/285),
stratified by age quartile × preterm status. Each half was fitted on its own
(5 starts, 1000 sweeps; all 20 fits converged), using the full training set's
standardisation so the halves share one scale. Each full-data axis was matched
to one axis in each half by Hungarian assignment on |cosine|, and the two
halves' matched axes were compared. Chance was simulated by applying the same
matching to random loading vectors.

**Axis by axis** (|cos| between halves; 1 = identical):

| axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| median over 10 splits | **0.91** | **0.82** | **0.74** | 0.48 | 0.44 | 0.38 | 0.34 | 0.34 |
| worst split | 0.78 | 0.69 | 0.59 | 0.25 | 0.27 | 0.20 | 0.10 | 0.17 |
| chance (95th percentile) | 0.17 | 0.18 | 0.18 | 0.17 | 0.18 | 0.16 | 0.17 | 0.17 |

**By block.** Axes of similar size (4–8 each carry 1.7–2.8%) can mix between
fits, which lowers the axis-by-axis match even when the space they span
replicates. Mean cos² of the principal angles between the two halves'
subspaces:

| block | median | worst | chance |
|---|---|---|---|
| axis 1 | 0.83 | 0.60 | 0.004 |
| axis 2 | 0.66 | 0.48 | 0.004 |
| axis 3 | 0.55 | 0.35 | 0.004 |
| axes 1–3 | **0.76** | 0.74 | 0.011 |
| axes 4–8 | **0.42** | 0.35 | 0.019 |

Interpretation:
- **Axes 1–3 are reproducible patterns** in independent sets of ~285 infants.
- **Axes 4–8 span a real but poorly determined subspace.** It overlaps
  between halves about 20× more than chance, but at half the sample its
  individual axes are not stably resolved.
- Half samples understate full-sample reliability; S20 gives the full-sample
  view, conditional on the model.

**Consequence for interpretation:** region-level claims are restricted to axes
1–3. Axes 4 and beyond (including the posterior-cingulate / time-since-birth
axis 5 of S14) are reported as a block of smaller, less stable structure.

Figure: `fig8_axis_significance`, panel B.

---

## S20 Parametric bootstrap of the selected model

> **Extended to 1,000 replicates in S25**, which gives the final intervals.
> The figures now use those.

*Produced by: `scripts/bootstrap.py` (job 4199434, 20 tasks × 10) and
`scripts/summarise_bootstrap.py` (job 4199759). Model: K = 1, q = 70 (the
selection, S16).*

Following the methods plan (§4), 200 datasets of 569 infants were simulated
from the fitted model at the observed ages exactly, so the age and prematurity
distribution matches by construction. Each was refitted from the fitted
parameters plus two random starts (1000 sweeps); all 200 converged. Each
replicate's loadings were rotated onto the original principal loadings by
orthogonal Procrustes before comparison, because the axes are identified only
up to rotation.

**Axis sizes** (% of variation beyond age, 95% percentile interval):

| axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| estimate | 6.82 | 4.83 | 4.10 | 2.88 | 2.45 | 2.19 | 2.02 | 1.87 |
| 2.5% | 6.21 | 4.57 | 3.76 | 2.82 | 2.45 | 2.24 | 2.05 | 1.89 |
| 97.5% | 7.83 | 5.53 | 4.66 | 3.38 | 2.89 | 2.55 | 2.39 | 2.16 |

From axis 5 on, the estimate sits at or below the lower percentile. This is
the known upward bias of bootstrapped sample eigenvalues: re-estimation noise
adds variance to every axis. The size intervals for axes ≥ 5 should not be
read as unbiased. A bias-corrected (BCa) interval would be the fix if they are
reported.

**Loadings.** The share of the 270 region × measure loadings whose 95%
interval excludes zero:

| axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| significant loadings | 195 | 185 | 206 | 210 | 180 | 175 | 179 | 173 |
| fraction | 0.72 | 0.69 | 0.76 | 0.78 | 0.67 | 0.65 | 0.66 | 0.64 |

These intervals are pointwise: if every loading were truly zero, about 14 of
270 per axis would exclude zero by chance.

Two caveats:
- **A parametric bootstrap assumes the model is true.** It measures how
  precisely loadings are estimated *if* the data come from this model. It does
  not test whether the patterns reproduce in real data. For axes 4 and beyond,
  the split-half result (S19) is the more conservative evidence, and the two
  disagree: the bootstrap says these loadings are precise, while split halves
  say the axes are not stably resolved at half the sample size. Region-level
  interpretation therefore follows S19.
- These intervals exclude model-selection uncertainty (methods plan §4).

The brain maps (S21) grey out loadings whose interval includes zero.

---

## S21 Brain surface renders

*Produced by: `scripts/make_brain_figures.py`. Figures `fig5_axis{1-5}_brain`,
`fig6_region_profiles`.*

Regions are drawn on the UNC neonatal T2w template using its own AAL90 labels
(1 mm, RAS), for display only. The connectomes used AAL90 in GA40 space. Both
atlases use the AAL numbering: their region-centroid layouts correlate at
r = 0.999 (x), 0.996 (y) and 0.994 (z) across the 90 regions. The connectome
node index (`data/atlases/AAL90_node_index.csv`, copied from ab15) maps node i
to label i.

How each hemisphere is rendered:
1. Its surface is extracted from the template brain mask (Gaussian σ = 1.3
   voxels; marching cubes).
2. It is smoothed (Taubin λ = 0.5, μ = −0.53, 25 iterations), subdivided once
   (~260k triangles per hemisphere) and smoothed again (10 iterations).
3. Each vertex takes the region whose Gaussian-smoothed mask (σ = 1 voxel) is
   strongest at that point. Vertices where the smoothed union of the
   hemisphere's regions falls below 0.12 (σ = 1.5) are left unlabelled.

Triangles are lit by orientation and painted back to front as polygons. No
image is resampled, so borders are smooth curves rather than voxel steps.

The deep-structure panel is an axial slice through the thalamic centroid at
4× resolution, with region borders from the same smoothed-mask rule.
Loadings whose 95% bootstrap interval includes zero (S20) are drawn light
grey; tissue outside the 90 regions is dark grey.

---

## S22 Compute ledger (CBU cluster)

Core-hours from SLURM accounting (CPUTimeRAW: wall time × allocated CPUs).

| run | jobs | core-hours |
|---|---|---|
| K × q sweep, 35 specs, 250 sweeps + 11 reruns at 1000 (S15) | 4132482, 4164190 | ≈ 420 |
| per-infant rerun, 17 specs at 1000 sweeps (S16) | 4199428 | 309.4 |
| final fits, selected + runner-up (S17) | 4199433 | 27.2 |
| permutation null, 100 fits (S18) | 4199430 | 14.2 |
| split-half, 2 × 10 splits (S19) | 4199429, 4199764 | 7.5 |
| parametric bootstrap, 200 fits at q = 70 (S20) | 4199434 | **843.9** |
| figures, summaries, validation, gate | various | < 2 |
| warm-start gate and bootstrap timing tests (S25) | 4282435, 4282307, 4282319 | ≈ 8 |
| permutations 101–1,000 (S25) | 4282437 | 151 |
| bootstrap replicates 201–1,000 at 4 CPUs (S25) | 4283506 | 1,567 |
| bootstrap at 8 CPUs, cancelled after 45 min for the 4-CPU rerun | 4282929 | ≈ 73 |
| tail anatomy, q = 70 split-half, measure evaluation (S23–S24) | 4281650–2, 4282577 | ≈ 85 |
| unthresholded refits and comparison (S26) | 4301050–1, 4303540 | ≈ 12 |
| measure sets: features, CV, fits, split-half (S27) | 4303541–5 | 82 |
| eigenvector set: K = 2 fit and inspection (S28b) | 4309285–6 | ≈ 2 |
| eigenvector set: K = 1/2 CV at q = 35, 50, 70 (S28a) | 4309287–8 | 261 |

The bootstrap was costed in advance for q = 15. The selection then chose
q = 70, which made each refit take about 28 minutes and the run about 4× the
estimate. Future runs should be costed after the selection is known.

---

## S23 What the axis tail encodes; split-half replication at q = 70

*Produced by: `scripts/characterise_tail.py` (jobs 4281652, 4282578) and
`scripts/split_half.py --q 70` (job 4281650). Model: K = 1, q = 70.
Figure: `fig9_tail_structure`.*

**Anatomy of each axis block.** Each block of axes implies a correlation
between every pair of the 270 coordinates (region × measure). The pairs are
grouped by anatomical relation:
- **within region:** a region's own three measures;
- **mirror:** a region and its homologue in the other hemisphere;
- **neighbour:** regions that touch in the connectome atlas, within 1 mm;
- **distant:** everything else.

For each block, the share of its squared implied correlation in each group is
compared with each group's share of pairs, which is what an evenly spread
block would give.

| | within region | mirror | neighbour | distant | local total |
|---|---|---|---|---|---|
| share of pairs (chance) | 0.7% | 1.1% | 9.9% | 88.2% | 11.8% |
| age-detrended data | 10.2% | 6.0% | 19.9% | 63.9% | 36.1% |
| axes 1–3 (15.7% of variation) | 0.4% (0.6×) | 2.0% (1.8×) | 12.3% (1.2×) | 85.2% | 14.8% |
| axes 4–8 (11.4%) | 2.5% (3.4×) | 4.1% (3.7×) | 12.4% (1.2×) | 81.0% | 19.0% |
| axes 9–20 (15.6%) | 5.6% (7.5×) | 5.3% (4.7×) | 15.3% (1.5×) | 73.8% | 26.2% |
| axes 21–70 (25.8%) | **24.4% (32.9×)** | 2.5% (2.3×) | 13.8% (1.4×) | 59.3% | **40.7%** |

The leading axes are brain-wide: their structure is spread across the brain
in close to chance proportions, and is if anything depleted within regions,
because each is dominated by a single measure. Moving down the tail, axes
become progressively local. In axes 21–70, a quarter of the implied
correlation lies between the three measures of a single region, 33× chance.
The data themselves carry this local coupling (36% local against 12% by
chance), so the tail models real structure. That structure is anatomically
expected (one region's measures co-vary; homologous and adjacent regions
co-vary) rather than a set of distinct patterns.

**Split-half replication at q = 70.** As in S19, but for the selected model:
10 stratified halves, each fitted at q = 70 (3 starts, 1000 sweeps; all 20
fits converged).

| axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| median \|cos\| | 0.91 | 0.80 | 0.73 | 0.48 | 0.40 | 0.40 | 0.31 | 0.28 | 0.28 | 0.21 |
| worst split | 0.78 | 0.70 | 0.57 | 0.25 | 0.21 | 0.19 | 0.06 | 0.07 | 0.16 | 0.02 |

| block | subspace overlap (median) | worst | chance |
|---|---|---|---|
| axes 1–3 | **0.75** | 0.71 | 0.011 |
| axes 4–8 | 0.38 | 0.33 | 0.019 |
| axes 9–20 | 0.27 | 0.24 | 0.044 |
| axes 21–70 | 0.44 | 0.43 | 0.185 |

Axes 1–3 replicate at q = 70 exactly as at q = 15 (S19). Every tail block
replicates well above chance as a subspace, but not as individual axes.

**Conclusion.** Interpretation stops at three axes because the axes beyond
them are local and not individually resolvable, not because they are noise.
The predictive gain from q > 3 is real, diffuse, local coupling.

---

## S24 Nodal measures: robustness to thresholding, redundancy and shape

*Produced by: `scripts/evaluate_measures.py` (job 4282577). Figure:
`fig10_measure_evaluation`.*

Question: is communicability too dependent on preprocessing to keep, and
would another nodal measure carry more distinct information?

All measures were computed for the 569 training infants on the primary
(per-bin 25% + MST) and the unthresholded networks. Three candidates were
added to the current three:
- weighted **betweenness** (lengths 1/w, normalised by (n−1)(n−2));
- **nodal efficiency** (mean inverse shortest-path length);
- **eigenvector centrality**.

Every measure's age trend was removed per region (penalised spline, as in the
model). All correlations are Spearman, taken per region across infants, with
the median over the 90 regions reported.

| measure | robust to thresholding | \|r\| with strength | \|r\| with clustering | median \|skew\| | zeros | age R² | communality (q = 70) | share on axes 1–3 |
|---|---|---|---|---|---|---|---|---|
| strength | 0.92 | — | 0.12 | 0.3 | 0% | 0.53 | 0.95 | 0.25 |
| clustering (log) | **0.22** | 0.12 | — | 0.3 | 0% | 0.09 | 0.48 | 0.15 |
| communicability | 0.84 | **0.77** | 0.15 | 0.4 | 0% | 0.13 | 0.88 | **0.05** |
| betweenness | 0.79 | 0.35 | 0.07 | 1.1 | **29%** | 0.08 | — | — |
| nodal efficiency | 0.97 | 0.72 | 0.09 | 0.5 | 0% | 0.77 | — | — |
| eigenvector centrality | 0.89 | 0.37 | 0.15 | 0.8 | 0% | 0.07 | — | — |

*Robust to thresholding:* per-region rank agreement between the thresholded
and unthresholded networks. *Communality:* the share of a coordinate's
variance carried by all 70 shared axes, rather than its region-specific term.

Findings:
1. **Communicability is not fragile to thresholding** (0.84). The fragile
   measure is **clustering** (0.22): infants are ranked almost differently by
   it in thresholded and unthresholded networks, and half its variance (0.52)
   is region-specific. Axis 1 is clustering-dominated and correlates with
   post-threshold density (S14). Together these make axis 1 the axis most
   at risk of being a thresholding artefact.
2. **Communicability is largely regional strength in another form**
   (|r| = 0.77 within region, after age). Its variance is well captured by
   the model (communality 0.88), but almost none of it by axes 1–3 (5%). It
   enters through the local tail, coupled to its own region's strength (the
   within-region coupling of S23), not as a brain-wide pattern of its own.
3. **Betweenness** is fairly robust (0.79) and less redundant (0.35), but it
   is zero-inflated (29% of values exactly 0) and the most skewed. That is a
   poor fit for a Gaussian factor model without a mixed discrete-continuous
   treatment.
4. **Nodal efficiency** is the most robust (0.97), but nearly as redundant
   with strength as communicability (0.72), and 77% of it is age.
5. **Eigenvector centrality** is robust (0.89), far less redundant with
   strength than communicability (0.37), has no zeros and only moderate skew.
   On these criteria it is the strongest replacement candidate.

---

## S25 Bootstrap and permutation null extended to 1,000 each

*2026-10-22/23. Jobs 4282437 (permutations), 4283506 (bootstrap) and 4283507
(summary). Supersedes the replicate counts in S18 and S20; their methods are
unchanged.*

**Why.** With 200 bootstrap replicates, a 95% interval rests on the 5 most
extreme replicates at each end. For 31–50 of the 270 loadings per axis (axes
1–8), whether the interval excluded zero was within Monte Carlo error of the
cut-off: another seed could flip the call, and with it the colour on the brain
maps. With 100 permutations, the smallest p-value expressible is about 0.01,
too coarse for a correction across 15 axes.

### Permutations: 1,000 (900 new, same settings as S18)

All 1,000 null fits converged. The largest axis in any of the 1,000 permuted
datasets carried 0.79% of variation; the smallest observed axis carries 0.94%.

| | every axis, 1–15 |
|---|---|
| p, against the null axis of the same rank | **< 0.001** (0 of 1,000 null axes as large; reported as 1/1,001) |
| p, family-wise corrected (max-statistic over all 15 axes) | **< 0.001** |

Every axis is larger than anything 1,000 shuffled datasets produced, after
correcting for testing 15 axes.

### Bootstrap: 1,000 replicates (800 new, same settings as S20)

All 1,000 refits converged. Every replicate used the same procedure as the
original 200 (warm start plus two random starts, 1000 sweeps), so all are
pooled.

**Axis sizes** (% of variation beyond age, 95% percentile interval):

| axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| estimate | 6.82 | 4.83 | 4.10 | 2.88 | 2.45 | 2.19 | 2.02 | 1.87 |
| 1,000 replicates | 6.26–7.86 | 4.51–5.54 | 3.77–4.67 | 2.79–3.43 | 2.46–2.91 | 2.23–2.59 | 2.05–2.37 | 1.91–2.17 |
| 200 replicates (S20) | 6.21–7.83 | 4.57–5.53 | 3.76–4.66 | 2.82–3.38 | 2.45–2.89 | 2.24–2.55 | 2.05–2.39 | 1.89–2.16 |

The upward bias noted in S20 for axes ≥ 5 (estimate at or below the lower
limit) persists, as expected: it is a property of bootstrapped eigenvalues,
not of the number of replicates.

**Loadings significantly non-zero** (95% interval excludes zero), out of 270:

| axis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 1,000 replicates | 195 | 182 | 204 | 205 | 180 | 174 | 178 | 175 |
| calls that changed from 200 replicates | 2 | 3 | 2 | 9 | 6 | 5 | 11 | 4 |
| calls still within Monte Carlo error | 12 | 14 | 12 | 17 | 11 | 14 | 21 | 16 |

Going from 200 to 1,000 replicates changed 2–3 significance calls on each of
the interpreted axes (1–3), out of 270 loadings. The number of calls within
Monte Carlo error of the cut-off fell from 31–50 to 11–21 per axis. The
conclusions of S20 stand. The intervals are now precise enough that further
replicates would mainly settle loadings sitting right at the cut-off.

### A faster refit was tested and rejected

Before the run, refitting from the warm start alone (no random starts) was
timed at ~5× faster. A gate job refitted 5 of the original replicates that way
on identical simulated data (`scripts/check_warm_start.py`, job 4282435). The
loadings differed by up to 0.033 and axis variances by up to 4%: at q = 70 a
random start sometimes reaches a better optimum than the warm start. The
shortcut was not used, and all 1,000 replicates share one procedure.

### Cost and duration

| run | core-hours | wall time | setup |
|---|---|---|---|
| 900 permutations | 151 | ~5 h | 4 CPUs × 8 tasks |
| 800 bootstrap replicates | 1,567 | ~17 h (12:28 → 05:26) | 4 CPUs × 24, then 32 tasks |

Bootstrap replicates averaged 1,789 s each at 4 CPUs (range 991–2,368 s),
against 1,265 s in a solo timing run. With eight tasks per node they compete
for memory bandwidth. Planning figure for future runs: about 2.0 core-hours
per replicate at q = 70, so ~10,000 core-hours (about 3.5 days on the 128-CPU
allowance) for 5,000.

---

## S26 Axes refitted on unthresholded networks

*2026-10-23. Produced by: `scripts/fit_final.py --variant unthresholded`
(job 4301050) and `scripts/compare_variant_axes.py` (jobs 4301051, 4303540).
Figure: `fig11_preprocessing_check`.*

**Why.** Clustering, which dominates axis 1, ranks infants very differently in
thresholded and unthresholded networks region by region (0.22, S24), and axis
1 correlates with post-threshold network density (S14). So axis 1 might have
been produced by the thresholding itself.

**What.** The one-group model was refitted at q = 15 and q = 70 to the same
three measures computed on the **unthresholded** networks: same infants,
training-moment standardisation, 20 starts and 1000 sweeps. Both fits
converged, and all 20 starts reached one optimum. A coordinate means the same
region × measure in both versions, so axes were compared directly
(Hungarian matching on |cosine|). Infants' scores on matched axes were also
correlated.

| | axis 1 | axis 2 | axis 3 | axes 4–8 |
|---|---|---|---|---|
| pattern similarity, \|cos\| (q = 70; q = 15 within 0.02) | **0.89** | **0.91** | **0.71** | 0.30–0.59 |
| same infants? score correlation | **0.88** | **0.97** | **0.82** | — |
| subspace overlap, axes 1–3 (chance 0.011) | 0.75 | | | 0.34 (4–8) |
| share of variation beyond age: primary → unthresholded | 6.8% → **20.6%** | 4.8% → 6.7% | 4.1% → 5.4% | similar |
| composition (primary → unthresholded) | 86% → 97% clustering | 82% → 82% strength | 40% → 48% communicability | |
| r with thresholded-network density | −0.28 → −0.20 | −0.27 → −0.31 | −0.13 → 0.01 | |
| r with head motion | 0.00 → −0.09 | +0.16 → +0.13 | −0.22 → −0.14 | |

Interpretation:
- **Axes 1–3 are not artefacts of thresholding.** The same three patterns
  re-emerge from networks that were never thresholded. They agree about as
  well as two independent halves of the sample do (S19: 0.91, 0.80, 0.73),
  and they order the same infants (r = 0.82–0.97).
- **Axis 1 is, if anything, stronger without thresholding:** 20.6% of
  variation, and even more purely clustering. This reconciles with S24: per
  region, clustering is threshold-sensitive (0.22), but its brain-wide shared
  component, which is what axis 1 captures, is robust because it averages over
  90 regions.
- **Axis 1's density link persists (−0.20) where no threshold acts.** The
  post-threshold density of an infant's network therefore reflects an
  underlying property of its edge weights, one that also drives brain-wide
  clustering, rather than being created by the threshold. Whether that
  property is biological or acquisition-related (e.g. tractography or
  streamline-count characteristics) is not settled by this check.
- **Axis 3's motion link weakens** (−0.22 → −0.14), and its density link
  disappears (−0.13 → 0.01).
- As in S19 and S23, axes 4 and beyond do not reproduce individually.

---

## S27 Measure sets: communicability dropped, or replaced by eigenvector centrality

*2026-10-23. Jobs 4303541–5, with the comparison re-run after a figure fix. Scripts:
`extract_alt_features.py`, `run_spec.py --variant`, `fit_final.py --variant`,
`split_half.py --variant` and `compare_measure_sets.py`. Figure:
`fig12_measure_sets`. Cost: 82 core-hours, 4.6 h wall.*

**Why.** Communicability largely restates regional strength (|r| = 0.77, S24).
Eigenvector centrality was the most promising replacement: robust to
thresholding (0.89), far less redundant with strength (0.37), no zeros.

**Sets**, each fitted to the same 569 training infants:
- **primary:** strength + clustering + communicability;
- **eig:** strength + clustering + eigenvector centrality. It is kept
  untransformed: by the D10 rule the log raised the median |skew| from 0.76
  to 1.06;
- **sc:** strength + clustering only (D = 180).

Strength and clustering are byte-identical across sets. Held-out scores are not
comparable across sets, since the predicted data differ, so each check
compares conclusions within a set.

**One group or two?** The paired per-infant K = 1 minus K = 2 difference (as
in S16), 20 starts per fold, 1000 sweeps:

| set | q = 15 | q = 25 |
|---|---|---|
| primary (S16) | +2,053 ± 272, z = **7.6** | +5,153 ± 355, z = **14.5** |
| sc | +1,528 ± 160, z = **9.5** | +3,523 ± 190, z = **18.5** |
| eig | −2,599 ± 644, z = **−4.0** (favours K = 2) | +330 ± 637, z = 0.5 (tie) |

**Same axes?** One-group model, q = 15. Each set's axes were matched to
primary axes 1–8 on the strength and clustering coordinates every set shares:

| | primary axis 1 (clustering) | axis 2 (strength) | axis 3 |
|---|---|---|---|
| sc: matching axis (\|cos\|) | axis 1 (0.995) | axis 2 (0.995) | axis 3 (0.935) |
| eig: matching axis (\|cos\|) | axis 2 (0.941) | axis 4 (0.984) | axis 6 (0.832) |
| split-half of the matched axis: sc / eig | 0.96 / 0.93 | 0.91 / 0.84 | 0.66 / 0.69 |

**What eigenvector centrality adds** (eig model, own axes):

| eig axis | share of variation beyond age | strength / clustering / eigenvector | split-half |
|---|---|---|---|
| 1 | **18.1%** | 0.02 / 0.06 / **0.92** | 0.98 |
| 2 (= primary axis 1) | 8.0% | 0.04 / 0.34 / 0.62 | 0.93 |
| 3 | 6.0% | 0.12 / 0.29 / 0.59 | 0.88 |
| 4 (= primary axis 2) | 4.3% | 0.83 / 0.07 / 0.09 | 0.84 |

Eigenvector centrality is skewed in many regions and looks bimodal in some:
the median per-region bimodality coefficient is 0.46, and 18% of regions
exceed 0.555, the usual bimodality flag.

Interpretation:
- **Communicability is not driving any conclusion.** Without it, the same
  axes 1–3 appear (|cos| 0.99, 0.99, 0.94), axes 1–2 reproduce at least as
  well, and one group beats two more decisively (z = 9.5 and 18.5). This is
  consistent with S24: communicability adds little beyond strength.
- **Eigenvector centrality carries a large brain-wide dimension of its own**
  (18% of variation, highly reproducible), which strength and clustering do
  not. The strength and clustering axes survive inside the eig model, mixed
  with eigenvector centrality.
- **With eigenvector centrality, the evidence for one group weakens.** Two
  groups win at q = 15 and the two tie at q = 25. Two explanations remain
  open:
  1. the extra dimensions need more axes before continuous variation absorbs
     them, the same substitution seen in the primary set at q ≤ 5 (S16);
  2. eigenvector centrality can jump between hubs when two regions compete for
     the network's leading eigenvector. That produces near-discrete,
     partly bimodal values, which a mixture model can capture as "groups".

  This is unresolved and should not be reported as evidence for subtypes
  without the follow-ups in DECISIONS.md.

---

## S28 What separates the eigenvector-set "groups"

*2026-10-23. Jobs 4309285 (K = 2, q = 15 fit, eigenvector set: 20 starts,
converged, restart spread 272) and 4309286, re-run after a figure revision.
Script: `inspect_eig_groups.py`. Figure: `fig13_eig_groups`.*

**Question.** In the strength + clustering + eigenvector set, two groups beat
one at q = 15 (S27). Is that structure among infants, or a property of the
measure?

| | result |
|---|---|
| group sizes | 321 / 248; every infant assigned with probability > 0.9 |
| what differs between group centres (at 40 weeks PMA) | **94% eigenvector centrality**; strength 2%, clustering 4% |
| largest differences | eigenvector centrality of left-hemisphere regions (putamen, precentral, insula, SMA, frontal: −1.4 to −1.6 SD in group 2), and right middle occipital (+1.4 SD) |
| membership vs each infant's dominant hub (region with the highest centrality) | Cramér's V = **0.77**: left middle frontal → group 1 in 153 of 154 infants; right middle temporal → group 2 in 138 of 154 |
| membership vs share of eigenvector centrality in the left hemisphere | group means **0.56 vs 0.23**; \|r\| = **0.81**; across infants the share ranges from 0.01 to 0.91 |
| membership vs the one-group model's eigenvector axis | \|r\| = 0.85 |
| PMA at scan, GA at birth, age since birth, network density | no difference (p = 0.50–0.99, Mann–Whitney) |
| preterm, female | no difference (p = 0.79, 0.25, χ²) |
| head motion | 1.76 vs 1.96 (p = 0.015 uncorrected; not significant after correcting for 7 tests) |

**Interpretation.** The two "groups" are infants whose eigenvector centrality
concentrates in the left hemisphere versus the right. Eigenvector centrality
is the leading eigenvector of the connectivity matrix. When a network has two
similarly strong, weakly coupled modules, here the two hemispheres, the
leading eigenvector tends to localise on whichever is marginally stronger.
Small differences in edge weights can then shift centrality wholesale from one
hemisphere to the other. That is a property of the measure, not a
distinct kind of infant: the groups show no differences in age, prematurity,
sex or network density, and strength and clustering barely differ between
them.

**Consequence.**
- The K = 2 preference in the eigenvector set is not evidence for neonatal
  subtypes.
- Eigenvector centrality is unsuitable as a regional measure for this
  question in these networks, because its hemispheric localisation dominates
  its variance (its 18% axis, S27).
- The primary measure set stands. Communicability is retained as planned, and
  S27 shows the conclusions do not depend on it.

### Part (a): one group vs two at higher q, eigenvector set

*Job 4309287, 6 tasks: K = 1, 2 at q = 35, 50, 70, 20 starts per fold, 1000
sweeps. Then job 4309288. 261 core-hours; the slowest fit (K = 2, q = 70)
took 24 h.*

Paired per-infant K = 1 minus K = 2, within the eigenvector set:

| q | difference ± SE | z | favours |
|---|---|---|---|
| 15 | −2,599 ± 644 | −4.0 | two groups |
| 25 | +330 ± 637 | 0.5 | tie |
| 35 | +4,178 ± 725 | **5.8** | one group |
| 50 | +12,805 ± 976 | **13.1** | one group |
| 70 | +28,751 ± 1,459 | **19.7** | one group |

The preference for two groups at q = 15 reverses once the model has enough
axes. This is the substitution seen in the primary set (S16), where two
groups also win at q ≤ 5 and lose from q = 15. It is simply shifted to higher
q, because eigenvector centrality adds large extra dimensions (S27).

Both explanations raised in S27 hold together:
- **at low q,** the extra (largely hemispheric) variance of eigenvector
  centrality is absorbed by a two-group split along the left/right
  localisation (part b);
- **with enough axes,** that variance is modelled as continuous variation and
  one group wins decisively.

**In no measure set does the evidence support infant subtypes.**

---

## S29 Axis 2 and the per-infant weight scale

*2026-10-26. `scripts/weight_scale_check.py` (job 4323693). Selected model,
K = 1, q = 70; axes as in S31. Every quantity is adjusted for PMA with the
model's penalised spline, and birth weight also for GA at birth.*

Strength is measured in raw edge-weight units, so anything that scales an
infant's whole connectome moves every region's strength together. Correlations
with axis scores:

| per-infant quantity (age-adjusted) | n | axis 1 | axis 2 | axis 3 |
|---|---|---|---|---|
| log total edge weight | 569 | −0.07 | **+0.89** | −0.16 |
| log maximum edge weight | 569 | **−0.62** | +0.50 | −0.19 |
| head circumference at scan | 543 | +0.06 | +0.34 | −0.09 |
| birth weight (also GA-adjusted) | 568 | +0.01 | +0.26 | −0.15 |
| post-threshold network density | 569 | −0.39 | −0.31 | −0.20 |

Between the quantities: total weight correlates with head circumference at
+0.40, with birth weight at +0.28 and with maximum weight at +0.63.

**Axis 2 is the infant's overall connection weight** (r = 0.89). Part of it
tracks body and head size (head circumference +0.34), so part is plausibly
biological. The rest is a global scaling of the weights whose origin
(biological or acquisition-related, e.g. tractography normalisation) the data
here cannot settle. Two ways to report it:
- as a global-scale axis, stated as such;
- as relative strength (each region's strength divided by the infant's total),
  to ask whether a regional strength pattern remains once the global scale is
  removed.

Axis 1's −0.62 with maximum weight is AUDIT.md finding A, tested in S32.

The sentinel codes in birth weight and head circumference (AUDIT finding I)
were cleaned before this analysis; n reflects the remaining missing values.

---

## S30 Developmental trajectories (research question 2)

*2026-10-26. `scripts/make_trajectory_figures.py` (jobs 4323694, 4323700).
Figures: `fig15_trajectories`, `fig16_trajectory_maps`.*

The selected model's mean trajectory μ(t) = B(t)β, back-transformed to each
measure's original units, with pointwise 95% bands from the 1,000 bootstrap
replicates (S25). Bands are conditional on the selected model.

**Averaged over the 90 regions, change from 30 to 42 weeks PMA:**

| measure | at 30 weeks | at 42 weeks | change (95% band) | change in SD units |
|---|---|---|---|---|
| strength (edge-weight units) | 9.17 | 19.80 | **+10.63 (10.18 to 11.02)** | +2.19 |
| log clustering | −4.173 | −4.116 | +0.057 (0.020 to 0.095) | +0.06 |
| communicability | 0.01637 | 0.01646 | +0.00009 (0.00005 to 0.00013) | +0.04 |

**By lobe (figure 15, D–F):**
- Strength rises steeply in every cortical lobe, most in parietal, but much
  less in subcortical regions.
- Clustering and communicability rise in frontal, parietal and occipital
  cortex but **fall in subcortical and limbic regions**.

Averaged over the brain, these opposing trends nearly cancel, which is why
clustering and communicability barely change on average. The regional maps
(figure 16, at 30, 34, 38 and 42 weeks) show the same divergence.

Caveats:
- The occipital clustering curve oscillates below 30 weeks, where few infants
  were scanned; its band is correspondingly wide.
- The global rise in strength overlaps axis 2's weight-scale question (S29):
  part of the developmental increase may be a change in overall weight scale.

---

## S31 Axes as eigenvectors of C; bootstrap principal angles

*2026-10-26. `scripts/make_axes_figures.py` (`principal_axes`) and
`scripts/summarise_bootstrap.py` (job 4323695). Resolves AUDIT.md findings B
and C.*

**Reporting now follows the methods plan (section 5).** Axes are the
eigenvectors of the within-group covariance C = WW′ + Ψ, largest eigenvalue
first, each signed so its largest-magnitude coordinate is positive:
- loadings are eigenvector × √eigenvalue;
- infant scores are projections of the age-detrended data onto each
  eigenvector, scaled to unit variance;
- shares are eigenvalue / trace(C).

Previously the axes were the principal axes of W alone. For axes 1–5 the two
agree to |cos| ≥ 0.994. Shares rise slightly because C's eigenvalues include
the uniquenesses' projection (q = 70: 6.82 / 4.83 / 4.10% → 6.99 / 4.88 /
4.21%). Under the new sign rule, axis 4 of the q = 15 model flips, and its
motion correlation now reads +0.26, not −0.26. Figures 3–6 were regenerated.
Earlier sections keep the numbers they were written with.

The bootstrap was run on the principal axes of W. Its intervals are
sign-aligned to the reported axes and applied as offsets around the reported
loadings, which is justified by the |cos| ≥ 0.994 agreement.

**Principal angles** (methods plan, section 4): for each of the 1,000
replicates, the angle between each leading axis and its best-matching
replicate axis, and the largest principal angle between the axis 1–3
subspaces.

| | axis 1 | axis 2 | axis 3 | axis 4 | axis 5 | subspace 1–3 |
|---|---|---|---|---|---|---|
| median angle | 12.1° | 18.6° | 20.6° | 26.7° | 36.3° | 16.5° |
| 95th percentile | 18.4° | 37.2° | 37.4° | 42.9° | 51.8° | 22.9° |

Axes 1–3 are well determined individually and very well as a subspace. From
axis 4 on, angles of 30–50° mean the directions are only loosely fixed. This
is consistent with the split-half results (S19, S23).

