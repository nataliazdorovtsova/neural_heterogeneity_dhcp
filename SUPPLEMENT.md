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
- Every K = 1 and K = 2 spec converged. Ten specs at K ≥ 3 and q ≥ 5 (plus
  K = 3, q = 10) drove a group's share to zero. Four of those, all at K ≥ 4
  with a vanishing group, still did not meet the strict convergence criterion
  at 1000 sweeps. Their lpd moved by < 20 units between 250 and 1000 sweeps,
  so the flag reflects a slowly vanishing group, not an unreliable score.
- 250 → 1000 sweeps changed the reruns' lpd by −152 to +96 (K = 1, q = 25:
  −171,011 → −170,915). No ranking changed.
- **q is still rising at 25 for K = 1.** The q grid has not bracketed the
  optimum (see open items).
- Compute: 52.7 task wall-hours at 8 CPUs per task ≈ 420 core-hours for the
  full grid, including the reruns.
