# Analysis decisions

A running record of choices made, why, and what is still open. Newest section
last. Numbers here were measured on the actual data, not assumed — the scripts
that produced them are named alongside.

---

## D1. Data source — local copies, read-only genetics
**2026-10-02, revised 2026-10-08**

All connectome and behavioural data is a **local copy** under `data/`, taken
from ab15's tree and verified byte-for-byte (`DATA_PROVENANCE.md`). Reason: she
reorganises frequently — the symlinks this project started with broke within 24
hours — and copying removes any chance of writing into her folder.

**Polygenic scores are the exception: read in place, never copied.** They sit
behind a restricted five-person access group. This project lives on
`/imaging/astle`, readable by the whole lab group. Copying would widen access
beyond who is authorised under the dHCP 2026 agreement. `paths.PRS` points at
the controlled location.

---

## D2. Inference engine — NumPyro / JAX, explicit marginalisation
**2026-10-01**

CPU-only (no GPUs on this cluster: 37 nodes, 32 CPUs each, no GRES, 7-day limit).
`JAX_ENABLE_X64=1` is mandatory — float32 is not adequate for Cholesky factors
and log-determinants of a 270-dim covariance.

The discrete cluster label is **marginalised explicitly with `logsumexp`**, not
enumerated. NumPyro's enumeration requires `funsor`, which is broken against
jax >= 0.4.3x. Explicit marginalisation is faster anyway and keeps per-infant
responsibilities available as `exp(lp - logsumexp(lp, axis=0))`.

---

## D3. The sample is 753 SESSIONS from 670 INFANTS
**2026-10-08**

Verified: 753 rows, 670 unique `dhcp_id`, **83 repeat scans**. The methods
supplement states "each of N infants contributes one usable connectome" — this
is not true of the stack as delivered.

**99% of repeat scans are preterm infants** (median gap 7.3 wk, max 16.7 wk) —
scanned once early and again near term-equivalent age. So the dependence is
confounded with the preterm contrast, which is the contrast most likely to drive
group structure.

| sample | n | corr(PMA, PNA) |
|---|---|---|
| all sessions | 753 | +0.164 |
| **earliest scan per infant** | **670** | **+0.039** |
| latest scan per infant | 670 | +0.233 |

**Decision:** earliest scan per infant as primary (near-orthogonal clocks, clean
independence, matches ab15's PLS-C selection); all 753 sessions as sensitivity.

**Non-negotiable either way:** cross-validation and the held-out validation split
must be grouped on `dhcp_id`, not on rows. Held-out predictive density is the
sole basis for selecting K, q, covariance sharing, trajectory shape and clock —
session-level splitting would inflate it and bias selection toward complexity,
and nothing else in the design would catch it.

---

## D4. Density is confounded with age and preterm status
**2026-10-08 — the most consequential finding so far**

The per-bin 25% + MST rule takes the top 25% of each subject's **observed**
edges. Since pre-threshold density falls steeply with PMA, that confound passes
almost perfectly through:

- `corr(retained edges, pre-threshold edges) = +0.999`
- post-threshold density: mean 0.218, sd 0.013, **range 0.178–0.251**
- `corr(density, PMA) = -0.669` (pre-threshold was -0.688 — barely reduced)
- preterm 0.2249 vs term 0.2144, t = 11.6, p = 1.3e-28

Effect on the three measures:

| measure | r(age) | r(density) | r(age \| density) |
|---|---|---|---|
| strength | +0.891 | -0.734 | +0.770 |
| **clustering** | **+0.133** | -0.418 | **-0.264** |
| log-communicability | +0.067 | -0.036 | +0.059 |

**Clustering's age trend reverses sign** once density is controlled. RQ1 would
otherwise report the opposite of what the data support for segregation.

**Decision:** request a fixed-density rebuild from ab15 — top 25% of *possible*
connections within each distance bin (1001 edges for everyone), same Betzel
bin logic, same MST union. Pending that, carry density as a covariate and treat
any uncorrected age effect in clustering as unsafe.

Rejected alternative: keep variable density and regress it out. At r = -0.669
with PMA, that removes genuine age signal too.

NB: the `density` column in `participants_with_index.csv` is the **pre**-threshold
figure (0.854). Post-threshold density must be computed from the stack.

---

## D5. Regional measures — definitions fixed here
**2026-10-08**

ab15's topology numbers were computed for a previous project and are not
authoritative (confirmed by her). We compute our own and pin the definitions:

- **Strength** — weighted row sum. (Her `localmean_degree` is also weighted
  strength, not binary degree; they coincide at 17.7.)
- **Clustering** — NetworkX weighted = **Onnela**, normalised by the
  **per-subject** maximum weight. Scale-free, unlike Barrat.
- **Communicability** — `expm(D^-1/2 W D^-1/2)` (Crofts & Higham), regional mean
  **excluding the diagonal**, then **log**-transformed.

The normalisation is not optional. `expm` on raw µ weights is dominated by
exp(λ_max): ab15's `localmean_sub_centrality` has mean 1.15e9, sd 4.1e9, max
6.3e10 — four orders of magnitude, unusable per-subject. Normalised, max is 1.24.

**Add Crofts & Higham (2009) to the reference list.**

---

## D6. Coordinates WILL be standardised
**2026-10-08 — overrides the methods supplement**

§1 says "I will not standardise the coordinates across measures", on the grounds
that factor analysis estimates a separate residual variance per coordinate. That
is correct **for the likelihood** and false for everything we report. Measured:

| measure | mean | sd |
|---|---|---|
| strength | 17.49 | 10.61 |
| clustering | 0.0189 | 0.0124 |
| log-communicability | -4.16 | 0.359 |

**Variance ratio strength : clustering = 733,000 : 1.**

Consequences if left unstandardised: "each measure averaged across regions" and
the group-ordering rule are numerically pure strength; eigenvectors of C_k are
not scale-invariant, so the leading axis is the strength block and the
segregation/integration contrast the study hopes to find cannot emerge; the
shared-λ smoothness variant is not scale-invariant; and PCA initialisation of
the loadings becomes a strength detector.

**Decision:** z-score each of the 270 coordinates using **training-fold moments
only**. Also makes the ψ-scaling of the spline penalty (raised in review) far less
damaging, since residual variances become comparable across coordinates.

---

## D7. Edge weights are not streamline counts
**2026-10-08**

Edge weight = Σ SIFT2 streamline weights × participant-specific µ = relative
fibre-bundle capacity. The proposal's Data section says "streamline-count
matrices" and **must be corrected**.

---

## D8. No additional motion exclusions — but motion is a reporting diagnostic
**2026-10-08 (ab15 confirmed the preprocessing)**

Motion was corrected at source (SHARD, Christiaens et al. 2021) and the dHCP QC
gate already removed 129 of 887 sessions on dMRI quality. ab15 confirms no
further exclusions are needed. **Agreed — we add none.**

Checked rather than assumed. Residual `qc_dmri_eddy_rel_motion` on our sessions
(745 of 753 have a QC row): median 1.56, p95 3.94, max 16.3.

| | r |
|---|---|
| motion vs PMA | −0.141 |
| motion vs GA at birth | −0.173 |
| preterm vs term (median) | 1.61 vs 1.52 |

Motion is **not** meaningfully confounded with age or preterm status — unlike
density (D4) — so it cannot masquerade as a developmental or group effect. That
is the reassurance that makes "no exclusions" safe.

But it is not inert:

| measure | r(motion) | r(motion \| age) |
|---|---|---|
| strength | −0.143 | +0.046 |
| clustering | +0.099 | +0.121 |
| **log-communicability** | **−0.266** | **−0.242** |

An association of ≈ −0.24 with log-communicability survives age adjustment. This
matters specifically for RQ2, which *is* the characterisation of between-infant
variation: a nuisance axis driven by scan quality could surface as one of the
recovered directions of variation.

**Decision:** no exclusions; instead, after fitting, correlate each infant's
position along each axis of Cₖ with residual motion. An axis that tracks motion
is an artefact, not biology. Same logic as the preterm-ARI diagnostic (Stage 5).

---

## D9. Ancestry PCs found — and the PRS are already EUR-restricted
**2026-10-08**

`paths.ANCESTRY_PCS` → `genetics_qc/plink/dhcp_imputed/dhcp.hg19.EUR.eigenvec`
(not in `prs/`, which is why the first search missed them). 50 PCs.

Two sets exist, and the distinction matters:

| file | n |
|---|---|
| `dhcp.hg19.eigenvec` (all genotyped) | 766 |
| `dhcp.hg19.EUR.eigenvec` | **569** |
| `dhcp_prs_all.csv` | **569** |

The PRS row count matches the EUR eigenvec exactly, so **the scores were
computed only for the European-ancestry subset**. Use the EUR PCs; they are the
ones that correspond to the PRS sample.

No further attrition: **all 415** of our PRS infants have PCs, and 341 have PRS
+ PCs + CBCL. Convention is the first 5–10 PCs as covariates (PC1–5 carry 21.9%
of genotypic variance, PC1–10 31.7%).

**Caveat to state in any write-up:** the genetics arm covers 415 of 670 infants
(62%), and the restriction is *by ancestry*. Whatever we find genetically does
not automatically generalise to the full cohort.

---

## D10. The log belongs on clustering, not communicability
**2026-10-08 — overrides the methods supplement, measured on the training set**

§1 says: "I will log-transform communicability because its raw distribution is
strongly right-skewed." On this data, after the degree normalisation of D5, that
is backwards. Median |skew| across the 90 regions, 569 training infants:

| coordinate | median \|skew\| |
|---|---|
| communicability, raw | **0.41** |
| communicability, logged | 0.67 — the log makes it *worse* |
| strength, raw | 0.26 |
| **clustering, raw** | **1.51** — 33 of 90 regions exceed \|skew\| 2 |
| clustering, logged | **0.24** |

The supplement's reasoning was sound for *unnormalised* communicability, which
is dominated by exp(λ_max) and wildly skewed. Adopting Crofts & Higham
normalisation (D5) removes the premise: the normalised measure is bounded and
already symmetric.

**Decision:** communicability enters raw; clustering enters as
`log(c + 1e-4)`. The offset sits below the smallest observed nonzero (4.6e-4),
so real values are undistorted, and it keeps the 62 exact zeros (0.12% of
values) finite. Resulting median |skew| is 0.25 / 0.25 / 0.43 across the three
measures, against 0.25 / 1.61 / 0.68 before.

Both transforms remain switchable in `features.regional_features()`, so the
supplement's original choice is one flag away.

**Flag for sign-off:** this changes two documented methods decisions. The
evidence is in `results/features/` and reproducible from
`scripts/extract_features.py`.

---

## D11. The three measures are not redundant — gate passed
**2026-10-08**

Degree-normalising communicability builds strength into it, so there was a real
risk that two of the three measures were one measure. Checked on the training
set, across-infant correlation per region:

| pair | median r | IQR |
|---|---|---|
| strength vs communicability | +0.595 | [+0.39, +0.74] |
| strength vs clustering | +0.066 | [-0.04, +0.18] |
| clustering vs communicability | -0.029 | [-0.19, +0.15] |

Strength and communicability share moderate variance but are far from
collinear; at the global-mean level r is only +0.29. Clustering is close to
orthogonal to both. **Three genuinely complementary measures — proceed as
specified.**

---

## D12. Non-converged specs are rerun at 1000 sweeps, not reported as-is

*2026-10-15.* In the K×q sweep (job 4132482, 250 ECM sweeps per restart), 9 of
35 specs had at least one fold whose best restart had not met the
convergence criterion (relative change < 1e-6 on the final sweep). They are
K1q15, K1q25, K2q25, K3q15, K3q25, K4q15, K5q5, K5q10 and K5q15, mostly at
large K or q. Two more specs, K4q25 and K5q25, crashed at initialisation: a
k-means start cluster had ≤ 25 members, so it had fewer than q residual
directions to seed the loadings.

- **Fix:** the missing loading columns are filled with small random columns
  (1% of the residual SD). Zero columns would never move under ECM. Specs
  whose start clusters were large enough draw no extra random numbers, so
  their initialisations are unchanged.
- **Rerun:** all 11 specs at 1000 sweeps (job 4164190, about 430 core-hours).
  The provenance files are timestamped, so collect_sweep takes the newest
  file per spec. The 250-sweep files stay on disk.
- **Why it matters:** the K1-vs-K2 comparison turns on lpd margins of tens to
  hundreds of units. An unconverged fold understates its spec's lpd, so the
  comparison has to wait for converged fits.

---

## D13. Figures follow the project style guide (Wong palette, serif)

*2026-10-16.* All figures now use Natalia's figure style guide in place of the
proposal-figure styling (S12). The guide names a `geodesic_style` package that
is not installed on the cluster, so `figstyle.py` implements it directly,
mirroring HMM_MEG/analysis/viz/style.py. That means the Wong palette, serif
type at 16/14/11 pt, L-shaped axes, the four standard sizes, grey dashed
reference lines, rounded-top bars with thin black edges, and PNG (150 DPI) +
PDF output.

Colours carry fixed meanings across figures:
- each K has its own Wong colour and marker;
- preterm is vermillion and term (the reference group) is neutral grey;
- ordered quantities (age, q) use Blues;
- signed quantities (loadings, correlations) use RdBu_r, red = higher.

Wong defines no continuous scales; Blues and RdBu_r follow the HMM_MEG choice.

Layout changes this required:
- the model-selection figure is two panels (double column), and the
  group-collapse panel became its own single-column figure (fig2b);
- the region-profile heatmap is split by hemisphere so 11 pt region names fit;
- panel titles are the guide's 16 pt bold titles, and notes are 11 pt.

---

## D14. One population (K = 1); q = 70 selected, q not bracketed; interpret axes 1–3

*2026-10-22. Evidence: SUPPLEMENT S16–S20.*

- **K = 1.** On paired per-infant comparison, one group beats two at every
  q ≥ 15 (z = 7.6 to 14.5) and ties at q = 10. Two groups win only when
  within-group covariance is too poor (q ≤ 5). This holds on the 101
  validation infants, scored once: +3,606 ± 446, with 96/101 infants
  favouring K = 1 (S17).
- **q = 70 by the pre-set one-SE rule**, but only because it is the largest q
  tried. The score still rises, and every step is several SEs. The rule cannot
  bracket q on this grid. q = 70 is the predictive model, and the
  validation score (104.5 per infant vs 106.1 in CV) shows it does not overfit.
- **Interpretation is restricted to axes 1–3.** All axes beat the permutation
  null (S18), but only axes 1–3 replicate in independent halves (S19).
  Axes 1–3 are identical across q = 5–70, so the choice of q does not change
  what is interpreted. Axes 4 onwards are reported as a block of smaller,
  less stable structure.

---

## Open — needs a decision or an answer
- **How far to take q** (D14). Options: (a) report q = 70 as "at least 70
  dimensions improve prediction" and stop; (b) extend to q = 100/150 to
  bracket the optimum, at roughly 400+ core-hours plus refits; or
  (c) pre-register a practical-significance rule (e.g. stop when a step adds
  < 1 log unit per infant). The leading axes do not depend on this choice.
- **BCa intervals for axis sizes** (S20): percentile intervals are biased
  upward for axes ≥ 5. Needed only if those sizes are reported.
- **Fixed-density rebuild** (D4) — asked, awaiting ab15.
- **Sign-off on D10** (log on clustering rather than communicability).
- **Knot span** — observed PMA is 26.71–45.14; the proposed 26–45 basis
  extrapolates for 2 infants. Widen to 26.5–45.5.
- **Basis size M** — M = 8 is small for a true P-spline (Eilers–Marx assume a
  generous basis with the penalty doing the work). Either raise to ~15–20 or
  stop calling it a P-spline.
- **Reference age** for the both-clocks model (raised in review) — "centre at 40 wk
  PMA" is underdetermined; need a profile, e.g. PMA 40 / GA 40 / PNA 0.
- **Group ordering statistic** — ordering by mean-across-coordinates is
  scale-dominated. Use mixing proportion π instead.
- **Stability design** (raised in review) — ARI needs partitions of the same
  infants; parametric bootstrap generates new ones. Assign the original infants
  to each bootstrap-fitted model by responsibility instead.
