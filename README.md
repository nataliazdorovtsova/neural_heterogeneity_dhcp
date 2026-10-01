# Development and Heterogeneity in the Neonatal Connectome

A study using the Developing Human Connectome Project (dHCP).

A single generative Bayesian model fitted to neonatal AAL90 structural
connectomes, which simultaneously accounts for cluster structure in the
population, the age trajectory of each cluster's average topology, and the
within-cluster variation around that average.

Three questions:

1. **Structure** — is the population one group, or several distinct subclusters?
2. **Development** — how does mean connectome organisation change with age
   across 26–45 weeks postmenstrual age?
3. **Variation** — at a given age, what does variation between infants look like?

Each infant is summarised as a 90 × 3 table: strength, clustering coefficient
and communicability per AAL90 region.

## Layout

| Path | Contents |
|---|---|
| `paths.py` | pointers to every external input; import this rather than hard-coding paths |
| `scripts/` | analysis code |
| `data/` | symlinks to the read-only sources (see below) — nothing original here |
| `results/` | fitted models, model-comparison tables, derived features |
| `figures/` | figures |
| `notes/` | proposal, methods supplement, working notes |

`results/`, `figures/` and `data/` are gitignored; everything in them should be
reproducible from `scripts/`.

## Data

All inputs live in **`/imaging/astle/ab15/dHCP project`** (colleague's tree,
treat as read-only) and are reached through symlinks in `data/`.

| Link | Target | What it is |
|---|---|---|
| `data/connectomes_perbin25_mst_n753` | `results/connectomes/perbin25_mst_n753_nbins10` | **primary stack.** `weighted_perbin_retain25_mst.npy`, float64 `(90, 90, 753)`. Per-bin top 25% by µ weight within each of 10 Euclidean distance bins, unioned with a max-weight MST backbone; original µ weights retained. Mean density 21.8%, 0 sessions with isolated nodes. |
| `data/connectomes_unthresholded_n753` | `results/connectomes/unthresholded_n753` | `mu_unthresholded_stack.npy`, same shape, no thresholding. Mean density 85.4%. Useful as a preprocessing variant. |
| `data/participants` | `data/participants` | `participants_with_index.csv` — `subject_index` aligns with axis 2 of the stacks. Also carries `scan_age` (PMA, weeks), `birth_age` (GA at birth, weeks), `sex`, `density`, `n_isolated_nodes`. Postnatal age = `scan_age − birth_age`. |
| `data/behavior` | `data/behavior` | `demographics.csv` (887 rows; birth weight, head circumference, radiology score, full dMRI QC: motion, SNR, CNR, outlier slices), `cbcl_clean.csv` (742 rows, 18-month CBCL t-scores), `ecbq_clean.csv` (741 rows, ECBQ surgency / negative affect / effortful control) |
| `data/atlases_templates` | `data/atlases_templates` | AAL90 neonatal atlas NIfTIs and registration outputs |

A copy of `participants_with_index.csv` also sits inside each connectome
directory; prefer the one next to the stack you are actually using, since the
index is specific to it.

QC for the thresholding variants is tabulated in
`data/connectomes_perbin25_mst_n753/qc_summary_with_nbins.csv`, and ab15's own
`PATH_MAP.md` / `METHODS_ANALYSIS.md` document how the stacks were built.

## Model variants to sweep

- clusters K = 1…5
- developmental time parameterised by postmenstrual age, postnatal age, or both
- axes of variation fitted jointly across clusters vs separately per cluster
- group separation constant vs age-varying
- preprocessing: thresholded vs unthresholded, weighted vs binarised

Selection is on held-out predictive performance throughout.
