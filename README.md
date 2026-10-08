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
| `activate.sh` / `activate.csh` | enter the environment (bash / tcsh) |
| `requirements.txt` | pinned package set |
| `scripts/` | analysis code |
| `data/` | symlinks to the read-only sources (see below) — nothing original here |
| `results/` | fitted models, model-comparison tables, derived features |
| `figures/` | figures |
| `notes/` | transcriptions of the proposal, methods supplement and ab15's first year report |
| `DECISIONS.md` | running record of analysis choices and why — read this first |
| `ANALYSIS_PLAN.md` | staged plan, with gates between stages |
| `DATA_PROVENANCE.md` | where the data came from, with checksums |

`results/`, `figures/` and `data/` are gitignored; everything in them should be
reproducible from `scripts/`.

## Data

All data is a **local copy** inside `data/`, owned by you (copied 2026-10-02
from `/imaging/astle/ab15/Project 1`). Nothing reads from a colleague's folder,
so their reorganising cannot break this project and we cannot touch their files.
Sources, checksums and what was deliberately left behind: **`DATA_PROVENANCE.md`**.

| Path | What it is |
|---|---|
| `data/connectomes/perbin25_mst_n753/` | **primary stack.** `weighted_perbin_retain25_mst.npy`, float64 `(90, 90, 753)`. Per-bin top 25% by µ weight within each of 10 Euclidean distance bins, unioned with a max-weight MST backbone; original µ weights retained. Mean density 21.8%, 0 sessions with isolated nodes. Plus QC tables and figures. |
| `data/connectomes/unthresholded_n753/` | `mu_unthresholded_stack.npy`, same shape, no thresholding. Mean density 85.4%. A preprocessing variant. |
| `data/participants_index/` | `participants_with_index.csv` — `subject_index` aligns with axis 2 of the stacks. Carries `scan_age` (PMA, weeks), `birth_age` (GA at birth), `sex`, `density`. Postnatal age = `scan_age − birth_age`. A copy also sits beside each stack; prefer that one, since the index is specific to it. |
| `data/behaviour_outcomes/` | `demographics.csv` (887 rows; birth weight, head circumference, radiology score, full dMRI QC), `cbcl_clean.csv` (742 rows, 18-month CBCL t-scores), `ecbq_clean.csv` (741 rows, ECBQ temperament) |
| `data/descriptives/` | ab15's topology descriptives for this same stack — useful to cross-check our own feature extraction against |
| `data/atlases/` | the 8 AAL90 / UNC neonatal atlas and template NIfTIs |

Sample: **n = 753**, of which **276 preterm** (GA < 37) and **477 term**.
PMA 26.7–45.1 weeks; postnatal age 0–19.7 weeks.

Import paths from `paths.py` rather than hard-coding them.

### Polygenic scores — read in place, never copied

`paths.PRS` → `/imaging/projects/external/dHCP_2026/data/genetics_qc/prs/dhcp_prs_all.csv`

569 infants × 48 scores, no missing values; IDs match `dhcp_id` directly. **415
of our 670 infants have them; 341 have both PGS and CBCL.** Coverage is balanced
on preterm status (63% vs 61%).

These sit behind a restricted five-person access group, while
this project is readable by the whole lab group. **Do not copy them into
`data/`** — that would widen access beyond who is authorised. See `DECISIONS.md`
D1. Ancestry principal components are not in the file and still need locating
before any PRS association is run.

### Also outstanding

- **Density is confounded with age and preterm status** (`DECISIONS.md` D4) —
  the most consequential open item. A fixed-density rebuild has been requested.
- **Five dead symlinks remain in `data/`**, pointing at the pre-rename
  `dHCP project` path. No data, superseded, harmless.

## Model variants to sweep

- clusters K = 1…5
- developmental time parameterised by postmenstrual age, postnatal age, or both
- axes of variation fitted jointly across clusters vs separately per cluster
- group separation constant vs age-varying
- preprocessing: thresholded vs unthresholded, weighted vs binarised

Selection is on held-out predictive performance throughout.

## Environment

```bash
source activate.sh      # bash
source activate.csh     # tcsh
python scripts/verify_env.py
```

Python 3.12 conda prefix at `env/` (1.3 GB), inside the project rather than
`~/.conda/envs`: it keeps the env with the code and off the 50 GB home quota.
Rebuild from scratch with `rm -rf env && bash scripts/setup_env.sh`.

Inference is **NumPyro / JAX**, with `arviz` for model comparison, `nilearn` +
`nibabel` for brain maps, `networkx` + `bctpy` for graph measures.

Things the activation scripts set, which matter:

- **`JAX_ENABLE_X64=1`** — JAX defaults to float32, which is not good enough for
  Cholesky factors and log-determinants of the 270-dim covariance. Anything run
  without sourcing an activate script will silently be in single precision.
- **`JAX_PLATFORMS=cpu`** — there are no GPUs on this cluster. Confirmed:
  `cluster2024` has one partition (`Main`), 37 nodes of 32 CPUs / 250 GB, a
  7-day limit, `GresTypes = (null)`, no node advertising any GRES, and no
  `/dev/nvidia*`. Set explicitly so a wrong expectation fails loudly instead of
  silently falling back to CPU.
- **thread count** — follows `SLURM_CPUS_PER_TASK` inside a job, defaults to 4
  outside one so interactive work does not take over a login node.

### Two environment gotchas on this system

1. **conda's CA bundle.** The site-wide config at
   `/imaging/local/software/miniconda/.condarc` (owned by IT, dated 2019) sets
   `ssl_verify: /etc/ssl/certs/ca-bundle.crt` — a RHEL path. The nodes run
   Ubuntu, where the bundle is `ca-certificates.crt`, so conda fails on every
   command that touches the network. Fixed by a personal `~/.condarc` setting
   `ssl_verify: true`, which makes conda use the CA bundle it ships with and
   works on both OSes. User config takes precedence over the site file, so the
   shared file is left alone. Don't hard-code either OS path.
2. The system `python3` has no `pip` or `ensurepip`, so a plain venv is not an
   option — hence the conda bootstrap.

### Mixture models: do not use enumeration

NumPyro can enumerate discrete latents, but that path requires `funsor`, which
is incompatible with jax >= 0.4.3x. Marginalise the cluster label explicitly:

```python
comp = dist.LowRankMultivariateNormal(mu[:, None, :], lam, psi)
lp = comp.log_prob(y) + jnp.log(pi)[:, None]        # (K, N)
numpyro.factor("lik", logsumexp(lp, axis=0).sum())
```

This is faster than enumeration, and per-infant responsibilities come back from
the same `lp` as `exp(lp - logsumexp(lp, axis=0))`. `scripts/verify_env.py`
checks this end to end against a known two-cluster synthetic.
