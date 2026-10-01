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
- **`JAX_PLATFORMS=cpu`** — there are no GPUs on this cluster (37 nodes, 32 CPUs
  and 250 GB each, no GRES, 7-day limit). Set explicitly so a wrong expectation
  fails loudly instead of silently falling back.
- **thread count** — follows `SLURM_CPUS_PER_TASK` inside a job, defaults to 4
  outside one so interactive work does not take over a login node.

### Two environment gotchas on this system

1. `~/.condarc` sets `ssl_verify: /etc/ssl/certs/ca-bundle.crt`, a RHEL path
   that does not exist on these Ubuntu nodes, so **conda fails for any command
   that hits the network**. `scripts/setup_env.sh` works around it with
   `CONDA_SSL_VERIFY=/etc/ssl/certs/ca-certificates.crt` rather than editing the
   global config. Worth fixing properly at some point.
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
