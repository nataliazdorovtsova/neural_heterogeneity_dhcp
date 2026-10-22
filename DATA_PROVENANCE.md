# Data provenance

Everything in `data/` is a **copy**, made 2026-10-02, now owned by `nz01`.
Nothing in this project reads from or writes to a colleague's folder any more.

Source for all of it: **`/imaging/astle/ab15/Project 1`** (read-only; verified
unchanged after copying).

## What was copied

| Here | Copied from (under `Project 1`) | Size |
|---|---|---|
| `data/connectomes/perbin25_mst_n753/` | `results/connectomes qcs/perbin25_mst_n753_nbins10/` | 49 MB |
| `data/connectomes/unthresholded_n753/` | `results/connectomes qcs/unthresholded_n753/` | 49 MB |
| `data/behaviour_outcomes/` | `data/behavior/` | 1.5 MB |
| `data/participants_index/` | `data/participants/` | 396 KB |
| `data/atlases/` | `data/atlases_templates/*.nii.gz` (the 8 loose NIfTIs only) | 7 MB |
| `data/descriptives/` | `archive_not_in_use/connectome_threshold_tests/topology_perbin25_mst_n753_nbins10/` | 34 MB |

All 41 files verified byte-for-byte identical to source with `diff -r` / `cmp`.

### Checksums (sha256) of the files that matter

```
c4f95936d32988b2aa6dc4ab3c723252ed298ca51235fd22dc07bca46586dd10  connectomes/perbin25_mst_n753/weighted_perbin_retain25_mst.npy
2d2e9a231c1e02c4c94a5723dc3dc26a1f47893d9d6df6bce244b1359da998e7  connectomes/unthresholded_n753/mu_unthresholded_stack.npy
29cba2386bef805f55e5b9357a8639d57929baf252fa10ac2bc0a07456663847  connectomes/perbin25_mst_n753/participants_with_index.csv
6111bbd137111d7d5fe3642ad6995423c3b3f7efb410d595a556f55c32f78fef  behaviour_outcomes/cbcl_clean.csv
53a22a495cfcc0694031520c82bdfb115f64e7d7abc1d162c38115c073579e5b  behaviour_outcomes/demographics.csv
51fd2da03815c8eb40e451321b14d2e9017f7713096c5d243b731720d07aba62  behaviour_outcomes/ecbq_clean.csv
c33c57a0eb9d8dc148cdf952e9f02c66514b8c16dbdcc067a8e5669fe8d91d12  descriptives/descriptives_topology_summary.csv
```

Re-check any time with `sha256sum -c` against this list.

## What was deliberately NOT copied

- **`data/atlases_templates/` in full — 23 GB.** Only the 8 loose `.nii.gz`
  atlas/template files (7 MB) were taken. The bulk is raw registration output
  (`Neonate/` 14 GB, `registration_aal90/` 4.8 GB, `dhcp_atlas_extended/`
  4.2 GB) which this project does not need. Reach for it in place if ever
  required.
- **`results/pls/` (8.5 MB)** — ab15's PLSC analysis output, not an input here.
- The raw pipeline trees (`final connectomes (scaleup_batalle_10M)/`,
  `derivatives_sc/`, `troubleshooting connectomes/`) — per-subject intermediates,
  hundreds of GB.

## ab15's 2026-10-01 renames

The folders were renamed again the day after this project was set up. Her
`README.md` and `PATH_MAP.md` still describe the *old* names, so they are stale:

| Was | Is now |
|---|---|
| `dHCP project` | `Project 1` |
| `results/connectomes/` | `results/connectomes qcs/` |
| `scaleup_batalle_10M/` | `final connectomes (scaleup_batalle_10M)/` |
| — | `troubleshooting connectomes/` (new) |

This broke the symlinks this project originally used — which is exactly why the
data is copied now rather than linked. Five dead symlinks remain in `data/`
(`behavior`, `participants`, `connectomes_perbin25_mst_n753`,
`connectomes_unthresholded_n753`, `atlases_templates`). They point at the old
`dHCP project` path, contain no data, and are harmless. They are superseded by
the real directories above and can be removed whenever convenient.

## Still missing: polygenic scores and the "dhcp2026" folder

**Not found.** Searched `/imaging` and `/home` for anything matching `dhcp2026`,
`dhcp*2026*`, `polygenic`, `pgs`, `prs` — nothing dHCP-related exists.

The polygenic-score material that does exist on `/imaging/astle` belongs to
other projects and other cohorts, not dHCP:

- `users/ts04/ALSPAC_Genetics/` — ALSPAC, PRSice IQ/education scores
- `am11/ABCD/pgs/`, `am11/talia_abcd/abcd_prs_external/` — ABCD
- `users/da04/Postdoc/pgs_dissim/`, `users/ai05/genetics/` — other work

None were copied. **Ask ab15 for the actual path** — the folder may be named
something else, may live outside `/imaging/astle`, or may not be readable with
current group permissions.

## History rewrite before publication (2026-10-22)

Before the repository was published, the git history was rewritten once:
- a colleague's unpublished report transcription (`notes/bhaskar_FYR.md`) was
  removed from every commit; it is kept locally and is now git-ignored;
- 23 commits authored with the cluster address were re-attributed to the
  author's own email.

No other file changed: the old and new final trees differ only by that one
file.

Rewriting changes commit IDs, and every provenance record in
`results/provenance/` stores the commit it ran on. **`COMMIT_MAP.tsv` maps
each old commit ID to its new one**, so a record written before the rewrite can
still be traced to its code. The original history is preserved as a git bundle
outside the repository
(`neural_heterogeneity_dhcp_backups/pre_public_rewrite_2026-10-22.bundle`).

Participant-level files (`results/`, `data/`) have never been committed.
