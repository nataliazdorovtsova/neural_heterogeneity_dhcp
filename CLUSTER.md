# Running on the CBU cluster

## What the cluster is

| | |
|---|---|
| partition | `Main` (the only one) — 37 nodes, **1184 CPUs** |
| account | `cbu` |
| per-node | 32 CPUs, ~250 GB; default 6500 MB per CPU |
| walltime | default 1 day, **maximum 7 days** |
| GPUs | none anywhere (`GresTypes = (null)`) |
| accounting | `sacct` records no `TotalCPU` or `MaxRSS` here, so CPU efficiency and peak memory can't be read back; size jobs from timings |

Per-user QOS caps (`sacctmgr show qos format=Name,MaxTRESPU`, checked 2026-10-15).
They are QOS levels, not partitions: `-q lopri`, never `-p lopri`.

| QOS | max CPUs | max memory | notes |
|---|---|---|---|
| `normal` (default) | 128 | 840 G | the cap that actually bites |
| `lopri` | 384 | 2500 G | preemptible: a job can be killed to make room |
| `standard` | 128 | 840 G | a separate cap from `normal`'s, so it still runs when `normal` is full |

The caps count CPUs held at the same moment across all your jobs. A job pending
with reason `QOSMaxCpuPerUserLimit` is waiting on your own other jobs, not on
the cluster.

## Rules

Adapted from Maros Rovny's (mr09) guidelines, 2026-10-14. Each one exists
because breaking it cost him real time.

1. **No compute on the login node.** Fits, sweeps, validation, benchmarks and
   figure builds all go through SLURM. Reading files, `squeue`/`sacct`, git
   and short one-liners are fine. *(Before 2026-10-15 the benchmarks,
   `validate_model.py` and feature extraction ran on the login node. They were
   short, but they belonged in jobs.)*
2. **Always ask for the maximum wall, `--time 7-00:00:00`.** A user can lower a
   running job's limit but not raise it, so a wall that is too short kills fits
   near their last step and can't be fixed afterwards. `submit.sh` now
   defaults to it. This replaces the earlier advice to set a short `--time`.
3. **Submit independent jobs at once**, and chain dependent ones with
   `--after <jobid>` (`--dependency=afterok`) instead of waiting by hand.
   SLURM holds them behind the caps itself.
4. **Price runs in core-hours; don't shrink the design to fit.** Say what the
   full design costs and let the design decide the compute.
5. **Cancel by job id, never `scancel -u`.** One bad call would kill every job
   you have running.
6. **Check for Windows line endings before every submit.** `sbatch` refuses
   scripts with CRLF line endings, and editing over the `U:` drive from Windows
   is how they creep in. `submit.sh` runs `scripts/slurm/check_launcher.sh`
   automatically, and `.gitattributes` forces LF in the repo.
7. **Cap every thread pool to the allocation.** `activate.sh` sets
   `OMP`/`MKL`/`OPENBLAS`/`NUMEXPR_NUM_THREADS` and XLA's intra-op threads from
   `SLURM_CPUS_PER_TASK`. Uncapped BLAS threads oversubscribe a small
   allocation and can make a job several times slower.
8. **Outputs on `/imaging/astle`, never `/home`** (quota). Copy into the share
   with `rsync -rlt`, not `rsync -a`: `-a` carries over home's private
   permissions and locks the group out.
9. **On resuming a session, check `squeue -u nz01` / `sacct` first** and report
   anything that finished or failed.

Not adopted (specific to Maros's project): `POT_BACKEND_DISABLE_TENSORFLOW`, the
[project] claims gate, and the work-stealing launchers with lopri fan-out. The
fan-out only works when every copy of a launcher pulls from a shared pool. Our
arrays split work by index, so two copies would duplicate it. If a sweep ever
outgrows the 128-CPU cap, the next step is to submit a second, disjoint array
on `-q lopri`.

### `--exclusive`

`--exclusive` gives a job a whole node, and every task is billed for 32 CPUs
against the cap. That suits one large allocation running many worker steps
(Maros's own launchers: 32 workers × 4 CPUs on one node). It does not suit
our arrays of small independent tasks, so `submit.sh` doesn't use it.

## Two different ways to be a bad neighbour

**Grabbing cores** is already handled: the QOS caps limit you to about 11% of the
cluster on `normal`.

**Flooding the queue** is *not* handled, and it's the one that causes trouble.
Nothing stops a user submitting thousands of jobs. At one point a single user
held **1,254 pending jobs** out of 1,331 cluster-wide. Their *running* footprint
was capped as designed, but everyone else's jobs sat behind a wall of pending
work. So the rule that matters is about **job count**, not core count.

## How we submit

Everything goes through `scripts/slurm/submit.sh` (`-A cbu -p Main`, job arrays,
logs named `%A_%a`), which adds the guardrails:

| guard | default | why |
|---|---|---|
| array size | **refuses above 64 tasks** | forces bundling instead of flooding |
| concurrent tasks | `%4`, never above 16 | leaves the cluster usable for others |
| peak CPUs | refuses above the chosen QOS's cap | asking for more just queues forever |
| wall | `7-00:00:00` | can't be raised after submission |
| CRLF | refuses DOS line endings | `sbatch` would reject them |
| commas in `--cmd` | refused | `--export` splits on commas |
| footprint report | always printed | you see the cost before you commit |

```bash
# always look first
bash scripts/slurm/submit.sh --name sweep --tasks 35 --cpus 8 --mem 32G \
     --cmd 'python scripts/run_spec.py --index $SLURM_ARRAY_TASK_ID' --dry-run

# then submit; chain a follow-up that starts only if every task succeeds
bash scripts/slurm/submit.sh ... (same flags, no --dry-run)     # -> job 123
bash scripts/slurm/submit.sh --name collect --tasks 1 --cpus 2 --mem 8G \
     --after 123 --cmd 'python scripts/collect_sweep.py'
```

If a chained job's prerequisite fails, it sits harmlessly in
`DependencyNeverSatisfied`. Cancel it by id.

## Bundle, don't fan out

The model-selection sweep is 35 specifications × 5 folds × 20 initialisations =
**3,500 EM fits**. JAX vectorises across initialisations, so one array task per
specification runs its 100 fits internally:

| approach | jobs | neighbourly? |
|---|---|---|
| one job per fit | 3,500 | no: queue flooded |
| **one task per specification** | **35** | yes |

The 200-replicate bootstrap bundles the same way: 10 tasks of 20 replicates, not
200 jobs.

## Gotcha: `--wrap` runs under dash, not bash

`sbatch --wrap="..."` generates a `#!/bin/sh` script. On these nodes `/bin/sh`
is dash, which has no `source` builtin, so every task dies within a second with
`source: not found` (exit 127). Your login shell being tcsh is a red herring.

`submit.sh` therefore submits a real job script, `scripts/slurm/_task.sh`, with
a `#!/bin/bash` shebang, passing the project path and command through
`--export`. Don't go back to `--wrap`.

## Measured costs

| run | per-spec wall | resources |
|---|---|---|
| K=1, q=0, 250 sweeps | 3.6 min / 2.5 min | 4 threads / 8 CPUs |
| K=3, q=10, 250 sweeps | 8.6 min | 4 threads |
| K×q sweep (job 4132482), 250 sweeps | 2.5 min to 1.8 h | 8 CPUs, 32 G |

Doubling from 4 to 8 CPUs gave a 1.4× speed-up, so 4 CPUs is more economical
per core-hour, and 8 finishes sooner. Wall time grows with K and q.

The full 35-spec sweep ran 4 at a time (32 CPUs, 2.7% of the cluster). The
2026-10-15 rerun (job 4164190) covers the 9 specs that did not converge plus
2 init crashes: 11 tasks × 8 CPUs, 1000 sweeps each, ≈ 430 core-hours,
88 CPUs peak.

## Housekeeping

- **Request memory you will use.** 32 GB has been enough for every spec; the
  stacks are ~50 MB.
- Logs land in `logs/` (gitignored). Check them; don't let failures pile up.
- **Test with a small `sbatch` job before submitting the array**, not on the
  login node. A bug multiplied by 35 is still a bug multiplied by 35.
