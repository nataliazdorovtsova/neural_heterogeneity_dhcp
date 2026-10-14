# Running on the CBU cluster

## What the cluster is

| | |
|---|---|
| partition | `Main` (the only one) — 37 nodes, **1184 CPUs** |
| account | `cbu` |
| per-node | 32 CPUs, ~250 GB; default 6500 MB per CPU |
| walltime | default 1 day, **maximum 7 days** |
| QOS | `normal` — **capped at 128 CPUs per user** |
| GPUs | none anywhere (`GresTypes = (null)`) |

## Two different ways to be a bad neighbour

These are not the same problem, and only one of them the cluster protects
against.

**Grabbing cores** — already handled. QOS `normal` caps you at 128 CPUs, about
11% of the cluster. You cannot monopolise compute even by accident.

**Flooding the queue** — *not* handled, and this is the one that causes trouble.
Nothing stops a user submitting thousands of jobs. At the time of writing, one
user held **1,254 pending jobs** out of 1,331 cluster-wide. Their *running*
footprint was capped as designed, but everyone else's jobs sat behind a wall of
pending work, and the queue became unreadable for everyone.

So the rule that matters is about **job count**, not core count.

## How we submit

Everything goes through `scripts/slurm/submit.sh`, which follows the lab's
existing convention (`-A cbu -p Main`, job arrays, logs named `%A_%a`) and adds
the guardrails:

| guard | default | why |
|---|---|---|
| array size | **refuses above 64 tasks** | forces bundling instead of flooding |
| concurrent tasks | `%4`, never above 8 | leaves the cluster usable for others |
| peak CPUs | refuses above the 128 QOS cap | asking for more just queues forever |
| footprint report | always printed | you see the cost before you commit |

```bash
# always look first
bash scripts/slurm/submit.sh --name sweep --tasks 32 \
     --cmd "python scripts/run_spec.py" --dry-run

# then submit
bash scripts/slurm/submit.sh --name sweep --tasks 32 \
     --cmd "python scripts/run_spec.py" --time 04:00:00
```

It prints, before anything is queued:

```
  array tasks      32  (at most 4 running at once)
  peak footprint   32 CPUs = 2.7% of the cluster, 32/128 of your QOS cap
  cluster now      1331 jobs queued cluster-wide, 0 of them yours, 862 CPUs idle
```

`--force` exists for genuinely large arrays, but bundling is nearly always the
better answer.

## Bundle, don't fan out

The model-selection sweep is ~32 specifications × 5 folds × 20 initialisations ≈
**3,200 EM fits**. Submitted naively that is 3,200 jobs — precisely the pattern
that caused the trouble above.

Each fit is seconds to minutes and JAX vectorises across initialisations, so the
work bundles cleanly:

| approach | jobs | peak CPUs | neighbourly? |
|---|---|---|---|
| one job per fit | 3,200 | 128 (capped) | no — queue flooded |
| **one task per specification** | **32** | **32** | yes |

One array task per specification, each running its 5 folds × 20 initialisations
internally. 32 tasks at `%4` concurrency and 8 CPUs each is **2.7% of the
cluster**, and the whole sweep finishes in a few hours of wall time.

The 200-replicate bootstrap bundles the same way: 10 tasks of 20 replicates, not
200 jobs.

## Housekeeping

- **Always set `--time`.** The 1-day default on a 10-minute job blocks the
  scheduler's backfill and delays everyone, including you.
- **Request memory you will use.** 16 GB is ample here; the stacks are ~50 MB.
- `activate.sh` sets `OMP_NUM_THREADS` from `SLURM_CPUS_PER_TASK`, so threads
  match the allocation automatically. Outside a job it defaults to 4 so
  interactive work does not take over a login node.
- Logs land in `logs/` (gitignored). Check them; don't let failures pile up.
- Watch with `squeue -u $USER`, cancel with `scancel <jobid>`.
- **Test one task interactively before submitting the array.** A bug multiplied
  by 32 is still a bug multiplied by 32.
