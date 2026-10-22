#!/bin/bash
# Guarded SLURM submission for this project.
#
#   bash scripts/slurm/submit.sh --name sweep --tasks 32 --cmd "python scripts/run_spec.py"
#
# Follows the lab's existing convention (-A cbu, -p Main, job arrays, logs named
# %A_%a) and adds the guardrail it lacks: a hard cap on how much of the queue
# this project can occupy. See CLUSTER.md for why.
#
# Options:
#   --name NAME        job name (required)
#   --tasks N          array size (required)
#   --cmd "..."        command each task runs; reads $SLURM_ARRAY_TASK_ID
#   --cpus N           CPUs per task        (default 8)
#   --mem SIZE         memory per task      (default 16G)
#   --time D-HH:MM:SS  wall limit per task  (default 7-00:00:00, the partition max;
#                      a wall cannot be raised after submission, so never cut it short)
#   --concurrent N     max tasks running at once (default 4)
#   --qos Q            normal (default) | lopri (384-CPU cap, preemptible) | standard
#                      (its own 128-CPU cap). QOS levels, not partitions: -q, never -p
#   --after JOBID      start only after JOBID succeeds (--dependency=afterok)
#   --dry-run          print the sbatch command and exit
#   --force            allow more than MAX_TASKS (think first)
#
# Never use --exclusive for these arrays: it bills a whole 32-core node against
# the per-user cap for every task. It suits one big allocation running many
# workers (Maros's own launchers), not small independent tasks.
#
# Cancel with `scancel <jobid>`, never `scancel -u`.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAX_TASKS=64            # refuse bigger arrays without --force
MAX_CONCURRENT=32       # never run more than this many at once (32 x 4 CPUs = the 128-CPU cap)
CLUSTER_CPUS=1184

NAME=""; TASKS=""; CMD=""; CPUS=8; MEM=16G; TIME=7-00:00:00
CONCURRENT=4; QOS=normal; AFTER=""; DRY=0; FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --tasks) TASKS="$2"; shift 2;;
    --cmd) CMD="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --concurrent) CONCURRENT="$2"; shift 2;;
    --qos) QOS="$2"; shift 2;;
    --after) AFTER="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --force) FORCE=1; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[[ -n "$NAME" && -n "$TASKS" && -n "$CMD" ]] || { echo "need --name, --tasks and --cmd" >&2; exit 2; }

# --- guardrails --------------------------------------------------------------
# Per-user caps (sacctmgr show qos format=Name,MaxTRESPU; checked 2026-10-15).
# The three QOS levels are capped separately.
case "$QOS" in
  normal|standard) QOS_CPU_CAP=128;;
  lopri)           QOS_CPU_CAP=384;;
  *) echo "unknown --qos $QOS (normal, lopri or standard)" >&2; exit 2;;
esac

# sbatch --export splits on commas, so a comma in the command would silently
# truncate it.
if [[ "$CMD" == *,* ]]; then
  echo "REFUSING: --cmd contains a comma, which --export would split. Use spaces." >&2; exit 1
fi

bash "${PROJECT}/scripts/slurm/check_launcher.sh" "${PROJECT}/scripts/slurm/_task.sh" >/dev/null \
  || { bash "${PROJECT}/scripts/slurm/check_launcher.sh" "${PROJECT}/scripts/slurm/_task.sh"; exit 1; }

if (( TASKS > MAX_TASKS && FORCE == 0 )); then
  cat >&2 <<MSG
REFUSING: $TASKS array tasks exceeds the project cap of $MAX_TASKS.

Large arrays are almost always avoidable: bundle more work into each task
instead. A sweep of 3,200 fits belongs in ~32 tasks of 100 fits, not 3,200
tasks. If you are certain, re-run with --force.
MSG
  exit 1
fi

if (( CONCURRENT > MAX_CONCURRENT )); then
  echo "REFUSING: --concurrent $CONCURRENT exceeds $MAX_CONCURRENT." >&2; exit 1
fi

PEAK_CPUS=$(( CONCURRENT * CPUS ))
if (( PEAK_CPUS > QOS_CPU_CAP )); then
  echo "REFUSING: peak ${PEAK_CPUS} CPUs exceeds the QOS '${QOS}' cap of ${QOS_CPU_CAP}." >&2
  echo "SLURM would queue the excess anyway; better to ask for what you can use." >&2
  exit 1
fi

# --- be a good neighbour: report the current state of the queue --------------
QUEUED=$(squeue -h 2>/dev/null | wc -l || echo 0)
MINE=$(squeue -h -u "$USER" 2>/dev/null | wc -l || echo 0)
IDLE=$(sinfo -h -o "%C" 2>/dev/null | cut -d/ -f2 || echo "?")
PCT=$(awk "BEGIN{printf \"%.1f\", 100*${PEAK_CPUS}/${CLUSTER_CPUS}}")

LOGDIR="${PROJECT}/logs"
mkdir -p "$LOGDIR"

cat <<MSG
submitting: ${NAME}
  array tasks      ${TASKS}  (at most ${CONCURRENT} running at once)
  per task         ${CPUS} CPUs, ${MEM}, ${TIME}, QOS ${QOS}${AFTER:+, after job ${AFTER} succeeds}
  peak footprint   ${PEAK_CPUS} CPUs = ${PCT}% of the cluster, ${PEAK_CPUS}/${QOS_CPU_CAP} of your '${QOS}' cap
  cluster now      ${QUEUED} jobs queued cluster-wide, ${MINE} of them yours, ${IDLE} CPUs idle
  logs             ${LOGDIR}/${NAME}_%A_%a.{out,err}
MSG

SBATCH=(sbatch --parsable
  -A cbu -p Main -q "$QOS"
  --job-name="$NAME"
  --array="0-$((TASKS - 1))%${CONCURRENT}"
  --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
  --output="${LOGDIR}/${NAME}_%A_%a.out"
  --error="${LOGDIR}/${NAME}_%A_%a.err"
  --export="ALL,PROJECT=${PROJECT},TASK_CMD=${CMD}"
  ${AFTER:+--dependency=afterok:${AFTER}}
  "${PROJECT}/scripts/slurm/_task.sh")

if (( DRY )); then
  printf '\ndry run, not submitted:\n'; printf '%q ' "${SBATCH[@]}"; printf '\n'; exit 0
fi

JID=$("${SBATCH[@]}")
echo
echo "submitted array job ${JID}"
echo "  watch:   squeue -u $USER"
echo "  cancel:  scancel ${JID}      (never scancel -u)"
