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
#   --time HH:MM:SS    wall limit per task  (default 02:00:00)
#   --concurrent N     max tasks running at once (default 4)
#   --dry-run          print the sbatch command and exit
#   --force            allow more than MAX_TASKS (think first)
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAX_TASKS=64            # refuse bigger arrays without --force
MAX_CONCURRENT=8        # never run more than this many at once
CLUSTER_CPUS=1184
QOS_CPU_CAP=128         # per-user cap on QOS 'normal'

NAME=""; TASKS=""; CMD=""; CPUS=8; MEM=16G; TIME=02:00:00
CONCURRENT=4; DRY=0; FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --tasks) TASKS="$2"; shift 2;;
    --cmd) CMD="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --concurrent) CONCURRENT="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    --force) FORCE=1; shift;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[[ -n "$NAME" && -n "$TASKS" && -n "$CMD" ]] || { echo "need --name, --tasks and --cmd" >&2; exit 2; }

# --- guardrails --------------------------------------------------------------
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
  echo "REFUSING: peak ${PEAK_CPUS} CPUs exceeds the QOS 'normal' cap of ${QOS_CPU_CAP}." >&2
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
  per task         ${CPUS} CPUs, ${MEM}, ${TIME}
  peak footprint   ${PEAK_CPUS} CPUs = ${PCT}% of the cluster, ${PEAK_CPUS}/${QOS_CPU_CAP} of your QOS cap
  cluster now      ${QUEUED} jobs queued cluster-wide, ${MINE} of them yours, ${IDLE} CPUs idle
  logs             ${LOGDIR}/${NAME}_%A_%a.{out,err}
MSG

SBATCH=(sbatch --parsable
  -A cbu -p Main
  --job-name="$NAME"
  --array="0-$((TASKS - 1))%${CONCURRENT}"
  --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
  --output="${LOGDIR}/${NAME}_%A_%a.out"
  --error="${LOGDIR}/${NAME}_%A_%a.err"
  --wrap="cd '${PROJECT}' && source activate.sh && ${CMD}")

if (( DRY )); then
  printf '\ndry run, not submitted:\n'; printf '%q ' "${SBATCH[@]}"; printf '\n'; exit 0
fi

JID=$("${SBATCH[@]}")
echo
echo "submitted array job ${JID}"
echo "  watch:   squeue -u $USER"
echo "  cancel:  scancel ${JID}"
