#!/bin/bash
# Step 3 of RESEARCH_PLAN_BIN10.md: trajectories and the spread around them,
# for the selected one-group model of each run (bin10: q = 5; bin10_no7: q = 6).
#
#   bash scripts/slurm/run_bin10_step3.sh --dry-run
#   bash scripts/slurm/run_bin10_step3.sh
#
#   boot    1,000 parametric bootstrap replicates (10 tasks x 100, one file each)
#   perm    1,000 permutations at the selected q (10 tasks x 100)
#   split   10 split-halves at the selected q
#   report  scripts/summarise_bin10_step3.py (summary, positions, figures b6-b9)
# Concurrency keeps both runs together within the 128-CPU normal cap.
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT="${PROJECT}/scripts/slurm/submit.sh"
DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(--dry-run)

jid() {
  local out; out=$(bash "$SUBMIT" "$@" "${DRY[@]}")
  echo "$out" >&2
  if (( ${#DRY[@]} )); then echo "DRY$RANDOM"; else echo "$out" | awk '/^submitted array job/ {print $4}'; fi
}

for RUN in bin10 bin10_no7; do
  Q=$(python3 -c "import json; print(json.load(open('${PROJECT}/results/${RUN}/selection.json'))['selected']['q'])")
  R="export NH_RUN=${RUN};"
  BOOT=$(jid --name ${RUN}_boot --tasks 10 --cpus 4 --mem 8G --concurrent 8 \
    --cmd "$R python scripts/bootstrap.py --per-replicate --chunk \$SLURM_ARRAY_TASK_ID --n-boot 100")
  PERM=$(jid --name ${RUN}_perm --tasks 10 --cpus 4 --mem 8G --concurrent 6 \
    --cmd "$R python scripts/permutation_null.py --q $Q --chunk \$SLURM_ARRAY_TASK_ID --n-perm 100")
  SPLIT=$(jid --name ${RUN}_split --tasks 10 --cpus 4 --mem 8G --concurrent 2 \
    --cmd "$R python scripts/split_half.py --q $Q --n-match $Q --split \$SLURM_ARRAY_TASK_ID")
  REP=$(jid --name ${RUN}_step3 --tasks 1 --cpus 2 --mem 8G --after "${BOOT}:${PERM}:${SPLIT}" \
    --cmd "$R python scripts/summarise_bin10_step3.py")
  echo "$RUN (q = $Q): boot $BOOT, perm $PERM, split $SPLIT -> report $REP" >&2
done
