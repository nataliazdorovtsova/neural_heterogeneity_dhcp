#!/bin/bash
# Step 2 of RESEARCH_PLAN_BIN10.md: model selection on the 12 lobe features
# (DECISIONS.md D19). Two runs, same settings:
#   bin10       all training infants
#   bin10_no7   without the 7 orbitofrontal-low infants (the D19 check; no validation)
#
#   bash scripts/slurm/run_bin10_step2.sh --dry-run
#   bash scripts/slurm/run_bin10_step2.sh
#
#   1 cv       K = 1, 2 at q = 0-6 (D = 12, so q <= 7 is identifiable), 1000 sweeps, 20 starts
#   2 select   paired comparison, 1-SE rule, sweep summary
#   3 final    full-training fits of the selected model and the runner-up
#   4 valid    validation set, bin10 only (a third look, flagged as such)
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
  R="export NH_RUN=${RUN};"
  CV=$(jid --name ${RUN}_cv --tasks 14 --cpus 4 --mem 8G --concurrent 14 \
    --cmd "$R K=(1 1 1 1 1 1 1 2 2 2 2 2 2 2); Q=(0 1 2 3 4 5 6 0 1 2 3 4 5 6); python scripts/run_spec.py --n-iter 1000 --K \${K[\$SLURM_ARRAY_TASK_ID]} --q \${Q[\$SLURM_ARRAY_TASK_ID]}")
  SEL=$(jid --name ${RUN}_select --tasks 1 --cpus 2 --mem 8G --after "$CV" \
    --cmd "$R python scripts/compare_models.py && python scripts/collect_sweep.py")
  FIN=$(jid --name ${RUN}_final --tasks 2 --cpus 4 --mem 8G --concurrent 2 --after "$SEL" \
    --cmd "$R A=('--from-selection selected' '--from-selection runner_up'); python scripts/fit_final.py --skip-existing \${A[\$SLURM_ARRAY_TASK_ID]}")
  if [[ "$RUN" == "bin10" ]]; then
    VAL=$(jid --name ${RUN}_valid --tasks 1 --cpus 2 --mem 8G --after "$FIN" \
      --cmd "$R python scripts/validate_once.py")
    echo "$RUN: cv $CV -> select $SEL -> final $FIN -> valid $VAL" >&2
  else
    echo "$RUN: cv $CV -> select $SEL -> final $FIN" >&2
  fi
done
