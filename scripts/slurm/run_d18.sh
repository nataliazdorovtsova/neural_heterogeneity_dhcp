#!/bin/bash
# The D18 rerun: the core pipeline on the D18 feature set (clustering with one
# normaliser for every infant), written to results/d18/ and figures/d18/ so the
# original run (results/, figures/) is untouched. See DECISIONS.md D18.
#
#   bash scripts/slurm/run_d18.sh --dry-run     # show every submission
#   bash scripts/slurm/run_d18.sh               # submit the chain
#
# Same settings as the original run (SUPPLEMENT S16-S25), normal QOS only:
#   1 cv       17 specs, 5-fold CV at 1000 sweeps (K = 1 at q 0-70, K = 2 at q 0-25)
#   2 select   paired comparison, 1-SE rule, sweep summary
#   3 final    full-training fits: selected, runner-up, and K = 1 at q = 5, 15, 25
#   4 valid    validation set (second look, flagged as such), after 3
#      perm    1,000 permutations at q = 15, after 3
#      boot    1,000 parametric bootstrap replicates of the selected model, after 3
#      split   split-half, 10 splits each at q = 15 and q = 70, after 3
#   5 report   summaries and figures, after everything in 4
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT="${PROJECT}/scripts/slurm/submit.sh"
DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(--dry-run)
R="export NH_RUN=d18;"

jid() {   # submit, echo the report to stderr, return the job id
  local out; out=$(bash "$SUBMIT" "$@" "${DRY[@]}")
  echo "$out" >&2
  if (( ${#DRY[@]} )); then echo "DRY$RANDOM"; else echo "$out" | awk '/^submitted array job/ {print $4}'; fi
}

CV=$(jid --name d18_cv --tasks 17 --cpus 8 --mem 32G --concurrent 16 \
  --cmd "$R I=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 35 36 37); python scripts/run_spec.py --n-iter 1000 --index \${I[\$SLURM_ARRAY_TASK_ID]}")

SEL=$(jid --name d18_select --tasks 1 --cpus 2 --mem 8G --after "$CV" \
  --cmd "$R python scripts/compare_models.py && python scripts/collect_sweep.py")

FIN=$(jid --name d18_final --tasks 5 --cpus 8 --mem 32G --concurrent 5 --after "$SEL" \
  --cmd "$R A=('--from-selection selected' '--from-selection runner_up' '--K 1 --q 5' '--K 1 --q 15' '--K 1 --q 25'); python scripts/fit_final.py --skip-existing \${A[\$SLURM_ARRAY_TASK_ID]}")

VAL=$(jid --name d18_valid --tasks 1 --cpus 2 --mem 8G --after "$FIN" \
  --cmd "$R python scripts/validate_once.py")

PERM=$(jid --name d18_perm --tasks 25 --cpus 4 --mem 16G --concurrent 25 --after "$FIN" \
  --cmd "$R python scripts/permutation_null.py --chunk \$SLURM_ARRAY_TASK_ID --n-perm 40")

BOOT=$(jid --name d18_boot --tasks 40 --cpus 4 --mem 16G --concurrent 32 --after "$FIN" \
  --cmd "$R python scripts/bootstrap.py --per-replicate --chunk \$SLURM_ARRAY_TASK_ID --n-boot 25")

SPLIT=$(jid --name d18_split --tasks 20 --cpus 4 --mem 16G --concurrent 8 --after "$FIN" \
  --cmd "$R Q=(15 70); python scripts/split_half.py --split \$((SLURM_ARRAY_TASK_ID % 10)) --q \${Q[\$((SLURM_ARRAY_TASK_ID / 10))]}")

jid --name d18_report --tasks 1 --cpus 4 --mem 16G --after "${VAL}:${PERM}:${BOOT}:${SPLIT}" \
  --cmd "$R python scripts/summarise_axes_significance.py && python scripts/summarise_bootstrap.py && python scripts/weight_scale_check.py && python scripts/characterise_tail.py && python scripts/make_figures.py && python scripts/make_axes_figures.py && python scripts/make_brain_figures.py && python scripts/make_trajectory_figures.py" >/dev/null

echo "chain: cv $CV -> select $SEL -> final $FIN -> valid $VAL, perm $PERM, boot $BOOT, split $SPLIT -> report" >&2
