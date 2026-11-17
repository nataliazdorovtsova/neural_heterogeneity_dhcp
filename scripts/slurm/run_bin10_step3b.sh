#!/bin/bash
# Step 3b of RESEARCH_PLAN_BIN10.md (section 4b): a spread that changes with age.
#
#   bash scripts/slurm/run_bin10_step3b.sh --dry-run
#   bash scripts/slurm/run_bin10_step3b.sh
#   bash scripts/slurm/run_bin10_step3b.sh --from-select   (cv outputs exist)
#
#   cv      scripts/run_scale_cv.py, one task per specification (31)
#   select  scripts/summarise_scale.py --stage select (comparison, final fit)
#   boot    scripts/bootstrap_scale.py, 10 tasks x 50 replicates
#   report  scripts/summarise_scale.py --stage report (bands, shape check,
#           positions_scaled.csv, figure b7b)
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT="${PROJECT}/scripts/slurm/submit.sh"
DRY=(); FROM_SELECT=0
for a in "$@"; do
  [[ "$a" == "--dry-run" ]] && DRY=(--dry-run)
  [[ "$a" == "--from-select" ]] && FROM_SELECT=1
done
RUN=bin10_no7
R="export NH_RUN=${RUN};"
N_SPECS=39   # len(scripts/run_scale_cv.SPECS): L0 q0-7, L1 q0-7, L2 q1-7, L1+motion q0-7, L1 hinge q0-7

jid() {
  local out; out=$(bash "$SUBMIT" "$@" "${DRY[@]}")
  echo "$out" >&2
  if (( ${#DRY[@]} )); then echo "DRY$RANDOM"; else echo "$out" | awk '/^submitted array job/ {print $4}'; fi
}

AFTER=()
if (( ! FROM_SELECT )); then
  CV=$(jid --name ${RUN}_scalecv --tasks "$N_SPECS" --cpus 4 --mem 8G --concurrent 24 \
    --cmd "$R python scripts/run_scale_cv.py --index \$SLURM_ARRAY_TASK_ID")
  AFTER=(--after "$CV")
else
  CV=done
fi
SEL=$(jid --name ${RUN}_scalesel --tasks 1 --cpus 4 --mem 8G "${AFTER[@]}" \
  --cmd "$R python scripts/summarise_scale.py --stage select")
BOOT=$(jid --name ${RUN}_scaleboot --tasks 10 --cpus 4 --mem 8G --concurrent 10 --after "$SEL" \
  --cmd "$R python scripts/bootstrap_scale.py --chunk \$SLURM_ARRAY_TASK_ID --n 50")
REP=$(jid --name ${RUN}_scalerep --tasks 1 --cpus 4 --mem 16G --after "$BOOT" \
  --cmd "$R python scripts/summarise_scale.py --stage report")
echo "step 3b: cv $CV ($N_SPECS specs) -> select $SEL -> boot $BOOT -> report $REP" >&2
