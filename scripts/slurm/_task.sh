#!/bin/bash
# Body of every array task. A real job script with a bash shebang, because
# `sbatch --wrap` is interpreted by /bin/sh (dash on these nodes), which has no
# `source` builtin -- and activate.sh needs bash.
set -euo pipefail
cd "$PROJECT"
source activate.sh
echo "task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus"
eval "$TASK_CMD"
