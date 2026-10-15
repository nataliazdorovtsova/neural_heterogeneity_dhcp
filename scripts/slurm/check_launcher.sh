#!/usr/bin/env bash
# Pre-sbatch gate, adapted from Maros Rovny's launcher-preflight check.
#
#   bash scripts/slurm/check_launcher.sh [file ...]   # check the named scripts
#   bash scripts/slurm/check_launcher.sh              # check every git-tracked *.sh
#
# Exits 1 if any file has Windows (CRLF) line endings: sbatch refuses those
# outright ("Batch script contains DOS line breaks"), and editing over the U:
# mapping from Windows is how they creep in. Warnings do not fail.
set -u

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "$#" -gt 0 ]; then
  files=("$@")
else
  mapfile -t files < <(git -C "$PROJECT" ls-files '*.sh' | sed "s|^|$PROJECT/|")
fi

fail=0
for f in "${files[@]}"; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f" >&2; fail=1; continue
  fi
  if grep -qI $'\r' "$f"; then
    echo "CRLF: $f    fix: sed -i 's/\r\$//' '$f'"; fail=1
  fi
  if [[ "$f" == */scripts/slurm/* ]] && [ "$(head -c 11 "$f")" != '#!/bin/bash' ] && [ "$(head -c 19 "$f")" != '#!/usr/bin/env bash' ]; then
    echo "WARN (no bash shebang; sbatch would run it under dash): $f"
  fi
  if grep -E '/home/nz01' "$f" | grep -qv '^\s*#'; then
    echo "WARN (/home referenced; outputs belong on /imaging/astle, home has a quota): $f"
  fi
done

[ "$fail" -eq 0 ] && echo "OK: ${#files[@]} file(s), no CRLF"
exit "$fail"
