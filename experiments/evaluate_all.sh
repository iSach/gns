#!/usr/bin/env bash
# Evaluate every finished run on the full test split and rebuild the figures.
#
#   bash experiments/evaluate_all.sh
#   GNS_EVAL_CHECKPOINT=latest.pt bash experiments/evaluate_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GNS_DATA_ROOT:=/mnt/ceph/users/slewin/gns-repro/datasets}"
: "${GNS_RUNS:=/mnt/ceph/users/slewin/gns-repro/runs}"
: "${GNS_EVAL_CHECKPOINT:=best.pt}"
: "${GNS_FIGURES:=figures}"
export GNS_DATA_ROOT

PYTHON=${PYTHON:-.venv/bin/python}

for checkpoint in "$GNS_RUNS"/*/"$GNS_EVAL_CHECKPOINT"; do
  [ -e "$checkpoint" ] || continue
  result="$(dirname "$checkpoint")/result_test_$(basename "$checkpoint" .pt).json"
  if [ -e "$result" ] && [ "$result" -nt "$checkpoint" ]; then
    echo "up to date: $result"
    continue
  fi
  echo "=== $checkpoint"
  "$PYTHON" -m gns.cli.evaluate --checkpoint "$checkpoint" --split test
done

"$PYTHON" -m gns.cli.figures all --results "$GNS_RUNS" --out "$GNS_FIGURES" \
  --pattern "**/result_test_$(basename "$GNS_EVAL_CHECKPOINT" .pt).json"
