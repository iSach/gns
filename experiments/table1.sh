#!/usr/bin/env bash
# Table 1: train the paper's default architecture on each 2D domain we replicate.
#
# The paper runs 20M steps with the learning rate decaying by a decade every 5M.
# GNS_STEPS below keeps that 4:1 ratio at a smaller budget; the trainer derives
# the decay horizon from --num-steps, and the run's config.json records it.
#
#   bash experiments/table1.sh              # default budget
#   GNS_STEPS=20000000 bash experiments/table1.sh   # the paper's budget
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GNS_DATA_ROOT:?set GNS_DATA_ROOT to the converted datasets}"
: "${GNS_RUNS:=runs}"
: "${GNS_STEPS:=3000000}"
: "${GNS_DATASETS:=Goop WaterRamps SandRamps}"
export GNS_DATA_ROOT

PYTHON=${PYTHON:-.venv/bin/python}
mkdir -p logs "$GNS_RUNS"

for dataset in $GNS_DATASETS; do
  run_dir="$GNS_RUNS/table1_${dataset,,}"
  echo "=== $dataset -> $run_dir ($GNS_STEPS steps)"
  "$PYTHON" -m gns.cli.train \
    --dataset "$dataset" \
    --run-dir "$run_dir" \
    --num-steps "$GNS_STEPS" \
    --num-workers 3 \
    --log-every 5000 \
    --checkpoint-every 25000 \
    --eval-every 100000 \
    --eval-trajectories 5 \
    >> "logs/table1_${dataset,,}.log" 2>&1
done
