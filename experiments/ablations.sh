#!/usr/bin/env bash
# Figure 4 (and part of C.1): one axis at a time on Goop, everything else at the
# paper's default.  The paper trains each of these to convergence; here they all
# share one reduced budget, so the comparison between bars is fair even though
# the absolute level sits above the published one.
#
# The default architecture is the default point of four separate axes, so it is
# trained once as `ablation_default` and the figure builder reuses it.
#
#   bash experiments/ablations.sh
#   GNS_ABL_STEPS=1000000 GNS_ABL_JOBS=2 bash experiments/ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GNS_DATA_ROOT:=/mnt/ceph/users/slewin/gns-repro/datasets}"
: "${GNS_RUNS:=/mnt/ceph/users/slewin/gns-repro/runs}"
: "${GNS_ABL_STEPS:=400000}"
: "${GNS_ABL_DATASET:=Goop}"
: "${GNS_ABL_JOBS:=3}"
export GNS_DATA_ROOT GNS_RUNS GNS_ABL_STEPS GNS_ABL_DATASET

PYTHON=${PYTHON:-.venv/bin/python}
export PYTHON
mkdir -p logs "$GNS_RUNS"

# The trainer parameterises noise by its standard deviation at the last of the
# C = 5 input steps; the paper's axis is the per-step sigma_v, sqrt(5) smaller.
noise_last_step () { "$PYTHON" -c "print($1 * 5 ** 0.5)"; }

# One job per line: <axis> <label> <json-value> [extra trainer flags]
jobs_file=$(mktemp)
trap 'rm -f "$jobs_file"' EXIT
{
  echo "default default null"
  for m in 1 2 5 15; do
    echo "message_passing_steps $m $m --message-passing-steps $m"
  done
  echo "shared_processor true true --shared-processor"
  for r in 0.003 0.007 0.011 0.02 0.03; do
    echo "connectivity_radius $r $r --connectivity-radius $r"
  done
  for sigma in 0 3e-05 0.0001 0.001 0.003; do
    echo "noise_std_per_step $sigma $sigma --noise-std $(noise_last_step "$sigma")"
  done
  echo "use_relative_positions false false --absolute-encoder"
} > "$jobs_file"

run_one () {
  local axis=$1 label=$2 json=$3; shift 3
  local name="ablation_${axis}_${label//./p}"
  [ "$axis" = default ] && name="ablation_default"
  local run_dir="$GNS_RUNS/$name"
  mkdir -p "$run_dir"
  printf '{"ablation_axis": "%s", "ablation_value": %s}\n' "$axis" "$json" \
    > "$run_dir/ablation.json"
  echo "=== $name"
  "$PYTHON" -m gns.cli.train \
    --dataset "$GNS_ABL_DATASET" --run-dir "$run_dir" \
    --num-steps "$GNS_ABL_STEPS" --num-workers 2 \
    --log-every 5000 --checkpoint-every 25000 \
    --eval-every 50000 --eval-trajectories 5 \
    "$@" >> "logs/$name.log" 2>&1
}
export -f run_one

xargs -P "$GNS_ABL_JOBS" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
