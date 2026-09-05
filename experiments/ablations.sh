#!/usr/bin/env bash
# Figure 4 (and part of C.1) on this workstation: one axis at a time on Goop,
# everything else at the paper's default.  The grid lives in gns.ablations, so
# this and experiments/slurm/ablations.sbatch always run the same seventeen
# configurations.
#
# The paper trains each of these to convergence; here they share one reduced
# budget, so the comparison between bars is fair even though every bar sits
# above the published level.
#
#   bash experiments/ablations.sh
#   GNS_ABL_STEPS=1000000 GNS_ABL_JOBS=2 bash experiments/ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GNS_DATA_ROOT:?set GNS_DATA_ROOT to the converted datasets}"
: "${GNS_RUNS:=runs}"
: "${GNS_ABL_STEPS:=400000}"
: "${GNS_ABL_DATASET:=Goop}"
: "${GNS_ABL_JOBS:=3}"
: "${PYTHON:=.venv/bin/python}"
export GNS_DATA_ROOT GNS_RUNS GNS_ABL_STEPS GNS_ABL_DATASET PYTHON
mkdir -p logs "$GNS_RUNS"

run_one () {
  local index=$1
  IFS=$'\t' read -r name axis json flags < <("$PYTHON" -m gns.ablations --index "$index")
  # shellcheck disable=SC2206  # word splitting is how the flags are passed on
  local flag_array=($flags)
  local run_dir="$GNS_RUNS/$name"
  mkdir -p "$run_dir"
  printf '{"ablation_axis": "%s", "ablation_value": %s}\n' "$axis" "$json" \
    > "$run_dir/ablation.json"
  echo "=== [$index] $name ${flag_array[*]+${flag_array[*]}}"
  "$PYTHON" -m gns.cli.train \
    --dataset "$GNS_ABL_DATASET" --run-dir "$run_dir" \
    --num-steps "$GNS_ABL_STEPS" --num-workers 2 \
    --log-every 5000 --checkpoint-every 10000 \
    --eval-every 50000 --eval-trajectories 5 \
    ${flag_array[@]+"${flag_array[@]}"} >> "logs/$name.log" 2>&1
}
export -f run_one

seq 0 $(( $("$PYTHON" -m gns.ablations --count) - 1 )) |
  xargs -P "$GNS_ABL_JOBS" -I {} bash -c 'run_one {}'
