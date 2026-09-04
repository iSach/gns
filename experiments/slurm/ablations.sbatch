#!/usr/bin/env bash
# Figure 4 ablations on Goop: three runs share one GPU per array task.
#
#   sbatch experiments/slurm/ablations.sbatch
#
# Three concurrent trainers on one GPU deliver about 2.2x the aggregate step
# rate of one, because a single trainer leaves the GPU idle while it waits on
# the loader. Six tasks cover the seventeen configurations in gns.ablations.
#
#SBATCH --job-name=gns-ablations
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/slurm/ablations_%A_%a.out
#SBATCH --requeue
set -euo pipefail
source "$(dirname "$0")/common.sh"

: "${GNS_ABL_STEPS:=400000}"
: "${GNS_ABL_DATASET:=Goop}"
: "${GNS_ABL_PER_TASK:=3}"

total=$("$PYTHON" -m gns.ablations --count)
first=$((SLURM_ARRAY_TASK_ID * GNS_ABL_PER_TASK))

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
pids=()
for offset in $(seq 0 $((GNS_ABL_PER_TASK - 1))); do
  index=$((first + offset))
  [ "$index" -lt "$total" ] || break

  IFS=$'\t' read -r name axis json flags < <("$PYTHON" -m gns.ablations --index "$index")
  # shellcheck disable=SC2206  # word splitting is how the flags are passed on
  flag_array=($flags)
  run_dir="$GNS_RUNS/$name"
  mkdir -p "$run_dir"
  printf '{"ablation_axis": "%s", "ablation_value": %s}\n' "$axis" "$json" \
    > "$run_dir/ablation.json"

  echo "=== [$index] $name  ${flag_array[*]}"
  train "$run_dir" --dataset "$GNS_ABL_DATASET" --num-steps "$GNS_ABL_STEPS" \
    "${flag_array[@]}" > "logs/slurm/$name.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
