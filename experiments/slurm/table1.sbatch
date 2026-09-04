#!/usr/bin/env bash
# Table 1: one dataset per array task, one GPU each.
#
#   sbatch experiments/slurm/table1.sbatch
#
# Resumes from latest.pt, so re-submitting after a time limit or a preemption
# continues rather than restarting. Stop any local trainer on the same run
# directories first: bash experiments/stop_local.sh
#
#SBATCH --job-name=gns-table1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/table1_%A_%a.out
#SBATCH --requeue
set -euo pipefail
source "$(dirname "$0")/common.sh"

: "${GNS_STEPS:=3000000}"
DATASETS=(Goop WaterRamps SandRamps)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
run_dir="$GNS_RUNS/table1_$(echo "$dataset" | tr 'A-Z' 'a-z')"

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "dataset $dataset -> $run_dir, $GNS_STEPS steps"
train "$run_dir" --dataset "$dataset" --num-steps "$GNS_STEPS"
