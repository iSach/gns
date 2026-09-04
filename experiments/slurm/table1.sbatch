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
# Full GPUs only. workergpu037 and workergpu067 publish MIG slices
# (a100_2g.20gb), which --gpus=1 happily matches and which run this model at
# roughly a fifth of the rate. Override with sbatch --constraint=... to widen.
#SBATCH --constraint="a100-80gb|h100-80gb|rtxblackwell"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/table1_%A_%a.out
#SBATCH --requeue
set -euo pipefail
# Slurm runs a spool copy of this file, so "$0" is not in the repository.
# SLURM_SUBMIT_DIR is where sbatch was invoked; GNS_REPO overrides it.
source "${GNS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/experiments/slurm/common.sh"

: "${GNS_STEPS:=3000000}"
# GNS_RUN_PREFIX keeps a longer budget in its own run directories, so a reduced
# run stays intact as a separate data point instead of being resumed into.
: "${GNS_RUN_PREFIX:=table1}"
DATASETS=(Goop WaterRamps SandRamps)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
run_dir="$GNS_RUNS/${GNS_RUN_PREFIX}_$(echo "$dataset" | tr 'A-Z' 'a-z')"

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "dataset $dataset -> $run_dir, $GNS_STEPS steps"
train "$run_dir" --dataset "$dataset" --num-steps "$GNS_STEPS"
