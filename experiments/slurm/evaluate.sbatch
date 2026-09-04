#!/usr/bin/env bash
# Evaluate every run on the full test split and rebuild the figures.
#
#   sbatch --dependency=afterany:<table1-job>:<ablations-job> \
#          experiments/slurm/evaluate.sbatch
#
#SBATCH --job-name=gns-evaluate
#SBATCH --partition=gpu
#SBATCH --gpus=1
# Full GPUs only. workergpu037 and workergpu067 publish MIG slices
# (a100_2g.20gb), which --gpus=1 happily matches and which run this model at
# roughly a fifth of the rate. Override with sbatch --constraint=... to widen.
#SBATCH --constraint="a100-80gb|h100-80gb|rtxblackwell"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/evaluate_%j.out
set -euo pipefail
# Slurm runs a spool copy of this file, so "$0" is not in the repository.
# SLURM_SUBMIT_DIR is where sbatch was invoked; GNS_REPO overrides it.
source "${GNS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/experiments/slurm/common.sh"

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
GNS_FIGURES=${GNS_FIGURES:-figures} bash experiments/evaluate_all.sh
"$PYTHON" scripts/write_results_doc.py --results "$GNS_RUNS" --out docs/RESULTS.md

for dataset in Goop WaterRamps SandRamps; do
  lower=$(echo "$dataset" | tr 'A-Z' 'a-z')
  checkpoint="$GNS_RUNS/table1_$lower/best.pt"
  [ -e "$checkpoint" ] || continue
  for trajectory in 0 1; do
    "$PYTHON" -m gns.cli.render --checkpoint "$checkpoint" \
      --trajectory "$trajectory" --out-dir "${GNS_FIGURES:-figures}"
  done
done
