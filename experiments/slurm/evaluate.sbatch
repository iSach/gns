#!/usr/bin/env bash
# Evaluate every run on the full test split and rebuild the figures.
#
#   sbatch --dependency=afterany:<table1-job>:<ablations-job> \
#          experiments/slurm/evaluate.sbatch
#
#SBATCH --job-name=gns-evaluate
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/evaluate_%j.out
set -euo pipefail
source "$(dirname "$0")/common.sh"

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
