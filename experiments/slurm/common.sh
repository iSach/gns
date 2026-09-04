# Shared settings for the Slurm jobs. Sourced, not executed.
#
# Override any of these in the environment before sbatch, e.g.
#   GNS_STEPS=6000000 sbatch experiments/slurm/table1.sbatch

: "${GNS_REPO:=/mnt/home/slewin/neuralmpm/GNS-dev}"
: "${GNS_DATA_ROOT:=/mnt/ceph/users/slewin/gns-repro/datasets}"
: "${GNS_RUNS:=/mnt/ceph/users/slewin/gns-repro/runs}"
: "${PYTHON:=$GNS_REPO/.venv/bin/python}"
export GNS_DATA_ROOT

cd "$GNS_REPO"
mkdir -p "$GNS_RUNS" logs/slurm

# Two loader workers per training process is enough: the k-d tree keeps up with
# the GPU at about 240 batches per second per worker, and the GPU tops out near
# 80.  Asking for more only makes the job harder to schedule.
: "${GNS_WORKERS:=2}"

# Checkpoints are the only thing standing between a preemption and lost work,
# and the trainer resumes from latest.pt automatically, so write them often.
: "${GNS_CHECKPOINT_EVERY:=10000}"

refuse_if_already_running () {   # refuse_if_already_running <run-dir>
  local checkpoint="$1/latest.pt"
  if [ -e "$checkpoint" ] && [ -n "$(find "$checkpoint" -mmin -3 2>/dev/null)" ]; then
    echo "REFUSING: $checkpoint was written in the last 3 minutes, so another" >&2
    echo "trainer is probably live on this run directory. Stop it first:" >&2
    echo "  bash experiments/stop_local.sh" >&2
    exit 3
  fi
}

# Set GNS_DRY_RUN=1 to print the trainer command instead of running it, which
# is how these scripts are checked without burning a GPU allocation.
train () {                        # train <run-dir> <trainer args...>
  local run_dir=$1; shift
  if [ "${GNS_DRY_RUN:-0}" = 1 ]; then
    echo "DRY RUN: train --run-dir $run_dir $*"
    return 0
  fi
  refuse_if_already_running "$run_dir"
  "$PYTHON" -m gns.cli.train \
    --run-dir "$run_dir" \
    --num-workers "$GNS_WORKERS" \
    --checkpoint-every "$GNS_CHECKPOINT_EVERY" \
    --log-every 5000 \
    --eval-every 100000 \
    --eval-trajectories 5 \
    "$@"
}
