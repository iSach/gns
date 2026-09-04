# Running on Slurm

These jobs resume from `latest.pt`, so re-submitting after a time limit or a
preemption continues rather than restarting. Every job writes to
`$GNS_RUNS`, which defaults to `/mnt/ceph/users/slewin/gns-repro/runs`.

## Before submitting

Stop any trainer on this workstation that targets the same run directories.
The jobs refuse to start if a checkpoint was written in the last three minutes,
so this is a guard rail rather than a race.

```bash
cd /mnt/home/slewin/neuralmpm/GNS-dev
bash experiments/stop_local.sh
```

## Submit

```bash
cd /mnt/home/slewin/neuralmpm/GNS-dev

# Table 1: three array tasks, one dataset and one GPU each, about 9 hours from
# the checkpoints that already exist.
sbatch experiments/slurm/table1.sbatch

# Figure 4: six array tasks, three ablation runs sharing each GPU, about 3
# hours. Independent of the Table 1 jobs, so submit both at once.
sbatch experiments/slurm/ablations.sbatch

# Evaluation and figures, once both finish. Substitute the two job ids.
sbatch --dependency=afterany:<table1-id>:<ablations-id> \
       experiments/slurm/evaluate.sbatch
```

Watch them with `squeue -u $USER` and `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`.

## Knobs

| variable | default | effect |
| --- | --- | --- |
| `GNS_STEPS` | 3000000 | Table 1 step budget. The trainer scales the learning-rate horizon with it. |
| `GNS_ABL_STEPS` | 400000 | Ablation step budget, shared by all seventeen runs. |
| `GNS_ABL_PER_TASK` | 3 | Ablation runs per GPU. Three is where the GPU saturates. |
| `GNS_RUNS` | `/mnt/ceph/users/slewin/gns-repro/runs` | Where run directories go. |
| `GNS_DATA_ROOT` | `/mnt/ceph/users/slewin/gns-repro/datasets` | Where the converted datasets are. |

Set them in the environment before `sbatch`, for example
`GNS_STEPS=6000000 sbatch experiments/slurm/table1.sbatch`.

## Why three runs share a GPU in the ablation job

One trainer leaves the GPU idle while it waits on the loader, so it reaches
about 79 steps per second on its own. Three concurrent trainers reach about 41
each, which is 2.2x the aggregate. Beyond three the GPU saturates and the
returns stop. The Table 1 jobs get a GPU each instead, because there the wall
clock of a single run is what matters.
