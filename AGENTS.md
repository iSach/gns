# AGENTS.md

Read `README.md` first for what this is and how to run it. This file adds the
architecture and the traps that only show up after reading several files.

## What this is

A reimplementation of Sanchez-Gonzalez et al., ICML 2020, kept faithful enough
to argue about published numbers with. Numerical correctness beats elegance;
the reference is the authors' TensorFlow code, at
`deepmind-research/learning_to_simulate`. Keep a read-only clone of it beside
this repository and check any change to the model against it.

## Where things live

Three environment variables point at everything outside the repository:

| variable | holds |
| --- | --- |
| `GNS_RAW_ROOT` | the released TFRecords, one directory per dataset |
| `GNS_DATA_ROOT` | the converted HDF5 datasets |
| `GNS_RUNS` | run directories: checkpoints, `curve.jsonl`, results |

Put all three on scratch or scale storage, not in the repository. Nothing under
them is in Git.

## The pipeline

```
scripts/download_datasets.sh   released TFRecords -> raw/
cli/convert.py                 raw/ -> datasets/<name>/{train,valid,test}/sim_*.h5
      + metadata.json (copied unchanged) + shapes.json (measured)
data/trajectories.py           HDF5 -> positions [T, N, dim] + types [N]
data/onestep.py                seven-frame windows, noise, radius graph, batching
models/simulator.py            features, normalization, semi-implicit Euler
models/graph_network.py        encoder, 10 unshared message-passing steps, decoder
training/loop.py               Adam, the decaying rate, validation selection
evaluation/metrics.py          one-step MSE, rollout, rollout MSE
```

Read in this order for a change to training: `cli/train.py` ->
`training/loop.py` -> `models/simulator.py` -> `models/graph_network.py`.

## Decisions that are easy to get wrong

- **Noise scale has two conventions.** The paper quotes `sigma_v = 3e-4` per
  input step; the released code's `noise_std` is the standard deviation the
  random walk has reached at the *last* of the five steps, `6.7e-4`. They differ
  by `sqrt(5)`. Everything in this repository uses the code's convention, and
  `gns.noise.per_step_std` converts. The Figure 4 noise axis is in the paper's
  units, so `experiments/ablations.sh` multiplies by `sqrt(5)`.
- **Normalization statistics are inflated by the noise scale.** Both the
  velocity and the acceleration std are combined with `noise_std` in
  quadrature, so the target stays unit variance once noise is added. Changing
  `--noise-std` therefore changes the output scale of the model, not just the
  data. This is why a run with `--noise-std 0` is not comparable in loss units
  to a default run.
- **The loss sums over spatial axes and averages over particles.** The released
  code divides the summed squared error by the *particle* count, not the element
  count, so the reported training loss is `dim` times the per-element MSE. The
  evaluation metrics are per element. Do not compare the two directly.
- **Obstacle particles are handled three different ways.** They get no training
  noise, they are excluded from the loss, and during a rollout their positions
  are overwritten with the ground truth at every step. In a one-step evaluation
  they are *not* overwritten, so the released code's one-step metric includes
  their prediction error.
- **Two metric conventions, both reported.** The released code averages over all
  particles; obstacles then contribute zero to a rollout and real error to a
  one-step evaluation. `result_*.json` carries `*_free_particles` and
  `*_all_particles` and the figures take a `--convention` flag. Goop has no
  obstacle particles at all, so its numbers are unambiguous; that is the cleanest
  dataset to compare against the paper.
- **Goop, Water, Sand and WaterDrop have no boundary particles in the released
  data.** Their walls are the clipped wall-distance node feature. The NeuralMPM
  copies of these datasets add about 1284 synthetic wall particles per scene,
  which changes the graph. Convert from the released TFRecords, not from those.
- **The paper pads batches; we do not.** Fixed-size tensors were a TPU
  requirement, and the paper calls the packing equivalent to a particle-count
  batch size. On a GPU the padded slots are wasted bandwidth, and this model is
  bandwidth bound, so batches are the nominal two windows with no padding.
- **`lr_decay_steps` scales with the budget.** The paper decays by a decade
  every 5M steps over 20M. A shorter run keeps the 4:1 ratio, so the rate still
  reaches its floor as training ends. Every run's `config.json` records the value
  actually used.

## Performance notes

Measured on one RTX PRO 6000 Blackwell, WaterRamps, batch of two windows:

| setting | steps/s |
| --- | --- |
| eager | 79 |
| `torch.compile(dynamic=True)` | 95 |
| three concurrent training processes | 41 each, 123 total |

The model is bandwidth bound, not launch bound: CUDA graph capture changed
nothing. Running one process per dataset is worth about 2.2x over running them
one after another, because a single process leaves the GPU idle while it waits
on the loader. The data loader is not the bottleneck at 3 workers, but only four
CPU cores are available on this box, so do not raise `--num-workers` much.

## Operational traps

- **Never edit a script while a job is running it.** Bash reads a script
  incrementally by byte offset, so editing `experiments/evaluate_all.sh` mid-job
  made it resume inside a word and try to run `ll` instead of `figures all`. The
  whole evaluation had already succeeded; only the last line died. Copy the
  script, or wait.
- **`--gpus=1` matches a MIG slice.** Two nodes here publish `a100_2g.20gb`, and
  landing on one costs a factor of five. The Slurm jobs constrain to full GPUs.
- **A Slurm batch script runs from a spool copy**, so `$0` is not in the
  repository. `tests/test_slurm_scripts.py` reproduces that and would have
  caught it.

## Hard rules

- Checkpoint selection uses validation rollout MSE only. The test split is read
  by `gns-evaluate` and nothing else.
- Never commit datasets, checkpoints, runs, rendered media or logs.
- Do not use Slurm. Site policy reserves job submission for the user, and these
  runs belong on the local GPU as background processes.
