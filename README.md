# GNS — Learning to Simulate Complex Physics with Graph Networks

A PyTorch reimplementation of Graph Network-based Simulators, built to reproduce
the numbers in

> Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying,
> Jure Leskovec, Peter W. Battaglia.
> *Learning to Simulate Complex Physics with Graph Networks.* ICML 2020.
> [arXiv:2002.09405](https://arxiv.org/abs/2002.09405)

The authors released a TensorFlow 1 / Sonnet / graph_nets implementation under
[`deepmind-research/learning_to_simulate`](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate).
Every modelling decision here follows that code; where the paper and the code
disagree, `docs/REPLICATION.md` says which one this repository implements and
why.

This is a baseline for later work, so the priority is that the numbers are
defensible and the code is reusable, not that it is clever.

## What is reproduced

| target | figure | status |
| --- | --- | --- |
| One-step and rollout MSE on Goop, WaterRamps, SandRamps | Table 1 / C.4 | `figures/table1.png` |
| Rollout error as a function of rollout step | Figure C.3 | `figures/error_vs_time.png` |
| Ablations on Goop: message passing steps, shared processor, connectivity radius, noise scale, relative encoder | Figure 4(a-j) | `figures/ablations.png` |
| Qualitative rollouts against ground truth | Figure 3 | `figures/rollout_*.png` |

## Install

```bash
uv sync --extra cu128          # or --extra cpu on a machine without a GPU
uv run --extra cu128 pytest tests/
```

The two accelerator extras are mutually exclusive and resolve from explicit
PyTorch indexes; switching between them re-syncs the environment.

## Data

The released datasets are TFRecord files on a public bucket. Download and
convert them once:

```bash
bash scripts/download_datasets.sh WaterRamps SandRamps Goop
uv run python -m gns.cli.convert \
    --raw  $GNS_RAW_ROOT/Goop \
    --out  $GNS_DATA_ROOT/Goop
```

Conversion writes the HDF5 layout this repository and NeuralMPM share,
`<dataset>/{train,valid,test}/sim_<i>.h5`, holding `boundary`, `particles` and
`types`; it copies the released `metadata.json` unchanged and measures
`shapes.json`. Point `GNS_DATA_ROOT` at the parent of the converted datasets.

## Train and evaluate

```bash
# The paper's default architecture; --num-steps also sets the decay horizon.
uv run python -m gns.cli.train --dataset Goop --run-dir runs/goop --num-steps 3000000

# One-step and rollout MSE on the full test split.
uv run python -m gns.cli.evaluate --checkpoint runs/goop/best.pt --split test

# Ground truth beside prediction, and a video.
uv run python -m gns.cli.render --checkpoint runs/goop/best.pt --trajectory 0 --video

# Every figure from every evaluated run.
uv run python -m gns.cli.figures all --results runs --out figures
```

Checkpoints are selected on validation rollout MSE over five held-out
trajectories, as in Section 4.3. The test split is only read by `gns-evaluate`.

The batch is the paper's nominal two windows. `experiments/table1.sh` and
`experiments/ablations.sh` are the exact recipes behind the figures above.

`AGENTS.md` documents the architecture and the traps worth knowing before
changing anything; `docs/REPLICATION.md` records every choice made where the
paper and the released code disagree.

## Layout

```
src/gns/
  datasets.py       dataset registry
  metadata.py       bounds, connectivity radius, normalization statistics
  neighbors.py      radius connectivity, k-d tree and pairwise backends
  noise.py          random-walk noise on the input velocities
  paper.py          the published numbers every figure is compared against
  data/             TFRecord reader, HDF5 trajectories, one-step windows
  models/           the graph network and the learned simulator
  training/         the training loop
  evaluation/       rollouts and the two published metrics
  render/           rollout figures and videos
  cli/              convert, train, evaluate, render, figures
```
