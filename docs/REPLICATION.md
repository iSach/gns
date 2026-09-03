# Replication notes

Every place this implementation had to choose between the paper's text, the
released TensorFlow code, or an unstated detail, and what it chose.

## Source of truth

- Paper: [arXiv:2002.09405](https://arxiv.org/abs/2002.09405), including
  Supplementary A-D.
- Code: `deepmind-research/learning_to_simulate` (TensorFlow 1, Sonnet 1,
  graph_nets 1.1).
- Data: the authors' TFRecords and `metadata.json` from
  `storage.googleapis.com/learning-to-simulate-complex-physics`.

Where paper and code disagree, this repository follows the code, because the
published numbers were produced by the code.

## Verified against the paper before training anything

The conversion and the connectivity rule were checked against Table B.1, which
lists the maximum particle count and the approximate maximum edge count of every
domain. Edge counts include self edges, as in the reference implementation.

| domain | max particles, paper | measured | max edges, paper | measured |
| --- | --- | --- | --- | --- |
| Goop | 1.9k | 1970 | 19k | 18.0k |
| WaterRamps | 2.3k | 2376 | 26k | 22.4k |
| SandRamps | 3.3k | 3435 | 32k | 29.2k |

The acceleration statistics measured on the test split also agree with the
released training statistics to within 10-20%, and a constant-velocity predictor
scores exactly the mean squared acceleration under our one-step metric. Together
those pin the data, the connectivity radius and the metric definition.

## Model

Faithful to `learned_simulator.py` and `graph_network.py`:

- C = 5 input velocities from 6 input positions; velocity is a position
  difference with dt folded away.
- Node features: 5 normalized velocities, `2 * dim` wall distances divided by the
  connectivity radius and clipped to `[-1, 1]`, and a 16-dimensional learned
  particle-type embedding. 30 features in 2D.
- Edge features: the relative displacement divided by the connectivity radius,
  and its norm. 3 features in 2D.
- Encoder, 10 unshared interaction networks with node and edge residuals, and a
  decoder. All MLPs have 2 hidden layers of 128 with ReLU and a linear output of
  128; every MLP except the decoder is followed by LayerNorm. 1,591,826
  parameters.
- Message aggregation is a sum over incoming edges.
- Semi-implicit Euler with dt = 1: `v' = v + a`, `p' = p + v'`.

Two details the paper leaves open:

- **The absolute encoder** of Figure 4(i,j) is not specified. Here it appends the
  particle's own position to its node features and replaces the edge
  displacement with both endpoint positions. The relative encoder is unchanged
  and is the default everywhere else.
- **Self edges** are included, following the reference implementation's default.
  Figure C.2(k,l) reports this choice makes little difference.

## Training

Faithful to `train.py`:

- Adam, learning rate `1e-6 + (1e-4 - 1e-6) * 0.1 ** (step / 5e6)`.
- Random-walk noise on the input velocities, `noise_std = 6.7e-4` at the last of
  the five input steps. The paper's `sigma_v = 3e-4` is the same quantity per
  step, `sqrt(5)` smaller.
- Velocity and acceleration normalization statistics are combined with the noise
  scale in quadrature.
- Obstacle particles get no noise and are excluded from the loss.
- The loss is the squared error on the normalized acceleration, summed over
  spatial axes and divided by the number of non-obstacle particles.
- The target acceleration is computed from the clean next position shifted by the
  noise on the last input position, so the model learns to undo the noise in the
  input velocity but not the noise in the input position.
- Trajectories are streamed in order, cut into seven-frame windows and shuffled
  through a 10,000-window buffer.
- Checkpoints are selected on the rollout MSE of five held-out validation
  trajectories.

### Deliberate deviations

1. **No fixed-size padding.** The paper pads every batch to the largest graph in
   the dataset because TPU cores need static shapes, then packs 1-3 windows into
   the slack, giving an effective batch of 2-6. It describes this as equivalent
   to a batch size measured in particles. We use the nominal batch of two windows
   with no padding: on this GPU the padded slots were about 45% of the work, and
   the model is bandwidth bound.
2. **A shorter budget with a proportionally shorter decay.** The paper runs up to
   20M gradient steps and decays the learning rate by a decade every 5M. A run
   here uses fewer steps and keeps the same 4:1 ratio, so the rate still reaches
   its floor as training ends. Without that, a short run would stop while the
   rate was still near its starting value. Each run's `config.json` records the
   horizon actually used, and the reported step count is stated with every
   number.
3. **Frozen normalization statistics.** The paper accumulates them online during
   training; the released code reads them from `metadata.json`. We use the file,
   which is what the released models were trained with and is reproducible.

## Metrics

Section 4.4 defines both metrics as particle-wise MSE averaged over time,
particles and spatial axes. The released code averages over *every* particle.
That is unambiguous on Goop, which has no obstacle particles, and ambiguous on
WaterRamps and SandRamps, where obstacles are a large fraction of the scene:

- In a **rollout** the obstacles are overwritten with the ground truth at every
  step, so they contribute exactly zero and dilute the average.
- In a **one-step** evaluation they are not overwritten, so their prediction
  error is included.

Every result file therefore carries both `*_free_particles` (obstacles excluded)
and `*_all_particles` (the released code's convention), and the figures take a
`--convention` flag. Goop is the dataset to compare against the paper without
caveats.

Rollouts start from the first six ground-truth frames and run
`sequence_length - 6` steps: 394 on Goop and SandRamps, 594 on WaterRamps.

## Results

See `figures/table1.md` for the current numbers and the step count they were
measured at, and `docs/RESULTS.md` for the discussion.

## Things that were tried and did not help

- **A more aggressive peak learning rate.** Section 4.3 says the models "can
  train in significantly less steps" and that the conservative rate exists to
  make comparisons across settings fair, which suggests raising it is the way to
  fit a smaller budget. Probing 3e-4 and 1e-3 against the paper's 1e-4 on Goop,
  all with the same decay horizon, the training loss at 30k steps was 0.179 and
  0.204 against about 0.175. Neither helped, so every run here uses the paper's
  1e-4. `--lr-start` is kept so the probe is repeatable.
- **CUDA graph capture.** The step issues several hundred small kernels, so
  launch overhead looked like the obvious target. Capturing the whole step
  changed the rate by less than 2%: the model is bandwidth bound, not launch
  bound. `torch.compile(dynamic=True)` does help, by about 20%, through fusion.
- **Splitting the edge update's first layer across its three inputs.** Projecting
  the node latents before gathering is the same function with a third of the
  arithmetic, but the extra gathers made it 40% slower than one concatenation and
  one matrix multiply.
- **A noise-free sanity run.** With `--noise-std 0` the normalized acceleration
  target reaches 71 standard deviations on Goop and training does not converge at
  all. Inflating the normalization by the noise scale compresses that to 17, so
  the paper's noise is not only a robustness device: it also makes the regression
  target well conditioned. A run with noise disabled is not a valid diagnostic.
