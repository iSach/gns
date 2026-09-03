"""One-step training examples.

The paper streams trajectories sequentially, cuts them into seven-frame windows
(six inputs and one target) and samples from a 10k-window shuffle buffer
(Supplementary B.3).  A batch is the nominal mini-batch of two windows,
concatenated into one disconnected graph.

The paper additionally pads every batch to the size of the dataset's largest
graph, because its TPU cores need fixed-size tensors, and packs extra windows
into the slack.  It calls that "equivalent to setting a mini batch size in terms
of total number of particles per batch".  We keep the nominal batch of two and
skip the padding: on a GPU the padded slots are wasted bandwidth, and this model
is bandwidth bound.

Noise is drawn and the graph is built here, in the loader's worker processes.
The paper adds noise to the input positions and rebuilds the graph on the noisy
positions, and the k-d tree is a CPU algorithm anyway.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from gns import INPUT_SEQUENCE_LENGTH, KINEMATIC_PARTICLE_ID
from gns.data.trajectories import TrajectoryStore
from gns.neighbors import radius_graph_numpy
from gns.noise import random_walk_noise


@dataclass
class Example:
    """One training window after noise and graph construction."""

    positions: np.ndarray  # [N, 6, dim], noisy inputs
    target: np.ndarray  # [N, dim], clean next position
    target_noise: np.ndarray  # [N, dim], noise on the last input position
    particle_types: np.ndarray  # [N]
    senders: np.ndarray  # [E]
    receivers: np.ndarray  # [E]


@dataclass
class Batch:
    """Several windows concatenated into one disconnected graph."""

    positions: torch.Tensor  # [N, 6, dim]
    target: torch.Tensor  # [N, dim]
    target_noise: torch.Tensor  # [N, dim]
    particle_types: torch.Tensor  # [N]
    senders: torch.Tensor  # [E]
    receivers: torch.Tensor  # [E]
    loss_mask: torch.Tensor  # [N], true for particles the model must predict
    num_examples: int

    def _map(self, fn) -> "Batch":
        return Batch(
            positions=fn(self.positions),
            target=fn(self.target),
            target_noise=fn(self.target_noise),
            particle_types=fn(self.particle_types),
            senders=fn(self.senders),
            receivers=fn(self.receivers),
            loss_mask=fn(self.loss_mask),
            num_examples=self.num_examples,
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return self._map(lambda t: t.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> "Batch":
        return self._map(lambda t: t.pin_memory())


def collate(examples: list[Example]) -> Batch:
    """Concatenate examples, shifting each one's edge indices into place."""
    offsets = np.cumsum([0] + [e.positions.shape[0] for e in examples[:-1]])
    types = np.concatenate([e.particle_types for e in examples])
    return Batch(
        positions=torch.from_numpy(
            np.concatenate([e.positions for e in examples])
        ),
        target=torch.from_numpy(np.concatenate([e.target for e in examples])),
        target_noise=torch.from_numpy(
            np.concatenate([e.target_noise for e in examples])
        ),
        particle_types=torch.from_numpy(types),
        senders=torch.from_numpy(
            np.concatenate([e.senders + o for e, o in zip(examples, offsets)])
        ),
        receivers=torch.from_numpy(
            np.concatenate([e.receivers + o for e, o in zip(examples, offsets)])
        ),
        loss_mask=torch.from_numpy(types != KINEMATIC_PARTICLE_ID),
        num_examples=len(examples),
    )


class OneStepDataset(IterableDataset):
    """Streams one-step training batches from a split.

    Args:
        path: dataset directory holding ``train/``, ``valid/`` and ``test/``.
        split: which split to stream.
        radius: connectivity radius used to build the graph.
        noise_std: random-walk noise scale at the last input step; 0 disables it.
        batch_size: windows per batch; the paper's nominal value is 2.
        shuffle_buffer: number of windows held across all workers.
        seed: base seed; each worker derives its own stream from it.
    """

    def __init__(
        self,
        path: str | Path,
        split: str,
        radius: float,
        noise_std: float,
        batch_size: int = 2,
        dim: int = 2,
        shuffle_buffer: int = 10_000,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.split = split
        self.radius = radius
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.shuffle = shuffle

    def _make_example(
        self,
        window: np.ndarray,
        particle_types: np.ndarray,
        rng: np.random.Generator,
    ) -> Example:
        inputs = window[:INPUT_SEQUENCE_LENGTH]
        target = window[INPUT_SEQUENCE_LENGTH]
        # [T, N, dim] -> [N, T, dim]: the model is indexed by particle.
        inputs = np.ascontiguousarray(inputs.transpose(1, 0, 2))

        if self.noise_std > 0.0:
            noise = random_walk_noise(inputs, self.noise_std, rng)
            # Obstacles do not move, so perturbing them would teach the model a
            # motion that never happens at test time.
            noise[particle_types == KINEMATIC_PARTICLE_ID] = 0.0
            inputs = inputs + noise
            target_noise = noise[:, -1]
        else:
            target_noise = np.zeros_like(target)

        senders, receivers = radius_graph_numpy(inputs[:, -1], self.radius)
        return Example(
            positions=inputs.astype(np.float32),
            target=target.astype(np.float32),
            target_noise=target_noise.astype(np.float32),
            particle_types=particle_types,
            senders=senders,
            receivers=receivers,
        )

    def _windows(self, store, order, rng, shuffler):
        """Yield examples, shuffled through a reservoir the paper's size."""
        buffer: list[Example] = []
        capacity = max(1, self.shuffle_buffer)
        while True:
            if self.shuffle:
                shuffler.shuffle(order)
            for index in order:
                trajectory = store[index]
                types = trajectory.particle_types
                for start in range(trajectory.num_frames - INPUT_SEQUENCE_LENGTH):
                    window = trajectory.positions[
                        start : start + INPUT_SEQUENCE_LENGTH + 1
                    ]
                    example = self._make_example(window, types, rng)
                    if not self.shuffle:
                        yield example
                        continue
                    if len(buffer) < capacity:
                        buffer.append(example)
                        continue
                    slot = shuffler.randrange(capacity)
                    buffer[slot], example = example, buffer[slot]
                    yield example
            if not self.shuffle:
                break
        yield from buffer

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers

        store = TrajectoryStore(self.path, self.split, self.dim)
        rng = np.random.default_rng(self.seed + 7919 * worker_id)
        shuffler = random.Random(self.seed + 104729 * worker_id)
        order = list(range(worker_id, len(store), num_workers))
        # The buffer is per worker, so split the paper's total across them.
        self.shuffle_buffer = max(1, self.shuffle_buffer // num_workers)

        pending: list[Example] = []
        for example in self._windows(store, order, rng, shuffler):
            pending.append(example)
            if len(pending) == self.batch_size:
                yield collate(pending)
                pending = []
        if pending:
            yield collate(pending)
