"""Windowing, noise and batching."""

import numpy as np
import pytest
import torch

from gns import INPUT_SEQUENCE_LENGTH, KINEMATIC_PARTICLE_ID
from gns.data.onestep import Example, collate
from gns.noise import DEFAULT_NOISE_STD, per_step_std, random_walk_noise


def test_random_walk_reaches_the_requested_last_step_std():
    """The scale is defined at the last input step, not per step."""
    rng = np.random.default_rng(0)
    positions = np.zeros((40_000, INPUT_SEQUENCE_LENGTH, 2), dtype=np.float32)
    noise = random_walk_noise(positions, DEFAULT_NOISE_STD, rng)
    velocity_noise = noise[:, 1:] - noise[:, :-1]
    assert noise[:, 0].std() == 0.0  # the first position is never perturbed
    assert velocity_noise[:, -1].std() == pytest.approx(DEFAULT_NOISE_STD, rel=0.03)
    # Earlier steps are smaller by sqrt(k + 1), which is what makes it a walk.
    ratio = velocity_noise[:, 0].std() / velocity_noise[:, -1].std()
    assert ratio == pytest.approx(1 / np.sqrt(5), rel=0.05)


def test_per_step_std_matches_the_paper_value():
    """6.7e-4 at the last of five steps is the paper's sigma_v = 3e-4."""
    assert per_step_std(DEFAULT_NOISE_STD, 5) == pytest.approx(3e-4, rel=0.01)


def _example(num_particles, num_edges, seed, kinematic=0):
    rng = np.random.default_rng(seed)
    types = np.full(num_particles, 5, dtype=np.int64)
    types[:kinematic] = KINEMATIC_PARTICLE_ID
    return Example(
        positions=rng.random((num_particles, INPUT_SEQUENCE_LENGTH, 2), dtype=np.float32),
        target=rng.random((num_particles, 2), dtype=np.float32),
        target_noise=np.zeros((num_particles, 2), dtype=np.float32),
        particle_types=types,
        senders=rng.integers(0, num_particles, num_edges),
        receivers=rng.integers(0, num_particles, num_edges),
    )


def test_collate_offsets_edges_into_the_second_graph():
    """Concatenating graphs must not create edges between them."""
    a = _example(30, 50, seed=0)
    b = _example(20, 40, seed=1, kinematic=5)
    batch = collate([a, b])

    assert batch.positions.shape[0] == 50
    assert batch.senders.shape[0] == 90
    assert batch.num_examples == 2
    # The second graph's indices are shifted by the first graph's size, so every
    # edge stays inside the graph it came from.
    torch.testing.assert_close(batch.senders[:50], torch.from_numpy(a.senders))
    torch.testing.assert_close(batch.senders[50:], torch.from_numpy(b.senders + 30))
    assert batch.senders.max() < batch.positions.shape[0]


def test_collate_masks_obstacles_out_of_the_loss():
    batch = collate([_example(30, 50, seed=0), _example(20, 40, seed=1, kinematic=5)])
    assert batch.loss_mask.sum() == 45
    assert not batch.loss_mask[30:35].any()
