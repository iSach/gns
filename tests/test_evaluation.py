"""Rollout mechanics and the two metric conventions."""

import numpy as np
import pytest
import torch

from gns import INPUT_SEQUENCE_LENGTH, KINEMATIC_PARTICLE_ID
from gns.data.trajectories import Trajectory
from gns.evaluation.metrics import RolloutResult, rollout
from gns.models.simulator import LearnedSimulator, SimulatorConfig, kinematic_mask
from tests.test_model import toy_metadata


def toy_trajectory(num_frames=12, num_particles=40, num_obstacles=8, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.2, 0.8, size=(1, num_particles, 2))
    drift = np.cumsum(
        rng.normal(scale=1e-3, size=(num_frames, num_particles, 2)), axis=0
    )
    positions = (start + drift).astype(np.float32)
    positions[:, :num_obstacles] = positions[0, :num_obstacles]  # obstacles are static
    types = np.full(num_particles, 5, dtype=np.int64)
    types[:num_obstacles] = KINEMATIC_PARTICLE_ID
    return Trajectory(positions=positions, particle_types=types)


def test_rollout_shapes_and_obstacle_handling():
    """Obstacles must be copied from the ground truth at every rollout step."""
    model = LearnedSimulator(toy_metadata(), SimulatorConfig(num_message_passing_steps=2))
    model.eval()
    trajectory = toy_trajectory()
    steps = trajectory.num_frames - INPUT_SEQUENCE_LENGTH
    result = rollout(model, trajectory, steps, torch.device("cpu"))

    assert result.predicted.shape == (steps, trajectory.num_particles, 2)
    assert result.ground_truth.shape == result.predicted.shape
    obstacles = kinematic_mask(torch.from_numpy(trajectory.particle_types)).numpy()
    np.testing.assert_allclose(
        result.predicted[:, obstacles], result.ground_truth[:, obstacles]
    )
    # Free particles are predicted, so they will not match by construction.
    assert not np.allclose(
        result.predicted[:, ~obstacles], result.ground_truth[:, ~obstacles]
    )


def test_metric_conventions_differ_only_by_the_obstacles():
    """The released convention dilutes a rollout with exactly-zero obstacle error."""
    trajectory = toy_trajectory()
    steps = 4
    predicted = np.zeros((steps, trajectory.num_particles, 2), dtype=np.float32)
    ground_truth = np.ones_like(predicted)
    obstacles = trajectory.particle_types == KINEMATIC_PARTICLE_ID
    predicted[:, obstacles] = 1.0  # obstacles are copied, so their error is zero

    result = RolloutResult(
        predicted=predicted,
        ground_truth=ground_truth,
        initial=trajectory.positions[:INPUT_SEQUENCE_LENGTH],
        particle_types=trajectory.particle_types,
    )
    free = float((~obstacles).sum())
    total = float(len(obstacles))
    assert result.mse(include_obstacles=False) == pytest.approx(1.0)
    assert result.mse(include_obstacles=True) == pytest.approx(free / total)


def test_per_step_curve_matches_the_average():
    trajectory = toy_trajectory()
    rng = np.random.default_rng(1)
    steps = 5
    predicted = rng.normal(size=(steps, trajectory.num_particles, 2)).astype(np.float32)
    result = RolloutResult(
        predicted=predicted,
        ground_truth=np.zeros_like(predicted),
        initial=trajectory.positions[:INPUT_SEQUENCE_LENGTH],
        particle_types=trajectory.particle_types,
    )
    assert result.per_step_mse.shape == (steps,)
    assert result.per_step_mse.mean() == pytest.approx(
        result.mse(include_obstacles=False), rel=1e-5
    )
