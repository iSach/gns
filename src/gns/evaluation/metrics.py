"""The two metrics the paper reports, plus the rollout that produces one of them.

Both are particle-wise mean squared errors averaged over time, particles and
spatial axes (Section 4.4).  The reference implementation averages over *every*
particle, obstacles included; obstacles are copied from the ground truth during
a rollout so they contribute exactly zero and simply dilute the mean, and in a
one-step evaluation their predicted motion is not overwritten so they contribute
real error.  Both conventions are computed and reported side by side: on
datasets with obstacles they differ by more than a factor of two, and the paper
does not say which one Table 1 uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gns import INPUT_SEQUENCE_LENGTH
from gns.data.trajectories import Trajectory, TrajectoryStore
from gns.models.simulator import LearnedSimulator, kinematic_mask
from gns.neighbors import radius_graph_torch


@dataclass
class RolloutResult:
    """A full rollout and its error against the ground truth."""

    predicted: np.ndarray  # [num_steps, N, dim]
    ground_truth: np.ndarray  # [num_steps, N, dim]
    initial: np.ndarray  # [INPUT_SEQUENCE_LENGTH, N, dim]
    particle_types: np.ndarray  # [N]

    @property
    def per_step_mse(self) -> np.ndarray:
        """MSE at each rollout step over non-obstacle particles."""
        free = self.particle_types != 3
        squared = (self.predicted[:, free] - self.ground_truth[:, free]) ** 2
        return squared.mean(axis=(1, 2))

    def mse(self, include_obstacles: bool) -> float:
        if include_obstacles:
            return float(((self.predicted - self.ground_truth) ** 2).mean())
        free = self.particle_types != 3
        return float(((self.predicted[:, free] - self.ground_truth[:, free]) ** 2).mean())


@dataclass
class RolloutMetrics:
    """Rollout error averaged over a set of trajectories."""

    mse_all_particles: float
    mse_free_particles: float
    per_step_mse: np.ndarray
    num_trajectories: int


@dataclass
class OneStepResult:
    """One-step error averaged over every window of a set of trajectories."""

    mse_all_particles: float
    mse_free_particles: float
    num_windows: int


@torch.no_grad()
def rollout(
    model: LearnedSimulator,
    trajectory: Trajectory,
    num_steps: int,
    device: torch.device,
) -> RolloutResult:
    """Roll the model out from the first six ground-truth frames.

    Obstacle particles are reset to their prescribed positions at every step, as
    in the reference implementation: their motion is an input to the simulation,
    not something the model is asked to predict.
    """
    positions = torch.as_tensor(trajectory.positions, device=device)
    types = torch.as_tensor(trajectory.particle_types, device=device)
    obstacles = kinematic_mask(types).unsqueeze(-1)

    current = positions[:INPUT_SEQUENCE_LENGTH].permute(1, 0, 2).contiguous()
    ground_truth = positions[
        INPUT_SEQUENCE_LENGTH : INPUT_SEQUENCE_LENGTH + num_steps
    ]

    predictions = []
    for step in range(num_steps):
        senders, receivers = radius_graph_torch(current[:, -1], model.radius)
        nxt = model(current, types, senders, receivers)
        nxt = torch.where(obstacles, ground_truth[step], nxt)
        predictions.append(nxt)
        current = torch.cat([current[:, 1:], nxt.unsqueeze(1)], dim=1)

    return RolloutResult(
        predicted=torch.stack(predictions).cpu().numpy(),
        ground_truth=ground_truth.cpu().numpy(),
        initial=positions[:INPUT_SEQUENCE_LENGTH].cpu().numpy(),
        particle_types=trajectory.particle_types,
    )


def rollout_metrics(
    model: LearnedSimulator,
    store: TrajectoryStore,
    num_steps: int,
    device: torch.device,
    limit: int | None = None,
    keep: list[RolloutResult] | None = None,
) -> RolloutMetrics:
    """Average rollout error over the first ``limit`` trajectories of a split."""
    count = len(store) if limit is None else min(limit, len(store))
    all_particles, free_particles, curves = [], [], []
    for index in range(count):
        result = rollout(model, store[index], num_steps, device)
        all_particles.append(result.mse(include_obstacles=True))
        free_particles.append(result.mse(include_obstacles=False))
        curves.append(result.per_step_mse)
        if keep is not None:
            keep.append(result)
    return RolloutMetrics(
        mse_all_particles=float(np.mean(all_particles)),
        mse_free_particles=float(np.mean(free_particles)),
        per_step_mse=np.mean(np.stack(curves), axis=0),
        num_trajectories=count,
    )


@torch.no_grad()
def one_step_metrics(
    model: LearnedSimulator,
    store: TrajectoryStore,
    device: torch.device,
    limit: int | None = None,
    stride: int = 1,
) -> OneStepResult:
    """Predicted-versus-true next position over every window of a split.

    Nothing is overwritten here: the model predicts obstacle particles too, and
    the ``all_particles`` figure includes that error.
    """
    count = len(store) if limit is None else min(limit, len(store))
    total_all = total_free = 0.0
    count_all = count_free = 0
    windows = 0
    for index in range(count):
        trajectory = store[index]
        positions = torch.as_tensor(trajectory.positions, device=device)
        types = torch.as_tensor(trajectory.particle_types, device=device)
        free = ~kinematic_mask(types)
        num_windows = trajectory.num_frames - INPUT_SEQUENCE_LENGTH
        for start in range(0, num_windows, stride):
            inputs = positions[start : start + INPUT_SEQUENCE_LENGTH]
            inputs = inputs.permute(1, 0, 2).contiguous()
            target = positions[start + INPUT_SEQUENCE_LENGTH]
            senders, receivers = radius_graph_torch(inputs[:, -1], model.radius)
            predicted = model(inputs, types, senders, receivers)
            squared = (predicted - target) ** 2
            total_all += float(squared.sum())
            count_all += squared.numel()
            total_free += float(squared[free].sum())
            count_free += int(squared[free].numel())
            windows += 1
    return OneStepResult(
        mse_all_particles=total_all / max(count_all, 1),
        mse_free_particles=total_free / max(count_free, 1),
        num_windows=windows,
    )
