"""Reading trajectories from the HDF5 layout shared with the NeuralMPM repo.

One file per simulation, ``<dataset>/<split>/sim_<i>.h5``, holding

    particles  [T, N, 2 * dim]   free particles: positions then velocities
    boundary   [T, M, 2 * dim]   obstacle particles: positions then normals
    types      [M + N]           released particle-type ids, obstacles first

Only the positions are read.  Velocities and accelerations are finite
differences of the positions, exactly as the paper defines them (B.2), so
storing them would only add a chance for the two to disagree.  Obstacle
particles come first so a dataset without any (Goop, Water, Sand, WaterDrop)
simply has ``M = 0`` and needs no special case anywhere downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """One simulation: positions over time plus a static type per particle."""

    positions: np.ndarray  # [T, N, dim], float32
    particle_types: np.ndarray  # [N], int64

    @property
    def num_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def num_particles(self) -> int:
        return self.positions.shape[1]


class TrajectoryStore:
    """Random access to the simulations of one split."""

    def __init__(self, path: str | Path, split: str, dim: int = 2) -> None:
        self.path = Path(path)
        self.split = split
        self.dim = dim
        self._counts: np.ndarray | None = None
        self.files = sorted(
            (self.path / split).glob("sim_*.h5"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not self.files:
            raise FileNotFoundError(f"No sim_*.h5 under {self.path / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Trajectory:
        with h5py.File(self.files[index], "r") as handle:
            boundary = handle["boundary"][:, :, : self.dim]
            particles = handle["particles"][:, :, : self.dim]
            positions = np.concatenate([boundary, particles], axis=1)
            types = handle["types"][()].astype(np.int64)
        return Trajectory(
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            particle_types=types,
        )

    def particle_counts(self) -> np.ndarray:
        """Particle count of every simulation, read from the headers only."""
        if self._counts is not None:
            return self._counts
        counts = []
        for file in self.files:
            with h5py.File(file, "r") as handle:
                counts.append(
                    handle["boundary"].shape[1] + handle["particles"].shape[1]
                )
        self._counts = np.asarray(counts, dtype=np.int64)
        return self._counts
