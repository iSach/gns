"""Dataset metadata: bounds, connectivity radius and normalization statistics.

The released datasets ship a ``metadata.json`` holding the exact statistics the
paper's models were trained with.  Section B.3 of the paper accumulates those
statistics online during training; the released code freezes them into this file
instead, and reproducing the published numbers means using the frozen values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Stats:
    """Elementwise mean and standard deviation of a vector quantity."""

    mean: np.ndarray
    std: np.ndarray

    def combine_std(self, extra_std: float) -> "Stats":
        """Widen the std to account for noise injected at training time.

        The released code inflates both the velocity and the acceleration std by
        the training noise scale, so the normalized inputs stay unit-variance
        once noise is added.
        """
        return Stats(self.mean, np.sqrt(self.std**2 + extra_std**2))


@dataclass(frozen=True)
class Metadata:
    """Everything about a dataset that the model needs but cannot infer."""

    bounds: np.ndarray  # [dim, 2], lower and upper wall position per axis
    sequence_length: int  # number of frames the paper reports rollouts over
    connectivity_radius: float
    dim: int
    dt: float
    vel: Stats
    acc: Stats

    @staticmethod
    def load(path: str | Path) -> "Metadata":
        path = Path(path)
        if path.is_dir():
            path = path / "metadata.json"
        raw = json.loads(path.read_text())
        f32 = lambda key: np.asarray(raw[key], dtype=np.float32)  # noqa: E731
        return Metadata(
            bounds=np.asarray(raw["bounds"], dtype=np.float32),
            sequence_length=int(raw["sequence_length"]),
            connectivity_radius=float(raw["default_connectivity_radius"]),
            dim=int(raw["dim"]),
            dt=float(raw["dt"]),
            vel=Stats(f32("vel_mean"), f32("vel_std")),
            acc=Stats(f32("acc_mean"), f32("acc_std")),
        )

    def normalization(self, noise_std: float) -> dict[str, Stats]:
        """Return the velocity/acceleration statistics used by the model."""
        return {
            "velocity": self.vel.combine_std(noise_std),
            "acceleration": self.acc.combine_std(noise_std),
        }
