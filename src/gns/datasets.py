"""Registry of the released GNS datasets.

Adding a dataset means adding an entry here, not a new trainer.  ``max_nodes``
and ``max_edges`` are the fixed tensor sizes the training loop pads every batch
to; the paper does the same on TPU (Supplementary B.3) and packs as many
examples as fit inside that budget.  They are measured from the data by
``gns.cli.convert`` and stored in the dataset's ``shapes.json``, so the values
below are only fallbacks for a dataset that has not been converted yet.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetEntry:
    """Static description of one dataset."""

    name: str
    dim: int
    sequence_length: int
    material_type: int  # released particle-type id of the simulated material
    has_boundary_particles: bool
    splits: dict[str, int]

    @property
    def path(self) -> Path:
        return data_root() / self.name


def data_root() -> Path:
    """Root holding ``<name>/{train,valid,test}/sim_*.h5`` for every dataset."""
    root = os.environ.get("GNS_DATA_ROOT")
    if root:
        return Path(root)
    raise RuntimeError(
        "Set GNS_DATA_ROOT to the directory holding the converted datasets, "
        "or pass --data-path explicitly."
    )


# Particle-type ids used by the released datasets (see render_rollout.py in the
# reference implementation): 0 rigid, 3 boundary, 5 water, 6 sand, 7 goop.
_REGISTRY: dict[str, DatasetEntry] = {
    entry.name: entry
    for entry in [
        DatasetEntry("WaterRamps", 2, 600, 5, True, {"train": 1000, "valid": 100, "test": 100}),
        DatasetEntry("SandRamps", 2, 400, 6, True, {"train": 1000, "valid": 100, "test": 100}),
        DatasetEntry("Goop", 2, 400, 7, False, {"train": 1000, "valid": 30, "test": 30}),
        DatasetEntry("Water", 2, 1000, 5, False, {"train": 1000, "valid": 30, "test": 30}),
        DatasetEntry("Sand", 2, 320, 6, False, {"train": 1000, "valid": 30, "test": 30}),
        DatasetEntry("WaterDrop", 2, 1000, 5, False, {"train": 1000, "valid": 30, "test": 30}),
    ]
}


def names() -> list[str]:
    return list(_REGISTRY)


def get(name: str) -> DatasetEntry:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Known: {', '.join(names())}")
    return _REGISTRY[name]


def shapes(path: str | Path) -> dict[str, int]:
    """Read the padded tensor budget measured at conversion time."""
    path = Path(path)
    if path.is_dir():
        path = path / "shapes.json"
    return json.loads(path.read_text())
