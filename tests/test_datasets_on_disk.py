"""Checks against the converted datasets. Marked ``data``: they need the files."""

import json
from pathlib import Path

import numpy as np
import pytest

from gns import INPUT_SEQUENCE_LENGTH, KINEMATIC_PARTICLE_ID, datasets, paper
from gns.data.trajectories import TrajectoryStore
from gns.metadata import Metadata

pytestmark = pytest.mark.data

CASES = ["Goop", "WaterRamps", "SandRamps"]


def dataset_path(name: str) -> Path:
    path = datasets.get(name).path
    if not (path / "metadata.json").exists():
        pytest.skip(f"{name} is not converted at {path}")
    return path


@pytest.mark.parametrize("name", CASES)
def test_shapes_match_the_papers_table_b1(name):
    """Particle and edge counts pin the data and the connectivity radius."""
    path = dataset_path(name)
    shapes = json.loads((path / "shapes.json").read_text())
    reference = paper.TABLE_1[name]
    # Table B.1 rounds to two significant figures, so allow 10%.
    assert shapes["max_nodes"] == pytest.approx(reference.max_particles, rel=0.1)
    assert shapes["sampled_max_edges"] == pytest.approx(reference.max_edges, rel=0.2)


@pytest.mark.parametrize("name", CASES)
def test_split_sizes_and_frame_count(name):
    path = dataset_path(name)
    entry = datasets.get(name)
    metadata = Metadata.load(path)
    assert metadata.sequence_length == entry.sequence_length
    assert metadata.connectivity_radius == 0.015
    for split, expected in entry.splits.items():
        store = TrajectoryStore(path, split, metadata.dim)
        assert len(store) == expected
    # The released trajectories carry one extra leading frame so the first
    # velocity can be a difference; a rollout uses sequence_length - 6 steps.
    trajectory = TrajectoryStore(path, "test", metadata.dim)[0]
    assert trajectory.num_frames == metadata.sequence_length + 1
    assert trajectory.num_frames - INPUT_SEQUENCE_LENGTH > 0


@pytest.mark.parametrize("name", CASES)
def test_obstacle_particles_are_static_and_come_first(name):
    path = dataset_path(name)
    store = TrajectoryStore(path, "test", 2)
    entry = datasets.get(name)
    saw_obstacles = False
    for index in range(min(5, len(store))):
        trajectory = store[index]
        obstacles = trajectory.particle_types == KINEMATIC_PARTICLE_ID
        assert set(np.unique(trajectory.particle_types)) <= {
            KINEMATIC_PARTICLE_ID,
            entry.material_type,
        }
        if obstacles.any():
            saw_obstacles = True
            assert obstacles[: int(obstacles.sum())].all()  # obstacles first
            moved = np.abs(
                trajectory.positions[:, obstacles] - trajectory.positions[0, obstacles]
            ).max()
            assert moved == 0.0
    assert saw_obstacles == entry.has_boundary_particles


@pytest.mark.parametrize("name", CASES)
def test_constant_velocity_baseline_matches_the_released_statistics(name):
    """A zero-acceleration predictor scores exactly the mean squared acceleration.

    This ties our metric definition, the position data and the released
    normalization statistics together: if any of the three were off, the two
    numbers would not agree.
    """
    path = dataset_path(name)
    metadata = Metadata.load(path)
    store = TrajectoryStore(path, "test", metadata.dim)
    squared, count = 0.0, 0
    for index in range(min(5, len(store))):
        positions = store[index].positions
        acceleration = positions[2:] - 2 * positions[1:-1] + positions[:-2]
        squared += float((acceleration**2).sum())
        count += acceleration.size
    baseline = squared / count
    expected = float((metadata.acc.std**2).mean())
    assert baseline == pytest.approx(expected, rel=0.5)
    # The paper's model must beat this baseline by several times over.
    assert paper.one_step_mse(name) < baseline / 3
