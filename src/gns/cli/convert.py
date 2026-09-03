"""Convert the released TFRecord datasets to the HDF5 layout used here.

    python -m gns.cli.convert --raw <raw>/WaterRamps --out <data>/WaterRamps

Writes ``<out>/{train,valid,test}/sim_<i>.h5``, copies ``metadata.json`` and
measures ``shapes.json``: the largest graph in the training split, which is the
fixed tensor budget the training loop pads every batch to.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from gns import KINEMATIC_PARTICLE_ID
from gns.data.tfrecord import read_trajectories
from gns.metadata import Metadata
from gns.neighbors import radius_graph_numpy


def write_split(raw: Path, out: Path, split: str, dim: int) -> list[int]:
    """Write one split and return the particle count of every trajectory."""
    destination = out / split
    destination.mkdir(parents=True, exist_ok=True)
    counts = []
    stream = read_trajectories(raw / f"{split}.tfrecord", dim)
    for index, (positions, types) in enumerate(tqdm(stream, desc=f"{out.name}/{split}")):
        obstacle = types == KINEMATIC_PARTICLE_ID
        # Obstacles first, matching the layout the NeuralMPM parsers expect.
        order = np.concatenate([np.flatnonzero(obstacle), np.flatnonzero(~obstacle)])
        positions = positions[:, order]
        types = types[order]
        num_obstacles = int(obstacle.sum())
        with h5py.File(destination / f"sim_{index}.h5", "w") as handle:
            handle.create_dataset("boundary", data=positions[:, :num_obstacles])
            handle.create_dataset("particles", data=positions[:, num_obstacles:])
            handle.create_dataset("types", data=types)
        counts.append(positions.shape[1])
    return counts


def measure_edges(
    out: Path,
    radius: float,
    dim: int,
    num_sims: int,
    num_frames: int,
    split: str = "train",
    seed: int = 0,
) -> int:
    """Largest edge count seen in a sample of training frames.

    Building the graph for all 600k training frames would cost more than it is
    worth; the budget only has to bound a single example, and the batch budget
    is a multiple of it.
    """
    from gns.data.trajectories import TrajectoryStore

    store = TrajectoryStore(out, split, dim)
    rng = np.random.default_rng(seed)
    sims = rng.choice(len(store), size=min(num_sims, len(store)), replace=False)
    largest = 0
    for index in tqdm(sims, desc=f"{out.name}/edges"):
        trajectory = store[int(index)]
        frames = rng.choice(trajectory.num_frames, size=num_frames, replace=False)
        for frame in frames:
            senders, _ = radius_graph_numpy(trajectory.positions[frame], radius)
            largest = max(largest, len(senders))
    return largest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--edge-sample-sims", type=int, default=40)
    parser.add_argument("--edge-sample-frames", type=int, default=25)
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=1.2,
        help="Safety factor on the sampled maximum edge count.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.raw / "metadata.json", args.out / "metadata.json")
    metadata = Metadata.load(args.out)

    counts: dict[str, list[int]] = {}
    for split in args.splits:
        counts[split] = write_split(args.raw, args.out, split, metadata.dim)

    budget_split = "train" if "train" in counts else args.splits[0]
    max_edges = measure_edges(
        args.out,
        metadata.connectivity_radius,
        metadata.dim,
        args.edge_sample_sims,
        args.edge_sample_frames,
        split=budget_split,
    )
    shapes = {
        "budget_split": budget_split,
        "max_nodes": int(max(counts[budget_split])),
        "max_edges": int(np.ceil(max_edges * args.edge_margin)),
        "sampled_max_edges": int(max_edges),
        "num_sims": {split: len(values) for split, values in counts.items()},
        "max_nodes_per_split": {
            split: int(max(values)) for split, values in counts.items()
        },
    }
    (args.out / "shapes.json").write_text(json.dumps(shapes, indent=2) + "\n")
    print(json.dumps(shapes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
