"""Render rollouts of a trained checkpoint.

    gns-render --checkpoint runs/goop/best.pt --trajectory 0 --video
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gns import INPUT_SEQUENCE_LENGTH, datasets
from gns.cli.evaluate import load_model
from gns.data.trajectories import TrajectoryStore
from gns.evaluation.metrics import rollout
from gns.metadata import Metadata
from gns.render.frames import render_comparison_strip, save_rollout_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-path", default=None, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--frames", type=int, nargs="+", default=None)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    name = args.dataset or state["dataset"]
    data_path = args.data_path or datasets.get(name).path
    metadata = Metadata.load(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, metadata, device)
    store = TrajectoryStore(data_path, args.split, metadata.dim)
    steps = metadata.sequence_length - INPUT_SEQUENCE_LENGTH
    result = rollout(model, store[args.trajectory], steps, device)

    frames = args.frames or [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]
    stem = f"{name}_{args.split}_{args.trajectory}"
    strip = render_comparison_strip(
        result,
        metadata.bounds,
        frames,
        args.out_dir / f"rollout_{stem}.png",
        title=name,
        point_size=args.point_size,
    )
    print(f"wrote {strip}")
    if args.video:
        video = save_rollout_video(
            result,
            metadata.bounds,
            args.out_dir / f"rollout_{stem}.mp4",
            point_size=args.point_size,
        )
        print(f"wrote {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
