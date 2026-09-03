"""Evaluate a checkpoint and write the numbers Table 1 reports.

    gns-evaluate --checkpoint runs/goop/best.pt --split test

Writes ``result.json`` next to the checkpoint with the one-step and rollout MSE
under both particle conventions, plus the per-step rollout error curve that
Figure C.3 plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gns import INPUT_SEQUENCE_LENGTH, datasets
from gns.data.trajectories import TrajectoryStore
from gns.evaluation.metrics import one_step_metrics, rollout_metrics
from gns.metadata import Metadata
from gns.models.simulator import LearnedSimulator, SimulatorConfig


def load_model(
    checkpoint: Path, metadata: Metadata, device: torch.device
) -> tuple[LearnedSimulator, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = SimulatorConfig(**state["simulator_config"])
    model = LearnedSimulator(metadata, config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-path", default=None, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rollout-trajectories", type=int, default=None)
    parser.add_argument("--one-step-trajectories", type=int, default=None)
    parser.add_argument(
        "--one-step-stride",
        type=int,
        default=1,
        help="Evaluate every n-th window; 1 is the paper's protocol.",
    )
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    name = args.dataset or state["dataset"]
    data_path = args.data_path or datasets.get(name).path
    metadata = Metadata.load(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model, state = load_model(args.checkpoint, metadata, device)

    store = TrajectoryStore(data_path, args.split, metadata.dim)
    steps = metadata.sequence_length - INPUT_SEQUENCE_LENGTH

    one_step = one_step_metrics(
        model, store, device, limit=args.one_step_trajectories,
        stride=args.one_step_stride,
    )
    rollouts = rollout_metrics(
        model, store, steps, device, limit=args.rollout_trajectories
    )

    # An ablation run drops an ablation.json beside its config; carry it into
    # the result so the figure builder can group runs by axis without parsing
    # directory names.
    ablation = args.checkpoint.parent / "ablation.json"
    tags = json.loads(ablation.read_text()) if ablation.exists() else {}

    result = {
        "dataset": name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "step": state["step"],
        "rollout_steps": steps,
        "one_step_mse_free_particles": one_step.mse_free_particles,
        "one_step_mse_all_particles": one_step.mse_all_particles,
        "one_step_windows": one_step.num_windows,
        "rollout_mse_free_particles": rollouts.mse_free_particles,
        "rollout_mse_all_particles": rollouts.mse_all_particles,
        "rollout_trajectories": rollouts.num_trajectories,
        "per_step_rollout_mse": rollouts.per_step_mse.tolist(),
        **tags,
    }
    out = args.out or args.checkpoint.with_name(
        f"result_{args.split}_{args.checkpoint.stem}.json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"{name} [{args.split}] at step {state['step']}")
    print(f"  one-step MSE  free {one_step.mse_free_particles:.3e}"
          f"   all {one_step.mse_all_particles:.3e}")
    print(f"  rollout  MSE  free {rollouts.mse_free_particles:.3e}"
          f"   all {rollouts.mse_all_particles:.3e}")
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
