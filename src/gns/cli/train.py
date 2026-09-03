"""Train one GNS model.

    gns-train --dataset Goop --run-dir runs/goop

Every flag below is a hyperparameter the paper either fixes or ablates; the
defaults are the paper's default architecture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gns import datasets
from gns.models.simulator import SimulatorConfig
from gns.noise import DEFAULT_NOISE_STD
from gns.training.loop import LR_DECAY_STEPS, TrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Goop", choices=datasets.names())
    parser.add_argument("--data-path", default=None, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--num-steps", type=int, default=20_000_000)
    parser.add_argument(
        "--lr-decay-steps",
        type=float,
        default=None,
        help="Steps per decade of learning-rate decay. Default: the paper's 5e6 "
        "for a 20M-step run, scaled down proportionally for a shorter one.",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch even if the run directory holds a checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--eval-trajectories", type=int, default=5)
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        default=0.0,
        help="Stop after this much wall clock; 0 runs to --num-steps.",
    )

    group = parser.add_argument_group("architecture (paper Figure 4 and C.1)")
    group.add_argument("--message-passing-steps", type=int, default=10)
    group.add_argument("--latent-size", type=int, default=128)
    group.add_argument("--mlp-hidden-size", type=int, default=128)
    group.add_argument("--mlp-hidden-layers", type=int, default=2)
    group.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    group.add_argument("--connectivity-radius", type=float, default=None)
    group.add_argument("--shared-processor", action="store_true")
    group.add_argument("--no-layer-norm", action="store_true")
    group.add_argument("--no-edge-updates", action="store_true")
    group.add_argument("--absolute-encoder", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    entry = datasets.get(args.dataset)
    data_path = args.data_path or entry.path

    simulator = SimulatorConfig(
        num_message_passing_steps=args.message_passing_steps,
        latent_size=args.latent_size,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_num_hidden_layers=args.mlp_hidden_layers,
        noise_std=args.noise_std,
        connectivity_radius=args.connectivity_radius,
        shared_processor=args.shared_processor,
        layer_norm=not args.no_layer_norm,
        update_edges=not args.no_edge_updates,
        use_relative_positions=not args.absolute_encoder,
    )
    decay = args.lr_decay_steps
    if decay is None:
        decay = LR_DECAY_STEPS * min(1.0, args.num_steps / 20_000_000)
    config = TrainConfig(
        dataset=args.dataset,
        lr_decay_steps=decay,
        compile_model=not args.no_compile,
        resume=not args.no_resume,
        data_path=str(data_path),
        run_dir=str(args.run_dir),
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        num_workers=args.num_workers,
        seed=args.seed,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        eval_trajectories=args.eval_trajectories,
        max_train_seconds=args.max_train_seconds,
        simulator=simulator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    train(config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
