#!/usr/bin/env python
"""Measure training throughput so the step budget can be planned honestly.

    python scripts/benchmark_step.py --data-path <data>/WaterRamps --steps 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gns.metadata import Metadata
from gns.models.simulator import LearnedSimulator, SimulatorConfig
from gns.training.loop import TrainConfig, loss_on_batch, make_loader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--message-passing-steps", type=int, default=10)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    metadata = Metadata.load(args.data_path)

    config = TrainConfig(
        data_path=str(args.data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=512,
        simulator=SimulatorConfig(
            num_message_passing_steps=args.message_passing_steps
        ),
    )
    model = LearnedSimulator(metadata, config.simulator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = make_loader(config, metadata)

    examples = 0
    started = None
    for step, batch in enumerate(loader):
        if step >= args.warmup + args.steps:
            break
        batch = batch.to(device, non_blocking=True)
        loss = loss_on_batch(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == args.warmup - 1:
            torch.cuda.synchronize()
            started = time.perf_counter()
            examples = 0
        elif started is not None:
            examples += batch.num_examples

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rate = args.steps / elapsed
    print(f"batch {args.batch_size} windows, workers {args.num_workers}")
    print(f"{rate:.1f} steps/s   {examples / args.steps:.2f} examples/step")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    for total in (1e6, 5e6, 2e7):
        print(f"  {total:.0e} steps -> {total / rate / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
