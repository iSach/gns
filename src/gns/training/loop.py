"""The training loop (paper Section 4.3, Supplementary B.3).

One loss, one optimizer, one schedule.  The loss is the mean squared error on
the *normalized* acceleration, taken over non-obstacle particles only, and the
learning rate decays exponentially from 1e-4 to 1e-6 with a half-life set so it
has dropped by a decade every 5M steps.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gns import INPUT_SEQUENCE_LENGTH
from gns.data.onestep import Batch, OneStepDataset
from gns.data.trajectories import TrajectoryStore
from gns.evaluation.metrics import rollout_metrics
from gns.metadata import Metadata
from gns.models.simulator import LearnedSimulator, SimulatorConfig
from gns.training.run_lock import RunLock, describe, held

LR_START = 1e-4
LR_FINAL = 1e-6
# The paper decays by a decade every 5M steps over a 20M-step budget, so the
# rate reaches its floor exactly as training ends.  A run with a smaller budget
# has to shorten this horizon in the same proportion or it stops while the rate
# is still near its starting value; ``TrainConfig.lr_decay_steps`` is that knob
# and every deviation from 5e6 is recorded in the run's config.json.
LR_DECAY_STEPS = 5e6


def learning_rate(
    step: int,
    decay_steps: float = LR_DECAY_STEPS,
    start: float = LR_START,
) -> float:
    """Exponential decay from ``start`` to 1e-6, as written in Supplementary B.3.

    ``start`` is 1e-4 in the paper.  Section 4.3 says the models "can train in
    significantly less steps" and that the conservative rate was chosen to make
    comparisons across settings fair, so raising it is the paper's own answer to
    a smaller step budget rather than a change of method.
    """
    return LR_FINAL + (start - LR_FINAL) * 0.1 ** (step / decay_steps)


@dataclass
class TrainConfig:
    """Everything the loop needs that is not the model itself."""

    dataset: str = "Goop"
    data_path: str = ""
    run_dir: str = "runs/debug"
    num_steps: int = 20_000_000
    # The paper's nominal mini-batch: the budget is this many copies of the
    # largest graph in the dataset, packed with as many examples as fit.
    batch_size: int = 2
    lr_decay_steps: float = LR_DECAY_STEPS
    lr_start: float = LR_START
    compile_model: bool = True
    shuffle_buffer: int = 10_000
    num_workers: int = 8
    seed: int = 0
    log_every: int = 1_000
    checkpoint_every: int = 50_000
    resume: bool = True
    eval_every: int = 250_000
    eval_trajectories: int = 5
    max_train_seconds: float = 0.0  # 0: no wall-clock limit
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)


def make_loader(config: TrainConfig, metadata: Metadata) -> DataLoader:
    dataset = OneStepDataset(
        path=config.data_path,
        split="train",
        radius=(
            config.simulator.connectivity_radius
            if config.simulator.connectivity_radius is not None
            else metadata.connectivity_radius
        ),
        noise_std=config.simulator.noise_std,
        batch_size=config.batch_size,
        dim=metadata.dim,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )


def loss_on_batch(model: LearnedSimulator, batch: Batch) -> torch.Tensor:
    """Mean squared error on the normalized acceleration, obstacles excluded.

    The target is the acceleration that reaches the *true* next position from
    the *noisy* last input position.  That makes the model correct the noise in
    the input velocity while leaving the noise in the input position alone,
    which is the choice Supplementary B.3 describes.
    """
    predicted = model.predict_normalized_acceleration(
        batch.positions, batch.particle_types, batch.senders, batch.receivers
    )
    target = model.normalized_acceleration_target(
        batch.target + batch.target_noise, batch.positions
    )
    squared = (predicted - target) ** 2
    mask = batch.loss_mask.unsqueeze(-1)
    return (squared * mask).sum() / mask.sum().clamp(min=1)


def train(config: TrainConfig, device: torch.device) -> Path:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if held(run_dir):
        holder = describe(run_dir)
        raise SystemExit(
            f"Another trainer holds {run_dir}: "
            f"pid {holder.get('pid')} on {holder.get('host')} "
            f"(Slurm job {holder.get('job') or 'none'}). "
            "Stop it, or delete RUNNING.json if you know it is gone."
        )
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    metadata = Metadata.load(config.data_path)

    torch.manual_seed(config.seed)
    model = LearnedSimulator(metadata, config.simulator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_start)
    loader = make_loader(config, metadata)

    # Every batch has a different number of particles and edges, so the graph is
    # compiled once for dynamic shapes rather than once per shape.
    step_fn = (
        torch.compile(loss_on_batch, dynamic=True)
        if config.compile_model and device.type == "cuda"
        else loss_on_batch
    )
    lr_at = lambda at: learning_rate(  # noqa: E731
        at, config.lr_decay_steps, config.lr_start
    )

    start_step = 0
    best = float("inf")
    latest = run_dir / "latest.pt"
    if config.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best = float(state.get("best_validation", float("inf")))
        print(f"resumed from {latest} at step {start_step}", flush=True)

    # Checkpoint selection follows the paper: full-length rollouts on a few
    # held-out validation trajectories, keeping the parameters with the best
    # rollout MSE.  The test split is never consulted during training.
    validation = TrajectoryStore(config.data_path, "valid", metadata.dim)
    rollout_steps = metadata.sequence_length - INPUT_SEQUENCE_LENGTH

    lock = RunLock(run_dir).__enter__()
    curve = (run_dir / "curve.jsonl").open("a")
    started = time.perf_counter()
    running = 0.0
    seen = 0
    step = start_step

    def evaluate(at_step: int) -> None:
        nonlocal best
        model.eval()
        metrics = rollout_metrics(
            model, validation, rollout_steps, device, limit=config.eval_trajectories
        )
        model.train()
        record = {
            "step": at_step,
            "valid_rollout_mse_free": metrics.mse_free_particles,
            "valid_rollout_mse_all": metrics.mse_all_particles,
            "seconds": time.perf_counter() - started,
        }
        curve.write(json.dumps(record) + "\n")
        curve.flush()
        print(
            f"[valid] step {at_step:>9d}  rollout MSE (free) "
            f"{metrics.mse_free_particles:.4e}",
            flush=True,
        )
        if metrics.mse_free_particles < best:
            best = metrics.mse_free_particles
            save_checkpoint(run_dir / "best.pt", model, config, at_step, best=best)

    for batch in loader:
        if step >= config.num_steps:
            break
        batch = batch.to(device, non_blocking=True)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        loss = step_fn(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()
        seen += batch.num_examples
        step += 1

        if step % config.log_every == 0:
            elapsed = time.perf_counter() - started
            record = {
                "step": step,
                "loss": running / config.log_every,
                "lr": lr_at(step),
                "examples": seen,
                "seconds": elapsed,
                "steps_per_second": step / elapsed,
            }
            curve.write(json.dumps(record) + "\n")
            curve.flush()
            lock.beat()
            print(
                f"step {step:>9d}  loss {record['loss']:.5f}  "
                f"lr {record['lr']:.2e}  {record['steps_per_second']:.1f} it/s",
                flush=True,
            )
            running = 0.0

        if step % config.checkpoint_every == 0:
            save_checkpoint(
                run_dir / "latest.pt", model, config, step,
                optimizer=optimizer, best=best,
            )

        if step % config.eval_every == 0:
            evaluate(step)

        if config.max_train_seconds and (
            time.perf_counter() - started > config.max_train_seconds
        ):
            break

    save_checkpoint(
        run_dir / "latest.pt", model, config, step, optimizer=optimizer, best=best
    )
    if step % config.eval_every != 0:
        evaluate(step)
    curve.close()
    lock.__exit__(None, None, None)
    return run_dir


def save_checkpoint(
    path: Path,
    model: LearnedSimulator,
    config: TrainConfig,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    best: float | None = None,
) -> None:
    """Write a checkpoint, atomically so a kill cannot leave a truncated file."""
    state = {
        "step": step,
        "model": model.state_dict(),
        "simulator_config": asdict(config.simulator),
        "dataset": config.dataset,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if best is not None:
        state["best_validation"] = best
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
