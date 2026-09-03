"""Shape, invariance and parameter-count checks for the model."""

import numpy as np
import pytest
import torch

from gns import INPUT_SEQUENCE_LENGTH
from gns.metadata import Metadata, Stats
from gns.models.simulator import LearnedSimulator, SimulatorConfig
from gns.neighbors import radius_graph_numpy, radius_graph_torch


def toy_metadata() -> Metadata:
    return Metadata(
        bounds=np.array([[0.1, 0.9], [0.1, 0.9]], dtype=np.float32),
        sequence_length=100,
        connectivity_radius=0.015,
        dim=2,
        dt=0.0025,
        vel=Stats(np.zeros(2, np.float32), np.full(2, 2e-3, np.float32)),
        acc=Stats(np.zeros(2, np.float32), np.full(2, 2e-4, np.float32)),
    )


def toy_batch(num_particles=200, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.15, 0.85, size=(num_particles, 1, 2))
    positions = positions + rng.normal(scale=2e-3, size=(num_particles, INPUT_SEQUENCE_LENGTH, 2))
    positions = positions.astype(np.float32)
    types = np.full(num_particles, 5, dtype=np.int64)
    senders, receivers = radius_graph_numpy(positions[:, -1], 0.05)
    return (
        torch.from_numpy(positions),
        torch.from_numpy(types),
        torch.from_numpy(senders),
        torch.from_numpy(receivers),
    )


def test_forward_shapes():
    model = LearnedSimulator(toy_metadata(), SimulatorConfig())
    positions, types, senders, receivers = toy_batch()
    out = model(positions, types, senders, receivers)
    assert out.shape == (positions.shape[0], 2)
    assert torch.isfinite(out).all()


def test_integrate_inverts_target():
    """``normalized_acceleration_target`` must invert ``integrate`` exactly."""
    model = LearnedSimulator(toy_metadata(), SimulatorConfig())
    positions, _, _, _ = toy_batch()
    accel = torch.randn(positions.shape[0], 2)
    next_position = model.integrate(accel, positions)
    recovered = model.normalized_acceleration_target(next_position, positions)
    # The round trip adds an acceleration of order 1e-4 to a position of order
    # 1e-1 and divides the difference back out, so float32 leaves about three
    # digits.  That cancellation is inherent to the paper's parameterisation.
    torch.testing.assert_close(recovered, accel, rtol=2e-3, atol=2e-3)


def test_relative_encoder_is_translation_invariant():
    """Shifting the whole scene must not change the predicted acceleration."""
    model = LearnedSimulator(toy_metadata(), SimulatorConfig()).eval()
    positions, types, senders, receivers = toy_batch()
    with torch.no_grad():
        base = model.predict_normalized_acceleration(positions, types, senders, receivers)
        shifted = model.predict_normalized_acceleration(
            positions + 0.02, types, senders, receivers
        )
    # The only position-dependent feature is the wall distance, and both scenes
    # sit further than one radius from every wall, so both are clipped to the
    # same value and the predictions must agree.
    torch.testing.assert_close(base, shifted, rtol=1e-4, atol=1e-5)


def test_parameter_count_matches_paper_architecture():
    """10 steps x 2 MLPs, plus encoder, decoder and the type embedding."""
    model = LearnedSimulator(toy_metadata(), SimulatorConfig())
    total = sum(p.numel() for p in model.parameters())
    # Reported by the reference architecture: roughly 1.6M parameters.
    assert 1.4e6 < total < 1.8e6


@pytest.mark.parametrize("shared", [False, True])
def test_shared_processor_parameter_count(shared):
    config = SimulatorConfig(shared_processor=shared)
    model = LearnedSimulator(toy_metadata(), config)
    total = sum(p.numel() for p in model.processor_parameters())
    assert (total < 2e5) if shared else (total > 1e6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_neighbor_backends_agree():
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.1, 0.9, size=(1500, 2)).astype(np.float32)
    a = radius_graph_numpy(positions, 0.05)
    b = radius_graph_torch(torch.as_tensor(positions).cuda(), 0.05)
    left = set(zip(a[0].tolist(), a[1].tolist()))
    right = set(zip(b[0].cpu().tolist(), b[1].cpu().tolist()))
    assert left == right
