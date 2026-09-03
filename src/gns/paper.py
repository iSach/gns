"""Numbers taken from the paper, so a figure can be compared against them.

Values are transcribed from Table C.4 (which repeats and extends Table 1) and
Table B.1 of Sanchez-Gonzalez et al., ICML 2020.  Nothing here is measured; it
is the reference every figure in ``gns.cli.figures`` is plotted against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperResult:
    """One row of Table C.4, in the paper's units."""

    one_step_mse: float  # x 1e-9
    rollout_mse: float  # x 1e-3
    max_particles: int  # Table B.1
    sequence_length: int
    max_edges: int  # Table B.1, approximate


TABLE_1: dict[str, PaperResult] = {
    "Water": PaperResult(2.82, 17.4, 1900, 1000, 27_000),
    "Sand": PaperResult(6.23, 2.37, 2000, 320, 21_000),
    "Goop": PaperResult(2.91, 1.89, 1900, 400, 19_000),
    "WaterDrop": PaperResult(1.52, 7.01, 1000, 1000, 12_000),
    "WaterRamps": PaperResult(4.91, 11.6, 2300, 600, 26_000),
    "SandRamps": PaperResult(2.77, 2.07, 3300, 400, 32_000),
}


def one_step_mse(dataset: str) -> float:
    """Paper one-step MSE in absolute units."""
    return TABLE_1[dataset].one_step_mse * 1e-9


def rollout_mse(dataset: str) -> float:
    """Paper rollout MSE in absolute units."""
    return TABLE_1[dataset].rollout_mse * 1e-3


# Figure 4 and C.1 sweep the following values on Goop, one axis at a time, with
# every other axis at the default.  The noise column is the paper's per-step
# sigma_v; the code parameterises noise by its standard deviation at the last of
# the C = 5 input steps, which is sqrt(5) times larger.
ABLATION_AXES: dict[str, dict] = {
    "message_passing_steps": {
        "label": "# message passing steps",
        "values": [1, 2, 5, 10, 15],
        "default": 10,
    },
    "shared_processor": {
        "label": "shared processor GNs",
        "values": [False, True],
        "default": False,
    },
    "connectivity_radius": {
        "label": "connectivity radius",
        "values": [0.003, 0.007, 0.011, 0.015, 0.02, 0.03],
        "default": 0.015,
    },
    "noise_std_per_step": {
        "label": "noise std",
        "values": [0.0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "default": 3e-4,
    },
    "use_relative_positions": {
        "label": "use relative positions",
        "values": [True, False],
        "default": True,
    },
}
