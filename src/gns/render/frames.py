"""Rendering rollouts the way the paper's figures show them.

Particles are drawn as flat dots coloured by material, ground truth beside
prediction, on the simulation bounds.  The colour map is the one the reference
implementation's renderer uses, so a figure here is directly comparable to a
figure in the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# From render_rollout.py in the reference implementation.
TYPE_COLORS = {
    0: "green",  # rigid solid
    3: "black",  # boundary particle
    5: "royalblue",  # water
    6: "gold",  # sand
    7: "magenta",  # goop
}


def _draw(ax, positions, particle_types, bounds, point_size):
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.6")
    for particle_type in np.unique(particle_types):
        mask = particle_types == particle_type
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            s=point_size,
            c=TYPE_COLORS.get(int(particle_type), "grey"),
            linewidths=0,
        )


def render_comparison_strip(
    result,
    bounds,
    frames: list[int],
    path: str | Path,
    title: str | None = None,
    point_size: float = 2.0,
    dpi: int = 200,
) -> Path:
    """Ground truth above, prediction below, at the given rollout steps."""
    fig, axes = plt.subplots(
        2, len(frames), figsize=(1.7 * len(frames), 3.6), squeeze=False
    )
    for column, frame in enumerate(frames):
        _draw(axes[0][column], result.ground_truth[frame], result.particle_types,
              bounds, point_size)
        _draw(axes[1][column], result.predicted[frame], result.particle_types,
              bounds, point_size)
        axes[0][column].set_title(f"step {frame}", fontsize=8)
    axes[0][0].set_ylabel("ground truth", fontsize=8)
    axes[1][0].set_ylabel("GNS (ours)", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_rollout_video(
    result,
    bounds,
    path: str | Path,
    fps: int = 30,
    point_size: float = 2.0,
    dpi: int = 120,
) -> Path:
    """Write a side-by-side ground-truth/prediction video of a full rollout."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in range(result.predicted.shape[0]):
            for ax, data, label in (
                (axes[0], result.ground_truth[frame], "ground truth"),
                (axes[1], result.predicted[frame], "GNS (ours)"),
            ):
                ax.clear()
                _draw(ax, data, result.particle_types, bounds, point_size)
                ax.set_title(label, fontsize=9)
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(image)
    plt.close(fig)
    return path
