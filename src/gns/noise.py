"""Random-walk noise on the input velocities (paper Section 4.3, B.3).

Independent samples are drawn per input state, particle and spatial dimension
and accumulated as a random walk over the ``C`` input velocities.  The scale is
parameterised by the standard deviation *at the last step*, so a change in ``C``
does not change how far the most recent input has drifted.  The paper quotes the
per-step scale instead (sigma_v = 3e-4); the two differ by ``sqrt(C)``, which is
why the reference implementation's default is 6.7e-4.
"""

from __future__ import annotations

import numpy as np

# Default from the reference implementation, equal to the paper's
# sigma_v = 0.0003 per step accumulated over C = 5 velocities.
DEFAULT_NOISE_STD = 6.7e-4


def per_step_std(noise_std_last_step: float, num_velocities: int) -> float:
    """Convert a last-step std into the per-step std the paper reports."""
    return noise_std_last_step / num_velocities**0.5


def random_walk_noise(
    position_sequence: np.ndarray,
    noise_std_last_step: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return position noise for a ``[num_particles, seq_len, dim]`` sequence.

    The first position is left untouched: it only enters the model through the
    first velocity difference, so perturbing it would double count.
    """
    num_velocities = position_sequence.shape[1] - 1
    scale = noise_std_last_step / num_velocities**0.5
    velocity_noise = rng.normal(
        scale=scale,
        size=(position_sequence.shape[0], num_velocities, position_sequence.shape[2]),
    ).astype(np.float32)
    velocity_noise = np.cumsum(velocity_noise, axis=1)
    position_noise = np.concatenate(
        [np.zeros_like(velocity_noise[:, :1]), np.cumsum(velocity_noise, axis=1)],
        axis=1,
    )
    return position_noise
