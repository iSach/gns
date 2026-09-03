"""PyTorch reimplementation of Graph Network-based Simulators (GNS).

Reference: Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec and Battaglia,
"Learning to Simulate Complex Physics with Graph Networks", ICML 2020.
https://arxiv.org/abs/2002.09405
"""

__version__ = "0.1.0"

# Fixed by the paper: the model conditions on C = 5 previous velocities, which
# needs C + 1 = 6 previous positions.
INPUT_SEQUENCE_LENGTH = 6

# The released datasets label particles with these ids.  The embedding table is
# sized for all of them so a checkpoint transfers between datasets.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

__all__ = [
    "INPUT_SEQUENCE_LENGTH",
    "KINEMATIC_PARTICLE_ID",
    "NUM_PARTICLE_TYPES",
    "__version__",
]
