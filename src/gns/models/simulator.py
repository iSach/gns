"""The learned simulator: features in, next positions out (paper Section 4.1).

This wraps :class:`~gns.models.graph_network.EncodeProcessDecode` with the
paper's input encoding, output normalization and semi-implicit Euler update.
Positions are always the raw simulator positions; every normalization happens
inside this module so the training loop and the rollout share one convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from gns import KINEMATIC_PARTICLE_ID, NUM_PARTICLE_TYPES
from gns.metadata import Metadata
from gns.models.graph_network import EncodeProcessDecode


@dataclass
class SimulatorConfig:
    """Every architectural choice the paper ablates in Figure 4 and C.1."""

    num_message_passing_steps: int = 10
    latent_size: int = 128
    mlp_hidden_size: int = 128
    mlp_num_hidden_layers: int = 2
    particle_type_embedding_size: int = 16
    shared_processor: bool = False
    layer_norm: bool = True
    update_edges: bool = True
    # Figure 4(i,j).  The relative encoder is the paper's default: edges carry
    # the displacement between the two particles and nodes never see where they
    # are.  The paper does not spell out the absolute variant, so we use the
    # natural counterpart -- nodes see their own position, edges see both
    # endpoint positions -- and say so rather than guessing silently.
    use_relative_positions: bool = True
    connectivity_radius: float | None = None  # None: take the dataset default
    noise_std: float = 6.7e-4


class LearnedSimulator(nn.Module):
    """Predicts the next particle positions from the last six positions."""

    def __init__(self, metadata: Metadata, config: SimulatorConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = metadata.dim
        self.radius = (
            config.connectivity_radius
            if config.connectivity_radius is not None
            else metadata.connectivity_radius
        )

        stats = metadata.normalization(config.noise_std)
        for name, stat in stats.items():
            self.register_buffer(f"{name}_mean", torch.as_tensor(stat.mean))
            self.register_buffer(f"{name}_std", torch.as_tensor(stat.std))
        self.register_buffer("bounds", torch.as_tensor(np.asarray(metadata.bounds)))

        self.particle_embedding = nn.Embedding(
            NUM_PARTICLE_TYPES, config.particle_type_embedding_size
        )
        # C = 5 velocities, 2 * dim clipped wall distances, the type embedding,
        # and, for the absolute encoder, the position itself.
        node_input_size = (
            5 * self.dim + 2 * self.dim + config.particle_type_embedding_size
        )
        if not config.use_relative_positions:
            node_input_size += self.dim
        # Relative displacement plus its magnitude, or both endpoint positions.
        edge_input_size = (
            self.dim + 1 if config.use_relative_positions else 2 * self.dim
        )

        self.network = EncodeProcessDecode(
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            output_size=self.dim,
            latent_size=config.latent_size,
            mlp_hidden_size=config.mlp_hidden_size,
            mlp_num_hidden_layers=config.mlp_num_hidden_layers,
            num_message_passing_steps=config.num_message_passing_steps,
            shared_processor=config.shared_processor,
            layer_norm=config.layer_norm,
            update_edges=config.update_edges,
        )

    # -- features -----------------------------------------------------------

    def encode(
        self,
        position_sequence: torch.Tensor,
        particle_types: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build node and edge features from ``[N, seq_len, dim]`` positions."""
        latest = position_sequence[:, -1]
        velocities = position_sequence[:, 1:] - position_sequence[:, :-1]

        normalized_velocity = (velocities - self.velocity_mean) / self.velocity_std
        node_features = [normalized_velocity.flatten(start_dim=1)]

        # Distance to each wall, clipped at the connectivity radius.  Clipping
        # is what keeps the feature translation invariant away from the walls.
        to_lower = latest - self.bounds[:, 0]
        to_upper = self.bounds[:, 1] - latest
        walls = torch.cat([to_lower, to_upper], dim=-1) / self.radius
        node_features.append(walls.clamp(-1.0, 1.0))

        node_features.append(self.particle_embedding(particle_types))
        if not self.config.use_relative_positions:
            node_features.append(latest)

        if self.config.use_relative_positions:
            displacement = (latest[senders] - latest[receivers]) / self.radius
            distance = torch.linalg.norm(displacement, dim=-1, keepdim=True)
            edge_features = [displacement, distance]
        else:
            edge_features = [latest[senders], latest[receivers]]
        return torch.cat(node_features, dim=-1), torch.cat(edge_features, dim=-1)

    # -- prediction ---------------------------------------------------------

    def predict_normalized_acceleration(
        self,
        position_sequence: torch.Tensor,
        particle_types: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        nodes, edges = self.encode(position_sequence, particle_types, senders, receivers)
        return self.network(nodes, edges, senders, receivers)

    def integrate(
        self, normalized_acceleration: torch.Tensor, position_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Semi-implicit Euler with dt = 1 (Supplementary A)."""
        acceleration = (
            normalized_acceleration * self.acceleration_std + self.acceleration_mean
        )
        latest = position_sequence[:, -1]
        velocity = latest - position_sequence[:, -2]
        new_velocity = velocity + acceleration
        return latest + new_velocity

    def normalized_acceleration_target(
        self, next_position: torch.Tensor, position_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of :meth:`integrate`: the target the loss is defined on."""
        latest = position_sequence[:, -1]
        velocity = latest - position_sequence[:, -2]
        next_velocity = next_position - latest
        acceleration = next_velocity - velocity
        return (acceleration - self.acceleration_mean) / self.acceleration_std

    def processor_parameters(self):
        """Parameters of the message-passing blocks only, for size comparisons."""
        return self.network.processor.parameters()

    def forward(
        self,
        position_sequence: torch.Tensor,
        particle_types: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        """Return the next position for every particle."""
        normalized = self.predict_normalized_acceleration(
            position_sequence, particle_types, senders, receivers
        )
        return self.integrate(normalized, position_sequence)


def kinematic_mask(particle_types: torch.Tensor) -> torch.Tensor:
    """True for obstacle particles, whose motion is prescribed, not predicted."""
    return particle_types == KINEMATIC_PARTICLE_ID
