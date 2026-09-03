"""The learnable Encode-Process-Decode graph network (paper Section 4.2).

Shapes follow the reference implementation: node and edge latents are both
``latent_size``; every MLP has ``mlp_num_hidden_layers`` ReLU hidden layers of
``mlp_hidden_size`` followed by a linear output.  Every MLP except the decoder
is followed by a LayerNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(
    input_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    output_size: int,
    layer_norm: bool = True,
) -> nn.Module:
    """MLP with ReLU hidden layers, a linear output and an optional LayerNorm."""
    layers: list[nn.Module] = []
    size = input_size
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(size, hidden_size), nn.ReLU()]
        size = hidden_size
    layers.append(nn.Linear(size, output_size))
    if layer_norm:
        layers.append(nn.LayerNorm(output_size))
    return nn.Sequential(*layers)


class InteractionNetwork(nn.Module):
    """One message-passing step with node and edge residual connections.

    The edge update sees ``[edge, receiver_node, sender_node]`` and the node
    update sees ``[sum of incoming edges, node]``, matching the feature order the
    reference implementation's graph blocks build.  Summation is the reducer the
    paper uses; it keeps the update meaningful when the neighbour count varies.
    """

    def __init__(
        self,
        latent_size: int,
        mlp_hidden_size: int,
        mlp_num_hidden_layers: int,
        update_edges: bool = True,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.update_edges = update_edges
        self.edge_fn = build_mlp(
            3 * latent_size, mlp_hidden_size, mlp_num_hidden_layers,
            latent_size, layer_norm,
        )
        self.node_fn = build_mlp(
            2 * latent_size, mlp_hidden_size, mlp_num_hidden_layers,
            latent_size, layer_norm,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self.edge_fn(
            torch.cat([edges, nodes[receivers], nodes[senders]], dim=-1)
        )
        aggregated = torch.zeros_like(nodes)
        aggregated.index_add_(0, receivers, messages)
        updated_nodes = self.node_fn(torch.cat([aggregated, nodes], dim=-1))
        updated_edges = messages if self.update_edges else torch.zeros_like(edges)
        return nodes + updated_nodes, edges + updated_edges


class EncodeProcessDecode(nn.Module):
    """Encoder, processor and decoder, as in the reference implementation."""

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        latent_size: int = 128,
        mlp_hidden_size: int = 128,
        mlp_num_hidden_layers: int = 2,
        num_message_passing_steps: int = 10,
        shared_processor: bool = False,
        layer_norm: bool = True,
        update_edges: bool = True,
    ) -> None:
        super().__init__()
        mlp = lambda in_size: build_mlp(  # noqa: E731
            in_size, mlp_hidden_size, mlp_num_hidden_layers, latent_size, layer_norm
        )
        self.node_encoder = mlp(node_input_size)
        self.edge_encoder = mlp(edge_input_size)

        self.shared_processor = shared_processor
        self.num_message_passing_steps = num_message_passing_steps
        num_blocks = 1 if shared_processor else num_message_passing_steps
        self.processor = nn.ModuleList(
            InteractionNetwork(
                latent_size, mlp_hidden_size, mlp_num_hidden_layers,
                update_edges=update_edges, layer_norm=layer_norm,
            )
            for _ in range(num_blocks)
        )
        self.decoder = build_mlp(
            latent_size, mlp_hidden_size, mlp_num_hidden_layers,
            output_size, layer_norm=False,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        for step in range(self.num_message_passing_steps):
            block = self.processor[0 if self.shared_processor else step]
            nodes, edges = block(nodes, edges, senders, receivers)
        return self.decoder(nodes)
