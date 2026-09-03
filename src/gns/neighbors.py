"""Radius connectivity.

The paper builds the interaction graph by connecting every particle to every
other particle within the connectivity radius, using a k-d tree
(Supplementary B.2).  Self edges are included, matching the reference
implementation's ``add_self_edges=True`` default.

Two backends produce the same edge set.  The k-d tree runs in the data loader's
worker processes, where it is free; the pairwise-distance backend runs on the
GPU and is what rollouts use, because there the positions only exist on device.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def radius_graph_numpy(
    positions: np.ndarray,
    radius: float,
    add_self_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(senders, receivers)`` for one graph.

    Edge ``k`` carries information from ``senders[k]`` to ``receivers[k]``; the
    relative displacement feature is ``pos[sender] - pos[receiver]``, so the
    receiver aggregates displacements pointing towards its neighbours.
    """
    tree = cKDTree(positions)
    neighbor_lists = tree.query_ball_point(positions, r=radius)
    counts = np.fromiter((len(n) for n in neighbor_lists), dtype=np.int64,
                         count=len(neighbor_lists))
    senders = np.repeat(np.arange(len(positions), dtype=np.int64), counts)
    receivers = np.concatenate(
        [np.asarray(n, dtype=np.int64) for n in neighbor_lists]
    ) if len(neighbor_lists) else np.zeros(0, dtype=np.int64)
    if not add_self_edges:
        keep = senders != receivers
        senders, receivers = senders[keep], receivers[keep]
    return senders, receivers


def radius_graph_torch(
    positions: torch.Tensor,
    radius: float,
    add_self_edges: bool = True,
    chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU equivalent of :func:`radius_graph_numpy` for a single graph.

    Distances are computed in chunks of rows so memory stays bounded for the
    large generalization scenes, which have tens of thousands of particles.
    """
    n = positions.shape[0]
    senders, receivers = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = torch.cdist(positions[start:stop], positions)
        hit = block <= radius
        if not add_self_edges:
            index = torch.arange(start, stop, device=positions.device)
            hit[torch.arange(stop - start, device=positions.device), index] = False
        rows, cols = torch.nonzero(hit, as_tuple=True)
        senders.append(rows + start)
        receivers.append(cols)
    return torch.cat(senders), torch.cat(receivers)
