"""
Spatio-temporal encoder for VoltGNN.

Implements Section 4.3 of the paper:
    GRU temporal embedding  -> Eq. (13)
    Multi-head GAT           -> Eq. (14)-(16)
    TGN memory              -> Eq. (17)-(19)
    Node representation      -> Eq. (20)
    Adjacency refinement     -> Eq. (21)

Default dimensions (Section 5.4 / Fig. 7):
    K (attention heads) = 4
    d_h (GRU hidden)    = 128
    d_m (TGN memory)    = d_h
    d_z = d_h + d_m
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUTemporalEncoder(nn.Module):
    """Per-node temporal embedding over the IoT measurement window (Eq. 13).

    h_v^t = GRU(x_v^{t-w:t}) in R^{d_h}

    Captures ramp rates and intra-day generation cycles within each node
    *independently*, before any graph propagation.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_window: (N, w, in_dim) node feature window.
        Returns:
            h: (N, d_h) temporal node embeddings.
        """
        # TODO: return the last hidden state per node.
        raise NotImplementedError("GRUTemporalEncoder.forward — see Eq. (13).")


class MultiHeadGATLayer(nn.Module):
    """Edge-weight-modulated multi-head graph attention (Eq. 14-16).

    e_uv   = LeakyReLU( a^T [ W_q h_v || W_k h_u ] )              (14)
    alpha  = softmax_u ( e_uv * W_uv^t )                          (15)  <- edge weight modulates logit
    h~_v   = sigma( sum_u alpha_uv * W_v h_u )                    (16)

    The dynamic operational edge weight W_uv^t (Eq. 11) multiplies the
    attention logit, so topology / congestion changes are reflected
    immediately in the attention distribution.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by number of heads"
        self.heads = heads
        self.head_dim = dim // heads
        # Per-head projections W_q, W_k, W_v in R^{d_h x (d_h/K)} and
        # per-head attention vector a in R^{2 d_h / K}.
        # TODO: declare parameters.

    def forward(self, h, edge_index, edge_weight):
        """
        Args:
            h:           (N, d_h) node embeddings from the GRU.
            edge_index:  (2, |E|) directed edges of G_t.
            edge_weight: (|E|,) dynamic operational weights W_uv^t (Eq. 11).
        Returns:
            h_tilde: (N, d_h) attention-refined node embeddings.
        """
        # TODO: compute e_uv (14), softmax with edge-weight modulation (15),
        #       aggregate per head (16), concatenate heads.
        raise NotImplementedError("MultiHeadGATLayer.forward — see Eq. (14)-(16).")


class TGNMemory(nn.Module):
    """Temporal Graph Network memory module (Eq. 17-19).

    msg_bar_v = MeanAgg{ msg_uv : (u,v) in E_t }                  (17)
    msg_uv    = MLP( m_u^{t-}, m_v^{t-}, dt_uv, eta_uv )          (18)
    m_v^t     = GRU( msg_bar_v, m_v^{t-} )                        (19)

    eta_uv is a static edge embedding initialised from W~_uv (Eq. 3).
    Memory is updated continuously; at deployment it advances WITHOUT
    gradient computation (implicit online adaptation, Section 4.6).
    """

    def __init__(self, num_nodes: int, memory_dim: int = 128, edge_dim: int = 16):
        super().__init__()
        self.memory_dim = memory_dim
        # Persistent per-node memory buffer m_v (not a learnable parameter).
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))
        # TODO: declare msg-MLP (Eq. 18) and update-GRU (Eq. 19).

    @torch.no_grad()
    def detach_memory(self) -> None:
        """Detach memory between windows to avoid backprop-through-time blowup."""
        self.memory = self.memory.detach()

    def forward(self, edge_index, edge_time, edge_static_emb):
        """Update and return the current memory vectors m_v^t."""
        # TODO: message computation (18), mean aggregation (17), GRU update (19).
        raise NotImplementedError("TGNMemory.forward — see Eq. (17)-(19).")


class AdjacencyRefinement(nn.Module):
    """Lightweight post-encoding adjacency refinement (Eq. 21).

    W_hat_uv^t = sigma( w_r^T [ z_u^t || z_v^t ] )

    This is a *learned* refinement that augments the adjacency for the next
    propagation step; it does NOT replace the operational weight W_uv^t used
    inside the attention (Eq. 15). It lets the model capture latent load
    correlations beyond what is directly observable from IoT measurements.
    """

    def __init__(self, z_dim: int):
        super().__init__()
        self.w_r = nn.Linear(2 * z_dim, 1)

    def forward(self, z, edge_index):
        # TODO: sigmoid( w_r^T [z_u || z_v] ) per edge.
        raise NotImplementedError("AdjacencyRefinement.forward — see Eq. (21).")


class SpatioTemporalEncoder(nn.Module):
    """Full encoder: GRU -> GAT -> TGN -> fuse (Eq. 13-21).

    Produces the fused node representation
        z_v^t = LayerNorm( h~_v^t || m_v^t )  in R^{d_z},  d_z = d_h + d_m   (20)
    consumed by the generative decision layer.
    """

    def __init__(
        self,
        in_dim: int,
        num_nodes: int,
        hidden_dim: int = 128,
        memory_dim: int = 128,
        heads: int = 4,
    ):
        super().__init__()
        self.gru = GRUTemporalEncoder(in_dim, hidden_dim)
        self.gat = MultiHeadGATLayer(hidden_dim, heads=heads)
        self.tgn = TGNMemory(num_nodes, memory_dim)
        self.z_dim = hidden_dim + memory_dim
        self.layer_norm = nn.LayerNorm(self.z_dim)
        self.adj_refine = AdjacencyRefinement(self.z_dim)

    def forward(self, x_window, edge_index, edge_weight, edge_time, edge_static_emb):
        """
        Returns:
            z:      (N, d_z) fused node representations (Eq. 20).
            w_hat:  (|E|,) refined adjacency weights for next step (Eq. 21).
        """
        # h  = self.gru(x_window)                               # Eq. (13)
        # h_tilde = self.gat(h, edge_index, edge_weight)        # Eq. (14)-(16)
        # m  = self.tgn(edge_index, edge_time, edge_static_emb) # Eq. (17)-(19)
        # z  = self.layer_norm(torch.cat([h_tilde, m], dim=-1)) # Eq. (20)
        # w_hat = self.adj_refine(z, edge_index)                # Eq. (21)
        # return z, w_hat
        raise NotImplementedError("SpatioTemporalEncoder.forward — see Section 4.3.")
