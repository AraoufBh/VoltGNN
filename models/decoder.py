"""
Generative decision layer for VoltGNN — the core novelty (Section 4.4).

Graph readout                 -> Eq. (22)
Decoder query / layer / head  -> Eq. (23)-(25)
Transformer sublayers          -> Eq. (26)-(28)
Dykstra constraint projection  -> Section 4.4 (after Eq. 28)

Defaults (Section 4.4 / 5.4):
    d_g = 256                (global graph embedding)
    L   = 4                  (decoder layers)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphReadout(nn.Module):
    """Permutation-invariant global graph embedding (Eq. 22).

    g_t = MLP( (1/N) sum_v z_v^t )  in R^{d_g},  d_g = 256

    Encodes the global energy state of the grid at time t; used as the
    cross-attention context for every decoding step.
    """

    def __init__(self, z_dim: int, g_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, g_dim),
            nn.ReLU(),
            nn.Linear(g_dim, g_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: mean over nodes then MLP.
        raise NotImplementedError("GraphReadout.forward — see Eq. (22).")


class GraphConditionedTransformerLayer(nn.Module):
    """One decoder layer with cross-attention over the graph context (Eq. 26-28).

    q'   = SelfAttn(q) + q                         (26)
    q''  = CrossAttn(q', g_t) + q'                 (27)
    q_out = FFN(q'') + q''                         (28)
    """

    def __init__(self, dim: int, g_dim: int, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        # TODO: self-attention, cross-attention (query=q', key/value=g_t), FFN,
        #       plus the residual LayerNorms.

    def forward(self, q, g_t, causal_mask=None):
        # TODO: Eq. (26)-(28).
        raise NotImplementedError("GraphConditionedTransformerLayer.forward — see Eq. (26)-(28).")


class PolicyDecoder(nn.Module):
    """Autoregressive graph-conditioned transformer decoder (Eq. 23-25).

    q_uv^(0)   = W_e [ z_u || z_v || W_uv^t || psi(t) ]              (23)
    q_uv^(l+1) = TransformerLayer( q_uv^(l), g_t, f_hat_<uv )        (24)
    f_hat_uv   = C_uv * sigma( w_f^T q_uv^(L) )                      (25)

    Edges are decoded in an order given by a priority score derived from
    the current congestion ratio rho_uv^t. Autoregressive conditioning:
    each edge attends to the flows already assigned to earlier edges
    (f_hat_<uv). The sigmoid guarantees f_hat_uv in [0, C_uv].
    """

    def __init__(self, z_dim: int, g_dim: int = 256, num_layers: int = 4, time_dim: int = 8):
        super().__init__()
        self.num_layers = num_layers
        query_in = 2 * z_dim + 1 + time_dim  # [z_u || z_v || W_uv || psi(t)]
        self.q_proj = nn.Linear(query_in, g_dim)                     # W_e (Eq. 23)
        self.layers = nn.ModuleList(
            GraphConditionedTransformerLayer(g_dim, g_dim) for _ in range(num_layers)
        )
        self.flow_head = nn.Linear(g_dim, 1)                         # w_f (Eq. 25)

    @staticmethod
    def priority_order(congestion_ratio: torch.Tensor) -> torch.Tensor:
        """Edge decoding order sorted by congestion ratio rho_uv^t (Section 4.4)."""
        # TODO: return argsort of the priority score.
        raise NotImplementedError("PolicyDecoder.priority_order — see Section 4.4.")

    def forward(self, z, edge_index, edge_weight, capacity, time_feats, g_t, congestion_ratio):
        """
        Returns:
            f_hat: (|E|,) decoded per-edge flow allocations, f_hat_uv in [0, C_uv].
        """
        # order = self.priority_order(congestion_ratio)
        # for each edge in order:
        #   q0 = self.q_proj([z_u || z_v || W_uv || psi(t)])          # Eq. (23)
        #   q  = run through self.layers conditioned on g_t and f_hat_<uv   # Eq. (24)
        #   f_hat_uv = C_uv * sigmoid(self.flow_head(q))              # Eq. (25)
        raise NotImplementedError("PolicyDecoder.forward — see Eq. (23)-(25).")


def dykstra_projection(f_hat, edge_index, capacity, node_injection, n_iter: int = 1):
    """Differentiable constraint projection (Section 4.4).

    Enforces the capacity bounds 0 <= f_uv <= C_uv and approximately
    satisfies flow conservation (Eq. 6) via a single step of the Dykstra
    alternating-projection algorithm. Differentiable wrt f_hat so gradients
    flow through it during training.

    Args:
        f_hat:          (|E|,) raw decoded flows.
        edge_index:     (2, |E|).
        capacity:       (|E|,) line capacities C_uv.
        node_injection: (N,) net active power p_v^t (Eq. 6 RHS).
        n_iter:         number of alternating projection sweeps (paper uses 1).
    Returns:
        f_proj: (|E|,) projected, (approximately) feasible flows.
    """
    # TODO: alternate (a) capacity box projection and (b) nodal-balance
    #       projection for n_iter sweeps; keep the operation differentiable.
    raise NotImplementedError("dykstra_projection — see Section 4.4.")
