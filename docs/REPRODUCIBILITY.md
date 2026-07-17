# Reproducibility guide

This document maps the paper's sections, equations, tables, and figures to the
source files in this repository, and records the exact settings needed to
reproduce each experiment.

## Paper ⇄ source map

| Paper | Source | Notes |
|-------|--------|-------|
| Eq. (1)–(3) dynamic energy graph | `data/graph_construction.py` | `DynamicEnergyGraph` |
| Eq. (4) IoT stream | `data/preprocessing.py` | `resample_and_impute` |
| Eq. (5)–(7) policy & constraints | `models/voltgnn.py`, `models/decoder.py` | conservation + capacity |
| Eq. (8) deployment objective J | `training/rl_refinement.py`, `utils/config.py` | α weights = 1:2:5:1 |
| Eq. (9) normalization | `data/preprocessing.py` | `normalize_power` |
| Eq. (10) time embedding | `data/preprocessing.py` | `sinusoidal_time_embedding` |
| Eq. (11) operational weight | `data/graph_construction.py` | `operational_weight` |
| Eq. (12) correlation edges | `data/graph_construction.py` | `correlation_edges`, τ_c = 0.7 |
| Eq. (13) GRU | `models/encoder.py` | `GRUTemporalEncoder` |
| Eq. (14)–(16) GAT | `models/encoder.py` | `MultiHeadGATLayer` |
| Eq. (17)–(19) TGN | `models/encoder.py` | `TGNMemory` |
| Eq. (20) fusion | `models/encoder.py` | `SpatioTemporalEncoder` |
| Eq. (21) adjacency refinement | `models/encoder.py` | `AdjacencyRefinement` |
| Eq. (22) readout | `models/decoder.py` | `GraphReadout`, d_g = 256 |
| Eq. (23)–(25) decoder | `models/decoder.py` | `PolicyDecoder`, L = 4 |
| Eq. (26)–(28) transformer sublayers | `models/decoder.py` | `GraphConditionedTransformerLayer` |
| Dykstra projection | `models/decoder.py` | `dykstra_projection` |
| Eq. (29) refinement residual | `models/voltgnn.py` | `PolicyRefinementNet` |
| Eq. (30) PPO reward | `training/rl_refinement.py` | `grid_reward` |
| Eq. (31)–(35) supervised losses | `training/losses.py` | μ = (5.0, 2.0, 0.5) |
| Section 4.6 online adaptation | `training/online_adaptation.py` | T_adapt=96, K_adapt=5, w=672 |
| Eq. (36) complexity | `docs/COMPLEXITY.md` | O(Nw d_h + \|E\|d_h + K\|E\|d_h + L\|E\|²) |
| Eq. (37)–(40) metrics | `utils/metrics.py` | EDE, OLR, RUR, PGA |
| Algorithm 1 | `main.py` + `training/*` | four phases |

## Fixed settings (paper defaults)

- Optimizer: Adam, lr = 3e-4, cosine-annealing over 200 epochs, grad-clip 1.0.
- Batch size: 32 graph snapshots; full-graph message passing (GraphSAINT node
  sampling is supported but not activated for any dataset used here).
- Early stopping: 20 consecutive non-improving validation epochs.
- PPO: 500 episodes, γ = 0.99, clip ε = 0.2; encoder + decoder frozen.
- Architecture optimum: H = 4 attention heads, d_h = 128 (Fig. 7).
- Hardware: 4× NVIDIA A100 (80 GB) for training; single NVIDIA T4 for the
  latency/energy profile (Table 13).
- Seed: 42 (set in every config; see `utils/config.py`).

## Metric applicability (important)

- `PGA` and `OLR` are reported **only** on IEEE 33-/118-Bus (physically exact
  topology + DC-OPF reference).
- `EDE` and `RUR` are reported on the field datasets, computed on the proxy
  (geographic/correlation) graph — relative indicators, not ohmic line losses.

## Leakage control

All correlation edges (Eq. 12) and the k-NN adjacency for field datasets are
computed on the **training partition only**. No val/test-window measurements
enter the graph topology.

## Determinism notes

Exact bit-for-bit reproduction across hardware is not guaranteed for CUDA
attention kernels. We fix seeds and report means over multiple runs/events
(e.g., 50 outage events for TAS, 100 injection events for the generation spike,
90-day rolling windows for online adaptation). Small deviations from the
tabulated numbers are expected and do not affect the qualitative conclusions.
