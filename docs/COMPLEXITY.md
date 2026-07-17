[COMPLEXITY.md](https://github.com/user-attachments/files/30121382/COMPLEXITY.md)# Complexity & efficiency

## Forward-pass complexity (Section 4.7)

Let `N` = number of nodes, `|E_t|` = number of edges, `w` = temporal window,
`d_h` = hidden dimension, `L_dec` = number of decoder layers. The per-step
forward-pass cost is (Eq. 36):

```
O( N·w·d_h  +  |E_t|·d_h  +  K·|E_t|·d_h  +  L_dec·|E_t|² )
     GRU        GAT           TGN msgs        decoder (dominant)
```

The `|E_t|²` term in the autoregressive decoder dominates for large topologies.
For the IEEE 118-Bus experiments (`N = 118`, `|E_t| ≈ 186`) this is tractable.
For larger distribution networks, block-sparse attention over topological
neighborhoods reduces it to `O(|E_t|·k_max²)`, where `k_max` is the maximum
node degree (future work).

## Measured profile (Table 13, NVIDIA T4)

| Model | GFLOPs | Mem (MB) | Energy (J) | Latency (ms) |
|-------|-------:|---------:|-----------:|-------------:|
| TGN | 0.061 | 1.8 | 0.71 | 21 |
| VoltGNN | 0.108 | 2.3 | 1.55 | 37 |
| VoltGNN (INT8) | — | — | 1.16 | 29 |

VoltGNN uses ≈1.55 J per decision step (≈2.2× TGN) at 37 ms latency (≈1.8× TGN).
Because heavier models also draw higher board power, energy overhead grows
slightly faster than latency. Contextually, this per-step energy is negligible
compared with the renewable energy managed per step (tens of kWh at 15-min
resolution). Post-INT8 quantization drops energy to ≈1.16 J and latency to
≈29 ms with PGA degradation below 1.5 %, suitable for mid-tier edge hardware
such as the NVIDIA Jetson Orin.

Energy is estimated via the NVIDIA NVML power-sampling API during the
latency-profiling runs: per-step energy = mean active board power over the
inference window × per-step latency; idle baseline power excluded.

