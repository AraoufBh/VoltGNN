# VoltGNN: A Graph Neural Network for Autonomous Renewable Energy Redistribution in IoT-Driven Smart Grids

<p align="center">
  <img src="visualization/assets/voltgnn_workflow.png" alt="VoltGNN workflow" width="820"/>
</p>

> Official reference implementation (research skeleton) for the paper
> **"VoltGNN: A graph neural network for autonomous renewable energy redistribution in IoT-driven smart grids"**,
> *Energy and AI*, 25 (2026) 100841. DOI: [10.1016/j.egyai.2026.100841](https://doi.org/10.1016/j.egyai.2026.100841)

---

## Overview

Conventional GNN-based approaches to smart-grid management are designed for **passive** tasks: load forecasting, anomaly detection, and demand prediction. They consume grid measurements and emit *scalar predictions*; the gap between prediction and decision is bridged, if at all, by a separate optimization solver or rule-based controller.

**VoltGNN** closes that gap. It is a **generative graph-intelligence framework** that transforms a standard spatio-temporal GNN into a *graph-conditioned policy synthesis* system for autonomous renewable energy distribution. Rather than predicting future demand or generation, VoltGNN **directly generates an energy redistribution policy** — a capacity-bounded power-flow allocation over every transmission edge of a dynamically constructed grid graph.

At each control step, VoltGNN:

1. **Ingests** a snapshot of multi-source IoT streams (smart meters, PV inverters, wind-turbine SCADA, EV charging stations, battery management systems).
2. **Constructs / updates** a dynamic energy graph whose node states and edge weights evolve at every sampling interval.
3. **Propagates** spatio-temporal information through a **GAT + TGN** backbone.
4. **Decodes** an explicit distribution policy via an autoregressive, **graph-conditioned transformer decoder**.
5. **Refines** the decoded policy with a **reinforcement-guided (PPO)** residual under multi-objective grid constraints — without re-running a full optimization solve.

> ⚠️ **Scope note (please read).** VoltGNN is evaluated on **reconstructed and simulated grids** (two real-world field datasets, one synthetic-but-physically-consistent dataset, and the IEEE 33-/118-Bus systems) rather than on a live operational network. The quantitative gains reported in the paper should be read as *controlled-setting evidence* that motivates — but does not replace — validation on real grid data.

---

## Repository status

This repository is published for **reproducibility and reference**. It provides the **full project scaffold**: module layout, class/function signatures, configuration files, dataset adapters, metric definitions, and the training/evaluation entry points that mirror the paper one-to-one.

Several core routines are intentionally released as **documented skeletons** (`raise NotImplementedError(...)` / `# TODO`) pending internal review and the data-sharing / privacy clearance discussed in the paper's *Future Work*. Each stub is accompanied by the exact equation number from the paper so the intended behaviour is unambiguous. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the mapping between paper sections and source files.

If you need a specific component for your own research before the full release, please open an issue or email the corresponding author.

---

## Key Features

- ✔ **Generative** decision layer — synthesizes executable flow-allocation policies, not scalar forecasts.
- ✔ **Dynamic IoT-aware graph construction** with adaptive edge weighting (congestion + renewable surplus) and correlation-based soft edges.
- ✔ **GAT–TGN spatio-temporal encoder** capturing both topology and long-range temporal memory.
- ✔ **Graph-conditioned transformer decoder** with autoregressive edge-flow assignment.
- ✔ **Differentiable constraint projection** (Dykstra) for capacity bounds + approximate flow conservation.
- ✔ **Reinforcement-guided policy refinement** (PPO residual) over a multi-objective grid reward.
- ✔ **Online adaptation** via lightweight periodic fine-tuning against IoT feedback.
- ✔ **Validated** on Pecan Street, Smart Meter, NREL Solar, IEEE 33-Bus, and IEEE 118-Bus.

---

## Architecture

VoltGNN consists of six interdependent modules (Fig. 1 in the paper):

| # | Module | Source | Key equations |
|---|--------|--------|---------------|
| 1 | IoT Data Acquisition & Preprocessing | `data/preprocessing.py` | Eq. (4), (9), (10) |
| 2 | Dynamic Graph Construction | `data/graph_construction.py` | Eq. (1)–(3), (11)–(12) |
| 3 | Spatio-Temporal Encoder (GRU→GAT→TGN) | `models/encoder.py` | Eq. (13)–(21) |
| 4 | Generative Decision Layer (transformer decoder) | `models/decoder.py` | Eq. (22)–(28) |
| 5 | Autonomous Optimization & Feedback (PPO) | `training/rl_refinement.py` | Eq. (29)–(30) |
| 6 | Final Energy Distribution Policy | `models/voltgnn.py` | Eq. (5)–(8) |

```
IoT streams ──▶ Dynamic Energy Graph ──▶ GRU ─▶ GAT ─▶ TGN ─▶ Readout
                                                                  │
                                                                  ▼
                                        Graph-conditioned Transformer Decoder
                                                                  │
                                                     Dykstra Constraint Projection
                                                                  │
                                                     PPO Policy Refinement (residual)
                                                                  │
                                                                  ▼
                                             π̂_final : E_t → ℝ≥0  (edge flow allocation)
```

---

## Installation

```bash
git clone https://github.com/AraoufBh/VoltGNN.git
cd VoltGNN
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

Tested with **Python 3.10**, **PyTorch 2.2**, and **PyTorch Geometric 2.5**.

---

## Datasets

| Dataset | Nodes | Duration | Freq. (min) | Role |
|---------|-------|----------|-------------|------|
| [Pecan Street](https://dataport.pecanstreet.org/) | 25 | 3 years | 15 | Field |
| [Smart Meter (London)](https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset) | 111 | 2 years | 30 | Field |
| [NREL Solar](https://www.kaggle.com/datasets/codingmaster24/nrel-solar-power-data-partitioned-data) | 50 | 1 year | 30 | Field (synthetic, physically consistent) |
| IEEE 33-Bus | 33 | Simulated | 5 | Simulation testbed |
| IEEE 118-Bus | 118 | Simulated | 5 | Simulation testbed |

DC-OPF ground-truth policies for the bus systems are computed with **[Pandapower](https://www.pandapower.org/)**.

**Metric applicability.** `PGA` and `OLR` presuppose line capacities and a DC-OPF reference and are reported **only** on the IEEE 33-/118-Bus systems. On the field graphs (geographic/correlation links between meters or PV sites), we report **only** `EDE` and `RUR`, computed on the proxy topology — they are *relative* distribution-efficiency / renewable-utilization indicators, not ohmic transmission losses on a real feeder.

See [`data/README.md`](data/README.md) for download instructions and the exact preprocessing pipeline.

---

## Quick start

```bash
# 1. Prepare a dataset (resample, normalize, build graphs)
python -m data.preprocessing --config configs/ieee33.yaml

# 2. Supervised pre-training against DC-OPF targets (Phase 2)
python main.py --config configs/ieee33.yaml --stage pretrain

# 3. RL fine-tuning of the policy-refinement network (Phase 3)
python main.py --config configs/ieee33.yaml --stage rl

# 4. Evaluate (PGA / OLR / RUR / EDE / TAS / DL)
python main.py --config configs/ieee33.yaml --stage eval --checkpoint runs/ieee33/best.pt
```

All hyperparameters live in `configs/*.yaml` and default to the paper's settings.

---

## Reproducing the paper

| Result | Command | Paper reference |
|--------|---------|-----------------|
| Main results (bus systems) | `bash scripts/run_main_bus.sh` | Table 4 |
| Main results (field data) | `bash scripts/run_main_field.sh` | Table 5 |
| Ablation study | `bash scripts/run_ablation.sh` | Table 6, Fig. 4 |
| N-1 / N-2 topology adaptation | `bash scripts/run_faults.sh` | Tables 7–8 |
| Noise robustness | `bash scripts/run_noise.sh` | Tables 9–10, Fig. 5 |
| Online adaptation | `bash scripts/run_online.sh` | Table 11, Fig. 6 |
| Hyperparameter sensitivity | `bash scripts/run_sensitivity.sh` | Tables 12, Fig. 7 |
| Efficiency / energy profile | `bash scripts/run_efficiency.sh` | Table 13 |

Default training uses **4× NVIDIA A100 (80 GB)**; inference latency is benchmarked on a **single NVIDIA T4** to simulate near-edge deployment.

---

## Results (summary)

On the IEEE bus systems VoltGNN attains a policy-generation deviation (PGA) of **0.106** (33-Bus) and **0.119** (118-Bus), an overload-reduction (OLR) of **79.4%** (33-Bus), and the fastest median N-1 recovery (**6.4 steps**). On the field datasets it raises renewable utilization (RUR) by up to **9.3 pp** over the strongest baseline while keeping decision latency at **37 ms**, within the 50 ms real-time budget. Full tables are in the paper.

*Numbers are provided for reference; see the paper for the complete experimental protocol and caveats.*

---

## Paper

The paper is published in *Energy and AI*:
**VoltGNN: A graph neural network for autonomous renewable energy redistribution in IoT-driven smart grids**, Energy and AI 25 (2026) 100841 — <https://doi.org/10.1016/j.egyai.2026.100841>

If you use this repository, please cite it (see [`CITATION.cff`](CITATION.cff)).

---

## Citation

```bibtex
@article{bahi2026voltgnn,
  title   = {VoltGNN: A graph neural network for autonomous renewable energy redistribution in IoT-driven smart grids},
  author  = {Bahi, Abderaouf and Khadir, Mohamed Tarek and Ourici, Amel and Din{\c{c}}er, Hasan and Y{\"u}ksel, Serhat and Lakhdara, Amira and Djebbar, Akila and Trari, Mohamed},
  journal = {Energy and AI},
  volume  = {25},
  pages   = {100841},
  year    = {2026},
  doi     = {10.1016/j.egyai.2026.100841},
  publisher = {Elsevier}
}
```

---

## Contact

For questions or research collaborations, please contact:
**Abderaouf Bahi** — <a.bahi@univ-eltarf.dz>

Computer Science and Applied Mathematics Laboratory (LIMA), Faculty of Science and Technology, Chadli Bendjedid University, El Tarf 36000, Algeria.

---

## License

Code is released under the [MIT License](LICENSE). The paper is published open access under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
