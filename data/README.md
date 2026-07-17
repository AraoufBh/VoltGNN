[README.md](https://github.com/user-attachments/files/30121104/README.md)
# Datasets

VoltGNN is evaluated on three field datasets and two IEEE bus-system testbeds.

| Dataset | Nodes | Duration | Freq. (min) | Regime | Reported metrics |
|---------|-------|----------|-------------|--------|------------------|
| Pecan Street | 25 | 3 years | 15 | Field | EDE, RUR |
| Smart Meter (London) | 111 | 2 years | 30 | Field | EDE, RUR |
| NREL Solar | 50 | 1 year | 30 | Field (synthetic, physically consistent) | EDE, RUR |
| IEEE 33-Bus | 33 | Simulated | 5 | Bus | PGA, OLR, RUR |
| IEEE 118-Bus | 118 | Simulated | 5 | Bus | PGA, OLR, RUR |

## Download

The raw data is **not redistributed** here. Obtain each source from its origin:

- **Pecan Street** — appliance-level consumption, solar PV, and EV-charging data
  for residential households in Austin, TX at 15-min resolution.
  <https://dataport.pecanstreet.org/> (registration required).
  We select 25 households with co-located PV + EV instrumentation and build a
  k-NN graph (k = 5) on geographic proximity supplemented by historical
  consumption correlation.

- **Smart Meter Energy Consumption** — 30-min readings from 111 London households
  over two years. Nodes are individual meters; edges from a correlation-based
  adjacency with threshold τ_c = 0.7.
  <https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset>

- **NREL Solar Power Data** — synthetic-but-physically-consistent half-hourly PV
  output for 50 sites across the US Southwest (GHI, DNI, AC generation). Sites
  connected by a geographic k-NN graph (k = 3) augmented with irradiance
  correlation edges.
  <https://www.kaggle.com/datasets/codingmaster24/nrel-solar-power-data-partitioned-data>

- **IEEE 33-Bus / 118-Bus** — standard published bus systems, loaded via
  [Pandapower](https://www.pandapower.org/). DC-OPF solutions serve as
  ground-truth optimal policies. Renewable injection profiles are sampled from
  the Pecan Street / NREL records and linearly scaled to the bus ratings; loads
  are synthesized to reproduce standard distribution loading.

Place raw files under `data/raw/<dataset_name>/` (see each config's `data_root`).

## Preprocessing pipeline (Section 5.1)

1. **Temporal alignment** — all streams resampled to the dataset grid interval Δt
   by forward-fill imputation.
2. **Node feature assembly** — `x_v^t` from normalized power signals (Eq. 9),
   meteorological co-variates, and sinusoidal time embeddings (Eq. 10).
3. **Static adjacency** — physical connectivity (bus systems) or k-NN geographic
   graph (field datasets).
4. **Correlation edges** — rolling Pearson over a one-day window `w_c`; soft edges
   added for pairs exceeding τ_c = 0.7 (Eq. 12).
5. **Edge-weight update** — Eq. (11) applied at every time step.
6. **Split** — 70 % / 10 % / 20 % chronological train/val/test, preserving order.

Run it with:

```bash
python -m data.preprocessing --config configs/ieee33.yaml
```

## Leakage control

All graph-construction statistics — the rolling Pearson correlation edges of
Eq. (12) and the k-NN consumption-correlation adjacency for the field datasets —
are computed using **only the training partition**. No validation- or test-window
measurements enter the adjacency, which prevents information leakage through the
graph topology (Section 5.4).

## Metric applicability

`PGA` and `OLR` presuppose line capacities and a DC-OPF reference, so they are
reported **only** on the IEEE 33-/118-Bus systems, whose topology and ratings are
physically exact. The field graphs are geographic/correlation links between
meters or PV sites rather than metered electrical conductors, so we make **no
claim of physical line flows** on them: on those datasets we report only `EDE`
and `RUR`, computed on the proxy graph as *relative* distribution-efficiency and
renewable-utilization indicators. All absolute power-flow feasibility, overload,
and OPF-deviation claims are confined to the bus-system experiments.
