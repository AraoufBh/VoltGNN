"""
Dataset adapters (Section 5.1).

Two regimes (Section 5.1, "data-to-graph mapping"):

  Field graphs (Pecan Street, Smart Meter, NREL Solar)
      - each node is a real physical entity (metered household / PV site);
      - features taken directly from measured demand, PV, EV-charging records;
      - edges are geographic / correlation links (proxy topology);
      - metrics reported: EDE, RUR only (no physical line flows claimed).

  Bus systems (IEEE 33-Bus, IEEE 118-Bus)
      - standard published topology (physically exact);
      - renewable injection sampled from Pecan Street / NREL, scaled to ratings;
      - bus loads synthesized to standard distribution loading;
      - DC-OPF (Pandapower) provides ground-truth optimal policies;
      - metrics reported: PGA, OLR (+ RUR).

All adapters return per-time-step dynamic graphs consumable by models/voltgnn.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetSpec:
    name: str
    nodes: int
    freq_min: int
    regime: str        # "field" or "bus"
    metrics: tuple     # subset of ("PGA", "OLR", "RUR", "EDE")


DATASETS = {
    "pecan_street": DatasetSpec("Pecan Street", 25, 15, "field", ("EDE", "RUR")),
    "smart_meter":  DatasetSpec("Smart Meter", 111, 30, "field", ("EDE", "RUR")),
    "nrel_solar":   DatasetSpec("NREL Solar", 50, 30, "field", ("EDE", "RUR")),
    "ieee33":       DatasetSpec("IEEE 33-Bus", 33, 5, "bus", ("PGA", "OLR", "RUR")),
    "ieee118":      DatasetSpec("IEEE 118-Bus", 118, 5, "bus", ("PGA", "OLR", "RUR")),
}


class VoltGNNDataset:
    """Base class yielding windows of dynamic energy graphs."""

    def __init__(self, root: str, spec: DatasetSpec, window: int, split: str = "train"):
        self.root = root
        self.spec = spec
        self.window = window
        self.split = split

    def __len__(self):
        raise NotImplementedError("VoltGNNDataset.__len__ — implement per dataset.")

    def __getitem__(self, idx):
        """Return a window {G_{t-w..t}} and (for bus systems) the DC-OPF target."""
        raise NotImplementedError("VoltGNNDataset.__getitem__ — see Section 5.1.")


class BusSystemDataset(VoltGNNDataset):
    """IEEE 33-/118-Bus with Pandapower DC-OPF ground-truth labels.

    Renewable injection profiles are drawn from Pecan Street / NREL and
    linearly scaled to the bus ratings; loads are synthesized to standard
    distribution loading. Topology perturbations (line outages, generation
    spikes, sensor noise) are injected at random intervals for the robustness
    experiments (Section 5.7).
    """

    def compute_dcopf_targets(self):
        """Solve DC-OPF with Pandapower to obtain pi*_t (Section 4.5)."""
        raise NotImplementedError("compute_dcopf_targets — requires pandapower.")


class FieldDataset(VoltGNNDataset):
    """Real metered field datasets on proxy (geographic/correlation) graphs."""
    pass
