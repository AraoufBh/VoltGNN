# Contributing to VoltGNN

Thanks for your interest in VoltGNN. This repository accompanies the paper
*"VoltGNN: A graph neural network for autonomous renewable energy redistribution
in IoT-driven smart grids"* (Energy and AI, 2026) and is maintained primarily
for reproducibility and research reference.

## Ways to contribute

- **Reporting issues** — bugs, unclear documentation, or questions about a
  specific equation/module. Please reference the relevant paper section or
  equation number where possible.
- **Requesting a component** — some core routines are released as documented
  skeletons pending internal review / data-sharing clearance. If you need one
  for your own research, open an issue describing your use case.
- **Improvements** — extensions such as AC-OPF supervision, block-sparse
  attention for larger grids, reactive-power / voltage-setpoint dispatch, or
  privacy / communication-efficiency analysis (all noted as future work in the
  paper) are especially welcome.

## Development guidelines

1. Fork the repository and create a feature branch.
2. Keep changes focused and reference the paper section/equation each change
   implements (the codebase uses `Eq. (n)` / `Section x.y` markers throughout).
3. Match the existing style: NumPy-style docstrings, type hints where practical,
   and equation references in module headers.
4. If you implement a skeleton (`raise NotImplementedError`), please include a
   short unit test demonstrating the intended behaviour.
5. Run `pre-commit` / your formatter of choice before opening a PR.

## Reproducibility

If your change affects any reported result, please note which table/figure it
touches and include the command used to regenerate it (see the `scripts/`
directory and the *Reproducing the paper* table in the README).

## Contact

For research collaborations or questions, contact the corresponding author:
**Abderaouf Bahi** — <a.bahi@univ-eltarf.dz>.
