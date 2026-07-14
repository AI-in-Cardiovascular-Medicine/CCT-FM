# CO₂ emission estimation

Carbon-footprint estimation for the training and inference of the segmentation models
reported in the manuscript, following the Green Algorithms framework
(Lannelongue et al., *Advanced Science*, 2021):

```
CO₂e (kg) = t × n_GPU × TDP × PUE × CI × 1e-6
```

## Contents

| Path | Purpose |
|------|---------|
| `compute_co2_emissions.py` | Per-model training and per-patient inference emissions → `output/co2_emissions_results.csv` |
| `compare_co2_emissions.py` | Supplementary figure (emissions under four grid carbon intensities + car-trip reference), methodology/results text, and machine-readable results |
| `data/hardware_and_grid_config.json` | GPU TDPs, PUE, grid carbon intensities, epochs, reference comparator — all inputs, with sources |
| `data/model_computational_specs.csv` | Measured runtime, size and parameter count per model |

No value is hardcoded in the scripts; everything is read from `data/`.

## Reproducing

Requires Python 3.10 (see `.python-version`); dependencies are pinned in `pyproject.toml`.

```bash
uv run python compute_co2_emissions.py
uv run python compare_co2_emissions.py
```

Both write to `output/` (git-ignored).
