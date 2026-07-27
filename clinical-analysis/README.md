# Clinical association analysis

Code accompanying the manuscript. It tests whether CT-derived cardiac substructure volumes
carry clinical information **beyond demographics** (age, sex, body-surface area), in an
aortic-stenosis / TAVI cohort:

- **Pairwise SHAP-η** — attribution of each predictor to each clinical/laboratory target, alone and adjusted for age and sex.
- **Incremental value** — out-of-fold ΔR² of a demographics + volumes model over a demographics-only baseline (CatBoost and RidgeCV, 5-fold CV, BCa bootstrap CIs, Benjamini–Hochberg FDR at q = 0.05).
- **Sensitivity analyses** — volume-only models, a BMI negative control, linearity and calibration checks, and an optional permutation test.

## Data availability

**The patient-level data are not released.** They are individual-level clinical imaging data and
cannot be shared publicly. This repository contains the analysis code only.

The pipeline expects a single spreadsheet at `data/raw_data_for_correlation (1).xlsx`
(path set in `src/config/__init__.py`), one row per patient, with these columns:

| Group | Columns |
|-------|---------|
| Demographics | `Base_Age`, `Base_Sex` (0 = male, 1 = female), `Base_BSA` (m²) |
| CT substructure volumes | `LVM__original_shape_VoxelVolume`, `LA__…`, `LV__…`, `RA__…`, `RV__…`, `Precardial Fat__…`, `Epicardial Fat__…`, `LA Appendage__…` (mm³) |
| Clinical / laboratory targets | `Creatinine`, `eGFR`, `Hb`, `Thrombocytes`, `Creatine kinase`, `CK-MB`, `Brain natriuretic peptide`, `Albumin`, `Leucocytes`, `Base_BMI`, `Base_ES2`, `Base_LoES`, `Base_STS`, `TTE_TEE_Ang_LVEF`, `TTE_TEE_Ang_MG`, `TTE_TEE_Ang_IAVA`, `Base_NYHA` |

Cohort filters, transforms, targets, seeds and model hyperparameters are all defined in
`src/config/__init__.py` — the single source of truth. No result is hardcoded anywhere in the code.

`Supplementary_Supporting_Data.xlsx` holds the **aggregate** results reported in the paper
(cohort characteristics, incremental-value table, SHAP-η matrices) and contains no
patient-level records.

## Running

Python 3.10 (see `.python-version`); dependencies pinned in `pyproject.toml` / `uv.lock`.

```bash
uv run python src/run_pipeline.py            # full pipeline (7 phases)
uv run python src/run_pipeline.py --from 4   # resume from phase 4
```

On Windows, set `PYTHONUTF8=1` (the runner already does this for its child phases).

Every analysis phase checkpoints to `output/archived/*.pkl`, so an interrupted run can be
resumed. All figures and tables are written to `output/` (git-ignored).

### Sex-stratified reruns

The same pipeline can be re-run on a single sex without touching the full-cohort output:

```bash
CCT_SEX_FILTER=male   CCT_OUT_DIR=output-male   uv run python src/run_pipeline.py
CCT_SEX_FILTER=female CCT_OUT_DIR=output-female uv run python src/run_pipeline.py
```

With both variables unset the run is the canonical full-cohort analysis.
