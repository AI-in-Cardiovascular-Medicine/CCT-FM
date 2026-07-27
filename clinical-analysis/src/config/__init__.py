"""Central configuration — features, targets, hyperparameters, seeds.
Every other module imports from here so the pipeline has a single source of truth.

Packaged as src/config/ (the module lives in __init__.py so `from config import …`
keeps working verbatim).
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────
# This file is src/config/__init__.py, so three dirname() calls climb
# __init__.py → config/ → src/ → project root.
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root (config package lives in src/config/)
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data_for_correlation (1).xlsx")
# OUT_DIR is env-overridable so the SAME pipeline can be re-run on a filtered
# cohort into a separate folder (e.g. output-male / output-female) without
# touching the canonical full-cohort run. When CCT_OUT_DIR is unset this is the
# default "output" folder — i.e. behaviour is byte-for-byte unchanged.
OUT_DIR   = os.environ.get("CCT_OUT_DIR", os.path.join(BASE_DIR, "output"))
CKPT_DIR  = os.path.join(OUT_DIR, "archived")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

# ─── Cohort filters ──────────────────────────────────────────────────────
BSA_RANGE = (1.2, 2.5)             # m²
N_MIN     = 150                    # minimum complete cases per target

# Optional sex stratification (env-driven). When CCT_SEX_FILTER is "male" or
# "female", core.load_cohort() additionally restricts the cohort to that sex.
# Base_Sex coding in this dataset is 0 = Male, 1 = Female (confirmed by the data
# owner). Sex is intentionally KEPT in the feature lists when stratifying: within
# a single-sex stratum it is a constant column, which CatBoost never splits on and
# StandardScaler+Ridge zero out, so it contributes nothing and leaves every other
# feature's R²/ΔR²/SHAP-η numerically identical — the pipeline stays "the same".
SEX_FILTER = os.environ.get("CCT_SEX_FILTER")     # "male" | "female" | None
SEX_CODE   = {"male": 0, "female": 1}

# ─── Feature definitions ─────────────────────────────────────────────────
RAW_VOLUMES = [
    "LVM__original_shape_VoxelVolume",
    "LA__original_shape_VoxelVolume",
    "LV__original_shape_VoxelVolume",
    "RA__original_shape_VoxelVolume",
    "RV__original_shape_VoxelVolume",
    "Precardial Fat__original_shape_VoxelVolume",
    "Epicardial Fat__original_shape_VoxelVolume",
    "LA Appendage__original_shape_VoxelVolume",
]
RATIOS        = []   # inter-chamber ratios are algebraic transforms of raw volumes — excluded
DEMOGRAPHICS  = ["Base_Age", "Base_Sex", "Base_BSA"]

# Convenience aliases
BASELINE_FEATURES = DEMOGRAPHICS                              # 3
FULL_FEATURES     = DEMOGRAPHICS + RAW_VOLUMES                 # 11 (ratios excluded: they are algebraic transforms of the raw volumes — collinear and redundant for tree-based models)

# Indexed VoxelVolume display names (= raw_volume / Base_BSA in code)
RAW_TO_INDEXED = {
    "LVM__original_shape_VoxelVolume":              "LVM_Vi",
    "LA__original_shape_VoxelVolume":               "LA_Vi",
    "LV__original_shape_VoxelVolume":               "LV_Vi",
    "RA__original_shape_VoxelVolume":               "RA_Vi",
    "RV__original_shape_VoxelVolume":               "RV_Vi",
    "Precardial Fat__original_shape_VoxelVolume":   "PcFat_Vi",
    "Epicardial Fat__original_shape_VoxelVolume":   "EpFat_Vi",
    "LA Appendage__original_shape_VoxelVolume":     "LAA_Vi",
}
INDEXED_VOLUMES = list(RAW_TO_INDEXED.values())   # 8 indexed volumes

# Display labels (no '_i' suffix anywhere)
FEATURE_DISPLAY = {
    "Base_Age": "Age", "Base_Sex": "Sex", "Base_BSA": "BSA",
    "LVM__original_shape_VoxelVolume": "LVM",
    "LA__original_shape_VoxelVolume":  "LA",
    "LV__original_shape_VoxelVolume":  "LV",
    "RA__original_shape_VoxelVolume":  "RA",
    "RV__original_shape_VoxelVolume":  "RV",
    "Precardial Fat__original_shape_VoxelVolume": "PcFat",
    "Epicardial Fat__original_shape_VoxelVolume": "EpFat",
    "LA Appendage__original_shape_VoxelVolume":   "LAA",
    "LVM_Vi": "LVM", "LA_Vi": "LA", "LV_Vi": "LV", "RA_Vi": "RA",
    "RV_Vi": "RV", "PcFat_Vi": "PcFat", "EpFat_Vi": "EpFat", "LAA_Vi": "LAA",
    "LA/LV": "LA/LV", "RA/RV": "RA/RV", "LAA/LA": "LAA/LA",
    "RV/LV": "RV/LV", "RA/LA": "RA/LA", "RA/LV": "RA/LV", "LA/RV": "LA/RV",
}

# ─── Targets ─────────────────────────────────────────────────────────────
CLINICAL_TARGETS = [
    "Creatinine", "eGFR", "Hb", "Thrombocytes", "Creatine kinase", "CK-MB",
    "Brain natriuretic peptide", "Albumin", "Leucocytes", "Base_BMI",
    "Base_ES2", "Base_LoES", "Base_STS", "TTE_TEE_Ang_LVEF",
    "TTE_TEE_Ang_MG", "TTE_TEE_Ang_IAVA", "Base_NYHA",
]
TARGET_DISPLAY = {
    "Creatinine": "Creatinine", "eGFR": "eGFR", "Hb": "Hemoglobin",
    "Thrombocytes": "Platelets", "Creatine kinase": "CK", "CK-MB": "CK-MB",
    "Brain natriuretic peptide": "BNP", "Albumin": "Albumin",
    "Leucocytes": "WBC", "Base_BMI": "BMI", "Base_ES2": "EuroSCORE II",
    "Base_LoES": "Log. EuroSCORE", "Base_STS": "STS Score",
    "TTE_TEE_Ang_LVEF": "LVEF", "TTE_TEE_Ang_MG": "Mean Gradient",
    "TTE_TEE_Ang_IAVA": "AVA Index", "Base_NYHA": "NYHA Class",
}

# ─── Pre-specified primary outcomes (confirmatory; rest are exploratory) ─
PRIMARY_OUTCOMES = ["TTE_TEE_Ang_LVEF", "Brain natriuretic peptide",
                    "TTE_TEE_Ang_MG"]
PRIMARY_DISPLAY  = [TARGET_DISPLAY[t] for t in PRIMARY_OUTCOMES]

# ─── Model + statistical settings ────────────────────────────────────────
N_FOLDS              = 5
INNER_VAL_FRAC       = 0.15        # inner validation slice for early stopping
N_BOOT               = 1000
N_PERM               = 500         # permutations for primary-outcome null
N_MIN_FOR_CV         = 150
FDR_Q                = 0.05        # Benjamini-Hochberg threshold
ALPHA                = 0.05

# Repeated CV knob — averages OOF predictions over multiple random fold-splits.
# Single seed 42 is the final reporting configuration (the run reported in the paper).
# The [0, 7, 13, 23, 42] 5×-repeated-CV option remains available but is not used.
RANDOM_STATE_LIST    = [42]
RANDOM_STATE_PRIMARY = 42          # canonical seed for reporting

# CatBoost default hyperparameters (used everywhere except the nested-CV
# step on primary outcomes, which tunes within this grid)
CB_PARAMS = dict(
    iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    random_seed=RANDOM_STATE_PRIMARY, verbose=0, thread_count=8,
    loss_function="RMSE", early_stopping_rounds=30,
)

# Nested-CV hyperparameter grid for primary outcomes (C3)
# Set ENABLE_PRIMARY_TUNING = True to engage nested CV hyperparameter tuning on
# the 3 primary outcomes (adds ~10 min); reviewer-grade rigour.
ENABLE_PRIMARY_TUNING = False

CB_TUNE_GRID = {
    "depth":         [4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg":   [1, 3, 5],
}

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# ─── Target-side preprocessing (applied in core.load_cohort) ───
# Standard clinical-chemistry transforms / plausibility clips for skewed or
# error-prone analytes.
TARGET_TRANSFORMS = {
    "Creatine kinase":           "log1p",  # right-skewed enzymatic assay
    "CK-MB":                     "log1p",  # right-skewed; detection-limit floor at 0.5
}
TARGET_CLIPS = {
    "Albumin": (None, 60.0),               # serum albumin caps biologically at ~60 g/L;
                                           # values > 60 → NaN (4 rows in this cohort)
}

