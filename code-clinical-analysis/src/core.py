"""Shared model-fitting and statistical helpers used by all analysis scripts.

Statistical hardening utilities:
  * fit_catboost_oof          — 5-fold CV with inner-validation early stopping
  * fit_catboost_oof_repeated — fits the above under multiple seeds (B1)
  * fit_ridge_oof             — RidgeCV linear baseline
  * paired_percentile_ci      — original percentile bootstrap
  * paired_bca_ci             — bias-corrected accelerated bootstrap (B2)
  * bootstrap_pvalue          — two-sided empirical p from a bootstrap delta sample
  * benjamini_hochberg        — multiplicity correction (A1)
  * perfold_paired_ttest      — parametric per-fold cross-check (A2)
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import (
    DATA_PATH, BSA_RANGE, N_FOLDS, INNER_VAL_FRAC, CB_PARAMS,
    RIDGE_ALPHAS, RAW_TO_INDEXED, RANDOM_STATE_PRIMARY, RANDOM_STATE_LIST,
    TARGET_TRANSFORMS, TARGET_CLIPS, SEX_FILTER, SEX_CODE,
)

# ─── Data loading ────────────────────────────────────────────────────────
def load_cohort():
    """Load the dataset, apply the BSA plausibility filter, optionally restrict
    to one sex (SEX_FILTER), derive BSA-indexed columns, and apply target-side
    plausibility clips and log-transforms (TARGET_CLIPS, TARGET_TRANSFORMS) so
    every downstream analysis sees the cleaned data. Returns a DataFrame."""
    df = pd.read_excel(DATA_PATH)
    df = df[(df["Base_BSA"] >= BSA_RANGE[0]) & (df["Base_BSA"] <= BSA_RANGE[1])].copy()

    # Optional single-sex stratification (Base_Sex: 0 = Male, 1 = Female).
    # Applied after the BSA filter so the cohort definition is BSA-then-sex.
    if SEX_FILTER:
        key = SEX_FILTER.strip().lower()
        if key not in SEX_CODE:
            raise ValueError(f"CCT_SEX_FILTER must be 'male' or 'female', got {SEX_FILTER!r}")
        n_pre = len(df)
        df = df[df["Base_Sex"] == SEX_CODE[key]].copy()
        print(f"  [load_cohort] sex filter = {key} (Base_Sex == {SEX_CODE[key]}); "
              f"n {n_pre:,} -> {len(df):,}")

    for raw, indexed in RAW_TO_INDEXED.items():
        df[indexed] = df[raw] / df["Base_BSA"]

    # Target-side plausibility clips: out-of-range values → NaN
    for col, (lo, hi) in TARGET_CLIPS.items():
        if col not in df.columns: continue
        n_pre = df[col].notna().sum()
        if lo is not None:
            df.loc[df[col] < lo, col] = np.nan
        if hi is not None:
            df.loc[df[col] > hi, col] = np.nan
        n_post = df[col].notna().sum()
        if n_pre != n_post:
            print(f"  [load_cohort] {col}: dropped {n_pre - n_post} out-of-range "
                  f"values (clip = [{lo}, {hi}]); remaining n = {n_post:,}")

    # Target-side transforms: log1p for skewed enzymatic assays
    for col, transform in TARGET_TRANSFORMS.items():
        if col not in df.columns: continue
        if transform == "log1p":
            df[col] = np.log1p(df[col])
            print(f"  [load_cohort] {col}: log1p-transformed for downstream analysis")
        else:
            raise ValueError(f"Unknown TARGET_TRANSFORMS value: {transform}")
    return df


# ─── Core model fits ─────────────────────────────────────────────────────
def fit_catboost_oof(df_sub, feats, target, random_state=RANDOM_STATE_PRIMARY,
                    cb_params=None):
    """5-fold CV CatBoost regression with inner-validation early stopping.

    Inner-val slice is taken from the training fold, so the held-out test
    fold is never used for model selection. Returns (y, OOF preds, stitched
    SHAP, per-fold R²s).
    """
    if cb_params is None:
        cb_params = {**CB_PARAMS, "random_seed": random_state}
    X, y = df_sub[feats].values, df_sub[target].values
    kf = KFold(N_FOLDS, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y))
    oof_shap = np.zeros_like(X, dtype=float)
    fold_r2  = []
    for tr, te in kf.split(X):
        X_tr, X_iv, y_tr, y_iv = train_test_split(
            X[tr], y[tr], test_size=INNER_VAL_FRAC, random_state=random_state)
        m = CatBoostRegressor(**cb_params)
        m.fit(Pool(X_tr, y_tr), eval_set=Pool(X_iv, y_iv))
        pred = m.predict(X[te])
        oof_pred[te] = pred
        sv = m.get_feature_importance(data=Pool(X[te], y[te]), type="ShapValues")
        oof_shap[te] = sv[:, :-1]
        fold_r2.append(r2_score(y[te], pred))
    return y, oof_pred, oof_shap, fold_r2


def fit_catboost_oof_repeated(df_sub, feats, target,
                              seeds=RANDOM_STATE_LIST,
                              cb_params=None):
    """B1 — Repeated 5-fold CV. Returns a dict with per-seed OOF preds and
    aggregate metrics, plus the mean SHAP across seeds (for stable attribution).
    """
    per_seed = []
    for s in seeds:
        y, pred, shap, fold_r2 = fit_catboost_oof(
            df_sub, feats, target, random_state=s, cb_params=cb_params)
        per_seed.append({"seed": s, "y": y, "pred": pred,
                         "shap": shap, "fold_r2": fold_r2})
    # Average OOF predictions (each patient is OOF under every seed)
    mean_pred = np.mean([p["pred"] for p in per_seed], axis=0)
    mean_shap = np.mean([p["shap"] for p in per_seed], axis=0)
    seed_r2s  = [r2_score(p["y"], p["pred"]) for p in per_seed]
    return {
        "y":          per_seed[0]["y"],
        "mean_pred":  mean_pred,
        "mean_shap":  mean_shap,
        "per_seed":   per_seed,
        "seed_r2s":   seed_r2s,
        "agg_r2":     float(np.mean(seed_r2s)),
        "agg_r2_sd":  float(np.std(seed_r2s)),
    }


def fit_ridge_oof(df_sub, feats, target, random_state=RANDOM_STATE_PRIMARY):
    """RidgeCV (α tuned in inner CV) under the same outer 5-fold CV."""
    X, y = df_sub[feats].values, df_sub[target].values
    kf = KFold(N_FOLDS, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Xtr, y[tr])
        oof[te] = r.predict(Xte)
    return oof


# ─── Bootstrap CIs ───────────────────────────────────────────────────────
def paired_percentile_ci(y, pred_a, pred_b, n_boot=1000, seed=42):
    """Original percentile bootstrap. Returns (CI for R²_b, CI for R²_a, CI for Δ),
    plus the full Δ-distribution for downstream p-value derivation."""
    rng = np.random.default_rng(seed)
    n = len(y); ba, bb, bd = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.var(y[idx]) < 1e-12: continue
        ra = r2_score(y[idx], pred_a[idx]); rb = r2_score(y[idx], pred_b[idx])
        ba.append(ra); bb.append(rb); bd.append(rb - ra)
    p = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    return p(ba), p(bb), p(bd), np.array(bd)


def paired_bca_ci(y, pred_a, pred_b, n_boot=1000, seed=42, ci=0.95):
    """B2 — Bias-corrected accelerated bootstrap on paired R² difference.
    Returns the same shape as paired_percentile_ci but with BCa intervals."""
    n = len(y)
    obs_a = r2_score(y, pred_a)
    obs_b = r2_score(y, pred_b)
    obs_d = obs_b - obs_a

    rng = np.random.default_rng(seed)
    boot_a, boot_b, boot_d = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.var(y[idx]) < 1e-12: continue
        ra = r2_score(y[idx], pred_a[idx]); rb = r2_score(y[idx], pred_b[idx])
        boot_a.append(ra); boot_b.append(rb); boot_d.append(rb - ra)
    boot_a = np.asarray(boot_a); boot_b = np.asarray(boot_b); boot_d = np.asarray(boot_d)

    # Jackknife for acceleration
    jk_a, jk_b, jk_d = [], [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        ra = r2_score(y[mask], pred_a[mask]); rb = r2_score(y[mask], pred_b[mask])
        jk_a.append(ra); jk_b.append(rb); jk_d.append(rb - ra)
    jk_a = np.asarray(jk_a); jk_b = np.asarray(jk_b); jk_d = np.asarray(jk_d)

    def _bca(boot, jk, obs):
        z0 = stats.norm.ppf(np.mean(boot < obs))
        if not np.isfinite(z0):
            z0 = 0.0
        diff = jk.mean() - jk
        num   = (diff ** 3).sum()
        denom = 6.0 * (((diff ** 2).sum()) ** 1.5 + 1e-12)
        a = num / denom
        z_alpha   = stats.norm.ppf((1 - ci) / 2)
        z_1malpha = stats.norm.ppf(1 - (1 - ci) / 2)
        alpha_lo  = stats.norm.cdf(z0 + (z0 + z_alpha)   / (1 - a * (z0 + z_alpha)))
        alpha_hi  = stats.norm.cdf(z0 + (z0 + z_1malpha) / (1 - a * (z0 + z_1malpha)))
        a_lo_pre, a_hi_pre = alpha_lo, alpha_hi
        alpha_lo  = float(np.clip(alpha_lo, 1e-4, 1 - 1e-4))
        alpha_hi  = float(np.clip(alpha_hi, 1e-4, 1 - 1e-4))
        if (a_lo_pre != alpha_lo) or (a_hi_pre != alpha_hi):
            print(f'    [BCa] alpha-clipping triggered (pre: [{a_lo_pre:.4f},{a_hi_pre:.4f}]); CI may be approximate.')
        return [float(np.percentile(boot, 100 * alpha_lo)),
                float(np.percentile(boot, 100 * alpha_hi))]

    ci_a = _bca(boot_a, jk_a, obs_a)
    ci_b = _bca(boot_b, jk_b, obs_b)
    ci_d = _bca(boot_d, jk_d, obs_d)
    return ci_a, ci_b, ci_d, boot_d


def bootstrap_pvalue(boot_delta, observed_delta=None):
    """Two-sided empirical p-value from a bootstrap-delta distribution.
    p = 2 * min(P(d ≤ 0), P(d ≥ 0)) — invariant to the sign of the effect."""
    boot_delta = np.asarray(boot_delta)
    p_lo = (boot_delta <= 0).mean()
    p_hi = (boot_delta >= 0).mean()
    return float(min(1.0, 2 * min(p_lo, p_hi)))


# ─── Multiplicity correction (A1) ────────────────────────────────────────
def benjamini_hochberg(pvalues, q=0.05):
    """Returns (adjusted_pvalues, reject_array) using BH FDR."""
    if not pvalues:
        return [], []
    reject, p_adj, _, _ = multipletests(pvalues, alpha=q, method="fdr_bh")
    return p_adj.tolist(), reject.tolist()


# ─── Per-fold paired t-test (A2) ─────────────────────────────────────────
def perfold_paired_ttest(fold_r2_a, fold_r2_b):
    """Paired t-test on ΔR² across folds. df = n_folds − 1.
    Returns (t_stat, two-sided p)."""
    if len(fold_r2_a) < 2 or len(fold_r2_a) != len(fold_r2_b):
        return float("nan"), float("nan")
    diffs = np.asarray(fold_r2_b) - np.asarray(fold_r2_a)
    if np.std(diffs, ddof=1) == 0:
        return float("nan"), 1.0
    t, p = stats.ttest_rel(fold_r2_b, fold_r2_a)
    return float(t), float(p)


# ─── SHAP η ──────────────────────────────────────────────────────────────
def shap_eta(shap_col, y):
    """Single-feature SHAP η = sqrt(Var(φ)/Var(y))."""
    vy = float(np.var(y))
    if vy < 1e-12: return 0.0
    return float(np.sqrt(np.var(shap_col) / vy))
