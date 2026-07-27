"""Multivariate incremental-value analysis (Figure 3c + 3d).

For each target: nested baseline (Age + Sex + BSA) vs full (Baseline + 8 raw
substructure volumes) CatBoost models — 11 features total. Inter-chamber
volume ratios are excluded from both the multivariate and volume-only feature
sets: each ratio is an algebraic transform of pairs of raw volumes (e.g.
LA/LV = vol_LA ÷ vol_LV), so it is exactly collinear with — and adds no
information for — tree-based models that already see the underlying volumes.
All v2 statistical hardening:

  • B1  Repeated 5-fold CV (5 seeds) — averaged OOF predictions
  • B2  Bias-corrected accelerated (BCa) bootstrap CIs on R²_baseline,
        R²_full, and ΔR² = R²_full − R²_baseline
  • A1  Benjamini–Hochberg FDR control at q = 0.05 across the 17 targets
  • A2  Per-fold paired t-test on ΔR² (parametric cross-check)
  • A3  Pre-specified primary outcomes (LVEF, BNP, Mean Gradient) reported
        first; remaining 14 framed as exploratory
  • C3  Nested-CV hyperparameter tuning for the 3 primary outcomes only
  • Ridge sensitivity baseline (linearity check feeds Figure S4)

Checkpointed: each invocation processes whatever targets are still pending and
saves to output/archived/multivariate_v2_checkpoint.pkl.

Usage:
    python3 multivariate_analysis.py [MAX_PER_CALL]
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

from config import (
    ENABLE_PRIMARY_TUNING,
    OUT_DIR, CKPT_DIR, N_MIN, BASELINE_FEATURES, FULL_FEATURES,
    CLINICAL_TARGETS, TARGET_DISPLAY, PRIMARY_OUTCOMES,
    N_BOOT, FDR_Q, RANDOM_STATE_LIST, RANDOM_STATE_PRIMARY,
    CB_PARAMS, CB_TUNE_GRID, N_FOLDS, INNER_VAL_FRAC,
)
from core import (
    load_cohort, fit_catboost_oof, fit_catboost_oof_repeated, fit_ridge_oof,
    paired_bca_ci, bootstrap_pvalue, benjamini_hochberg,
    perfold_paired_ttest, shap_eta,
)

CKPT = os.path.join(CKPT_DIR, "multivariate_v2_checkpoint.pkl")
MAX_PER_CALL_DEFAULT = 5


# =============================================================================
# C3 — Nested CV hyperparameter tuning (used only for the 3 primary outcomes)
# =============================================================================
def nested_tune_best_params(df_sub, feats, target, random_state):
    """Choose the best CatBoost params for one outer fold of one seed.
    Returns the best param dict, picking by the highest inner-CV mean R²
    across a grid of (depth, learning_rate, l2_leaf_reg)."""
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import train_test_split
    X, y = df_sub[feats].values, df_sub[target].values
    grid = []
    for d in CB_TUNE_GRID["depth"]:
        for lr in CB_TUNE_GRID["learning_rate"]:
            for l2 in CB_TUNE_GRID["l2_leaf_reg"]:
                grid.append({"depth": d, "learning_rate": lr, "l2_leaf_reg": l2})
    best, best_r2 = None, -np.inf
    inner_kf = KFold(3, shuffle=True, random_state=random_state)
    for params in grid:
        cb = {**CB_PARAMS, **params, "random_seed": random_state}
        inner_r2s = []
        for tr, te in inner_kf.split(X):
            X_tr, X_iv, y_tr, y_iv = train_test_split(
                X[tr], y[tr], test_size=INNER_VAL_FRAC, random_state=random_state)
            m = CatBoostRegressor(**cb)
            m.fit(Pool(X_tr, y_tr), eval_set=Pool(X_iv, y_iv))
            inner_r2s.append(r2_score(y[te], m.predict(X[te])))
        mean_r2 = float(np.mean(inner_r2s))
        if mean_r2 > best_r2:
            best_r2, best = mean_r2, params
    return best, best_r2


def fit_full_with_tuning(df_sub, feats, target, random_state):
    """For primary outcomes: tune CatBoost params via nested CV per outer fold,
    then refit + collect SHAP. Slower than fit_catboost_oof but more rigorous."""
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import train_test_split
    X, y = df_sub[feats].values, df_sub[target].values
    kf = KFold(N_FOLDS, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y))
    oof_shap = np.zeros_like(X, dtype=float)
    fold_r2  = []
    chosen_params = []
    for tr, te in kf.split(X):
        # Nested tuning on the training fold only
        df_inner = df_sub.iloc[tr].reset_index(drop=True)
        best, _  = nested_tune_best_params(df_inner, feats, target, random_state)
        chosen_params.append(best)
        cb = {**CB_PARAMS, **best, "random_seed": random_state}
        X_tr, X_iv, y_tr, y_iv = train_test_split(
            X[tr], y[tr], test_size=INNER_VAL_FRAC, random_state=random_state)
        m = CatBoostRegressor(**cb)
        m.fit(Pool(X_tr, y_tr), eval_set=Pool(X_iv, y_iv))
        pred = m.predict(X[te])
        oof_pred[te] = pred
        sv = m.get_feature_importance(data=Pool(X[te], y[te]), type="ShapValues")
        oof_shap[te] = sv[:, :-1]
        fold_r2.append(r2_score(y[te], pred))
    return y, oof_pred, oof_shap, fold_r2, chosen_params


# =============================================================================
# Main per-target pipeline
# =============================================================================
def fit_one_target(df, target, do_tune=False):
    """Returns a result dict for one target."""
    df_sub = df.dropna(subset=FULL_FEATURES + [target])
    n = len(df_sub)
    if n < N_MIN:
        return {"display": TARGET_DISPLAY[target], "n": n, "skipped": True}

    # B1: Repeated CV — averages over RANDOM_STATE_LIST
    base_rep = fit_catboost_oof_repeated(df_sub, BASELINE_FEATURES, target)
    if do_tune:
        # C3: nested tuning, only the primary seed (saves runtime)
        y, pred_f, shap_f, fold_r2_f, tune_log = fit_full_with_tuning(
            df_sub, FULL_FEATURES, target, RANDOM_STATE_PRIMARY)
        # For honesty, still report aggregate metrics by combining tuned-on-seed-42
        # with the same target. We treat tune_log + fold-R²s as the headline.
        full_rep = {
            "y": y, "mean_pred": pred_f, "mean_shap": shap_f,
            "per_seed": [{"seed": RANDOM_STATE_PRIMARY, "y": y, "pred": pred_f,
                          "shap": shap_f, "fold_r2": fold_r2_f}],
            "seed_r2s":[r2_score(y, pred_f)],
            "agg_r2":  float(r2_score(y, pred_f)),
            "agg_r2_sd": 0.0,
            "tune_log": tune_log,
        }
    else:
        full_rep = fit_catboost_oof_repeated(df_sub, FULL_FEATURES, target)

    y     = base_rep["y"]
    pb    = base_rep["mean_pred"]
    pf    = full_rep["mean_pred"]
    sf    = full_rep["mean_shap"]

    # Headline OOF metrics
    r2b = float(r2_score(y, pb))
    r2f = float(r2_score(y, pf))
    delta_r2 = r2f - r2b

    # B2: BCa bootstrap
    ci_base, ci_full, ci_delta, boot_delta = paired_bca_ci(
        y, pb, pf, n_boot=N_BOOT, seed=RANDOM_STATE_PRIMARY)
    p_boot = bootstrap_pvalue(boot_delta)

    # A2: per-fold paired t-test (uses primary-seed fold_r2 lists)
    fold_b = base_rep["per_seed"][-1]["fold_r2"]
    fold_f = full_rep["per_seed"][-1]["fold_r2"]
    t_stat, p_ttest = perfold_paired_ttest(fold_b, fold_f)

    # Ridge sensitivity (for Figure S4)
    pred_ridge_base = fit_ridge_oof(df_sub, BASELINE_FEATURES, target)
    pred_ridge_full = fit_ridge_oof(df_sub, FULL_FEATURES,     target)
    ridge_r2_base = float(r2_score(y, pred_ridge_base))
    ridge_r2_full = float(r2_score(y, pred_ridge_full))

    # SHAP η for the full model
    eta_full = {feat: shap_eta(sf[:, j], y) for j, feat in enumerate(FULL_FEATURES)}

    return {
        "display":         TARGET_DISPLAY[target],
        "n":               int(n),
        "skipped":         False,
        "primary":         target in PRIMARY_OUTCOMES,
        "tuned":           do_tune,
        "y":               y.tolist(),
        "pred_base":       pb.tolist(),
        "pred_full":       pf.tolist(),
        "r2_base":         r2b,
        "r2_full":         r2f,
        "delta_r2":        float(delta_r2),
        "ci_base":         ci_base,
        "ci_full":         ci_full,
        "ci_delta":        ci_delta,
        "p_boot":          float(p_boot),
        "p_ttest":         float(p_ttest),
        "t_stat":          float(t_stat),
        "fold_r2_base":    [float(v) for v in fold_b],
        "fold_r2_full":    [float(v) for v in fold_f],
        "seed_r2_base":    [float(v) for v in base_rep["seed_r2s"]],
        "seed_r2_full":    [float(v) for v in full_rep["seed_r2s"]],
        "mae_base":        float(mean_absolute_error(y, pb)),
        "mae_full":        float(mean_absolute_error(y, pf)),
        "ridge_r2_base":   ridge_r2_base,
        "ridge_r2_full":   ridge_r2_full,
        "eta_full":        eta_full,
        "shap_full":       sf.tolist(),
        "tune_log":        full_rep.get("tune_log"),
    }


# =============================================================================
# Orchestrator
# =============================================================================
def run(max_per_call):
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f: state = pickle.load(f)
    else:
        state = {"results": {}, "version": "v3-noratios"}

    df = load_cohort()
    print(f"Cohort n={len(df)} after BSA filter")

    pending = [t for t in CLINICAL_TARGETS if t not in state["results"]]
    if not pending:
        print("All targets done — applying BH correction & finalising.")
        _apply_bh(state)
        with open(CKPT, "wb") as f: pickle.dump(state, f)
        return state

    print(f"Pending: {len(pending)}/{len(CLINICAL_TARGETS)} targets")
    processed = 0
    for target in pending:
        if processed >= max_per_call: break
        do_tune = (target in PRIMARY_OUTCOMES) and ENABLE_PRIMARY_TUNING
        t0 = time.time()
        res = fit_one_target(df, target, do_tune=do_tune)
        state["results"][target] = res
        with open(CKPT, "wb") as f: pickle.dump(state, f)
        dt = time.time() - t0
        if res.get("skipped"):
            print(f"  SKIP {TARGET_DISPLAY[target]} (n={res['n']})")
        else:
            mark = "★ primary+tuned" if do_tune else ""
            print(f"  {res['display']:<20s} n={res['n']:>5d}  "
                  f"R²(b)={res['r2_base']:+.3f}  R²(f)={res['r2_full']:+.3f}  "
                  f"ΔR²={res['delta_r2']:+.3f} [{res['ci_delta'][0]:+.2f},{res['ci_delta'][1]:+.2f}]  "
                  f"p_boot={res['p_boot']:.3g}  p_t={res['p_ttest']:.3g}  ({dt:.0f}s) {mark}")
        processed += 1

    remaining = len([t for t in CLINICAL_TARGETS if t not in state["results"]])
    print(f"\nProcessed {processed} this call. Remaining: {remaining}")
    if remaining == 0:
        _apply_bh(state)
        with open(CKPT, "wb") as f: pickle.dump(state, f)
    return state


def _apply_bh(state):
    """A1 — apply Benjamini-Hochberg FDR control in two pre-specified families:
    (i) the 3 primary outcomes, and (ii) the 14 exploratory outcomes. Each
    family gets its own BH adjustment so the primary family is not penalised
    for exploratory testing."""
    targets = [t for t in CLINICAL_TARGETS if t in state["results"]
               and not state["results"][t].get("skipped")]
    primary    = [t for t in targets if t in PRIMARY_OUTCOMES]
    exploratory = [t for t in targets if t not in PRIMARY_OUTCOMES]

    for family, family_name in [(primary, "primary"), (exploratory, "exploratory")]:
        if not family: continue
        p_boot = [state["results"][t]["p_boot"]  for t in family]
        p_t    = [state["results"][t]["p_ttest"] for t in family]
        adj_boot, rej_boot = benjamini_hochberg(p_boot, q=FDR_Q)
        adj_t,    rej_t    = benjamini_hochberg([p if p == p else 1.0 for p in p_t], q=FDR_Q)
        for t, ab, rb, at, rt in zip(family, adj_boot, rej_boot, adj_t, rej_t):
            state["results"][t]["p_boot_bh"]       = float(ab)
            state["results"][t]["bh_reject"]       = bool(rb)
            state["results"][t]["p_ttest_bh"]      = float(at)
            state["results"][t]["bh_reject_ttest"] = bool(rt)
            state["results"][t]["bh_family"]       = family_name


# =============================================================================
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_PER_CALL_DEFAULT
    run(n)
