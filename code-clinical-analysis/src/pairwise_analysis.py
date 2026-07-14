"""Pairwise SHAP η analysis (Figure 2 + Figure S1).

For every (predictor, target) pair fit a 5-fold CV CatBoost regression with
inner-validation early stopping and extract the SHAP η of the predictor.

Predictors:
  • 8 indexed substructure volumes (raw_volume / Base_BSA)
  • 3 demographics (Age, Sex, BSA)
  • 7 inter-chamber ratios (kept in the supplementary table; dropped from
    the displayed heatmap to keep the row count tractable)

Configs:
  • unadjusted  →  feature_set = [predictor]                  (1 feature)
  • adjusted    →  feature_set = [predictor + age + sex]      (with duplicate
                   handling when predictor IS age or sex)

Checkpointed to output/archived/pairwise_v2_checkpoint.pkl. Each call processes
up to MAX_PER_CALL (config, target) combinations. The script extends an
existing v1 checkpoint when present.

Usage:
    python3 pairwise_analysis.py [MAX_PER_CALL]
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from config import (
    OUT_DIR, CKPT_DIR, N_MIN, CLINICAL_TARGETS, TARGET_DISPLAY,
    INDEXED_VOLUMES, DEMOGRAPHICS, RATIOS, RAW_TO_INDEXED, FEATURE_DISPLAY,
)
from core import load_cohort, fit_catboost_oof, shap_eta

CKPT = os.path.join(CKPT_DIR, "pairwise_v2_checkpoint.pkl")
MAX_PER_CALL_DEFAULT = 4

CONFOUNDERS = ["Base_Age", "Base_Sex"]
PREDICTORS  = DEMOGRAPHICS + INDEXED_VOLUMES + RATIOS    # 18 predictors


def _feats_for(predictor, config_name):
    """Decide the feature list for one pairwise fit (with duplicate handling)."""
    if config_name == "unadjusted":
        return [predictor]
    # adjusted — include age + sex, but skip predictor if it would duplicate
    return [predictor] + [c for c in CONFOUNDERS if c != predictor]


def fit_one(df, predictor, target, config_name):
    extra = CONFOUNDERS if config_name == "adjusted" else []
    # Use the same dropna set for every predictor in a given (target, config)
    # so the η values for that column are computed on a common cohort.
    df_sub = df.dropna(subset=PREDICTORS + extra + [target])
    if len(df_sub) < N_MIN:
        return {"n": len(df_sub), "skipped": True}
    feats = _feats_for(predictor, config_name)
    # Degenerate guard: if every selected feature is constant in this cohort, a
    # model cannot use it and CatBoost errors ("all features constant/ignored").
    # This occurs only for the unadjusted Sex predictor inside a single-sex
    # stratum; the SHAP-η of a constant predictor is 0 by definition, so we
    # short-circuit. In the full cohort Sex varies, so this never triggers and
    # the canonical run is unchanged.
    if np.all(df_sub[feats].values.var(axis=0) < 1e-12):
        return {"n": int(len(df_sub)), "skipped": False, "shap_eta": 0.0,
                "feats": feats, "degenerate_constant": True}
    y, _, shap_arr, _ = fit_catboost_oof(df_sub, feats, target)
    eta = shap_eta(shap_arr[:, 0], y)
    return {"n": int(len(df_sub)), "skipped": False,
            "shap_eta": float(eta), "feats": feats}


def run(max_per_call):
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f: state = pickle.load(f)
    else:
        state = {
            "config": {
                "PREDICTORS": PREDICTORS, "CONFOUNDERS": CONFOUNDERS,
                "TARGETS": CLINICAL_TARGETS, "TARGET_DISPLAY": TARGET_DISPLAY,
                "FEATURE_DISPLAY": FEATURE_DISPLAY,
            },
            "results": {"unadjusted": {}, "adjusted": {}},
        }

    df = load_cohort()
    print(f"Cohort n={len(df)} after BSA filter")

    pending = []
    for cfg_name in ("adjusted", "unadjusted"):
        for t in CLINICAL_TARGETS:
            if t in state["results"][cfg_name]: continue
            pending.append((cfg_name, t))
    if not pending:
        print("All combos done.")
        return state

    print(f"Pending (target, config) combos: {len(pending)}")
    processed = 0
    for cfg_name, target in pending:
        if processed >= max_per_call: break
        t0 = time.time()
        per_pred = {}
        for predictor in PREDICTORS:
            per_pred[predictor] = fit_one(df, predictor, target, cfg_name)
        state["results"][cfg_name][target] = per_pred
        with open(CKPT, "wb") as f: pickle.dump(state, f)
        dt = time.time() - t0
        if all(v.get("skipped") for v in per_pred.values()):
            print(f"  SKIP [{cfg_name}] {TARGET_DISPLAY[target]} (n insufficient)")
        else:
            top = max(per_pred, key=lambda p: per_pred[p].get("shap_eta", 0))
            top_eta = per_pred[top]["shap_eta"]
            n = per_pred[top]["n"]
            print(f"  [{cfg_name}] {TARGET_DISPLAY[target]:<18s} n={n:>5d}  "
                  f"top η={top_eta:.3f} ({FEATURE_DISPLAY.get(top, top)})  ({dt:.0f}s)")
        processed += 1

    rem = sum(1 for cfg in ("unadjusted", "adjusted")
              for t in CLINICAL_TARGETS if t not in state["results"][cfg])
    print(f"\nProcessed {processed} this call. Remaining: {rem}")
    return state


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_PER_CALL_DEFAULT
    run(n)
