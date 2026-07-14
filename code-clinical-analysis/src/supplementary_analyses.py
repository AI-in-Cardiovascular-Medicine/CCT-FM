"""Supplementary analyses + figures (S4, S5, S6).

  S4  Linearity check     — CatBoost-Full R² vs RidgeCV-Full R² per target.
                            No new fitting — values are already in the
                            multivariate v2 checkpoint.
  S5  BMI sensitivity     — Four nested CatBoost models predicting BMI under
                            different feature sets. Disentangles BSA → BMI
                            tautology from any cardiac-volume contribution.
  S6  Calibration plots   — For the 3 primary outcomes: observed vs OOF
                            prediction with a locally-weighted line.

Usage:
    python3 supplementary_analyses.py
"""
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import linregress

from config import (
    OUT_DIR, CKPT_DIR, BASELINE_FEATURES, FULL_FEATURES, RAW_VOLUMES,
    DEMOGRAPHICS, PRIMARY_OUTCOMES, TARGET_DISPLAY, N_BOOT,
    RANDOM_STATE_PRIMARY, BSA_RANGE, N_MIN,
)
from core import load_cohort, fit_catboost_oof, paired_bca_ci
from figures import (
    apply_rcparams, black_box, cm, save_triplet,
    COL_GREY, COL_GREY2, COL_RED, COL_DARKR,
)

CKPT_MULTI = os.path.join(CKPT_DIR, "multivariate_v2_checkpoint.pkl")


# =============================================================================
# Figure S4 — Linearity check
# =============================================================================
def render_S4():
    apply_rcparams()
    with open(CKPT_MULTI, "rb") as f: state = pickle.load(f)
    rows = [(r["display"], r["ridge_r2_full"], r["r2_full"], r["n"])
            for r in state["results"].values() if not r.get("skipped")]
    rows.sort(key=lambda x: x[2], reverse=True)
    labels = [r[0] for r in rows]
    ridge  = np.array([r[1] for r in rows])
    cb     = np.array([r[2] for r in rows])
    ns     = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(cm(18), cm(13.5)))
    y_pos = np.arange(len(labels)); bar_h = 0.36
    ax.barh(y_pos + bar_h/2, np.clip(ridge, 0, None), height=bar_h,
            color=COL_GREY, edgecolor='black', linewidth=0.5,
            label="RidgeCV (linear)")
    ax.barh(y_pos - bar_h/2, np.clip(cb, 0, None), height=bar_h,
            color=COL_RED, edgecolor='black', linewidth=0.5,
            label="CatBoost (non-linear)")
    xlim_max = min(0.95, max(0.80, max(np.clip(cb,0,None).max(),
                                       np.clip(ridge,0,None).max()) + 0.22))
    for i, (_, rr, cc, _) in enumerate(rows):
        xpos = max(max(rr, 0), max(cc, 0)) + 0.012
        sig = "*" if (cc - rr) > 0.05 else ""
        bold = bool(sig)
        ax.text(xpos, y_pos[i], f"Δ = {cc-rr:+.3f}{sig}", fontsize=7,
                va='center', ha='left',
                color=COL_DARKR if bold else 'black',
                fontweight='bold' if bold else 'normal')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{lab}  (n = {n:,})" for lab, n in zip(labels, ns)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Out-of-fold R²  (Full feature set: Age + Sex + BSA + volumes)', fontsize=9)
    ax.set_xlim(0, xlim_max)
    ax.xaxis.grid(True, alpha=0.30, linewidth=0.5, color='#888'); ax.set_axisbelow(True)
    black_box(ax); ax.tick_params(axis='y', length=0, pad=3)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False,
                    edgecolor='black', fontsize=8, handlelength=1.5,
                    handleheight=0.9, borderpad=0.4)
    leg.get_frame().set_linewidth(0.8)
    ax.text(1.0, -0.12,
            "* CatBoost − Ridge > 0.05 (substantial non-linear gain)",
            transform=ax.transAxes, fontsize=6.8, ha='right', va='top', color='black')
    save_triplet(fig, OUT_DIR, "SuppFig_Za_linearity_check")


# =============================================================================
# Figure S5 — BMI sensitivity
# =============================================================================
def render_S5():
    apply_rcparams()
    df = load_cohort()
    target = "Base_BMI"
    ALL = DEMOGRAPHICS + RAW_VOLUMES        # full model: no ratios
    df_sub = df.dropna(subset=ALL + [target])
    n = len(df_sub)
    if n < N_MIN: return

    models = [
        ("M1: Age + Sex",                                       ["Base_Age", "Base_Sex"]),
        ("M2: Age + Sex + BSA  (baseline)",                     DEMOGRAPHICS),
        ("M3: Age + Sex + BSA + volumes  (full)",               ALL),
        ("M4: Age + Sex + volumes  (BSA removed)",
         ["Base_Age", "Base_Sex"] + RAW_VOLUMES),
    ]
    y_true = df_sub[target].values
    # Resumable compute: fit each model once and checkpoint it, so the fit can
    # survive the sandbox wall-clock cap. Re-running finishes any pending models.
    S5_CKPT = os.path.join(CKPT_DIR, "bmi_sensitivity_checkpoint.pkl")
    cache = pickle.load(open(S5_CKPT, "rb")) if os.path.exists(S5_CKPT) else {}
    print(f"BMI sensitivity (n = {n:,})")
    for label, feats in models:
        if label in cache: continue
        _, pred, _, _ = fit_catboost_oof(df_sub, feats, target)
        r2 = r2_score(y_true, pred)
        rng = np.random.default_rng(RANDOM_STATE_PRIMARY)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if np.var(y_true[idx]) < 1e-12: continue
            boots.append(r2_score(y_true[idx], pred[idx]))
        ci = np.percentile(boots, [2.5, 97.5]).tolist()
        cache[label] = (float(r2), [float(ci[0]), float(ci[1])])
        pickle.dump(cache, open(S5_CKPT, "wb"))
        print(f"  {label:60s} R²={r2:+.3f}  [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    if any(label not in cache for label, _ in models):
        print("  [S5] compute incomplete — re-run to finish, then it will plot."); return
    results = [(label, cache[label][0], cache[label][1]) for label, _ in models]

    fig, ax = plt.subplots(figsize=(cm(15), cm(5.7)))   # 40% shorter; bars therefore thinner
    y_pos = np.arange(len(results))
    vals  = np.array([r[1] for r in results])
    lows  = np.array([r[2][0] for r in results])
    highs = np.array([r[2][1] for r in results])
    err = [np.clip(vals - lows, 0, None), np.clip(highs - vals, 0, None)]
    COLS = [COL_GREY, COL_GREY2, COL_RED, COL_DARKR]
    ax.barh(y_pos, vals, height=0.55, color=COLS, edgecolor='black', linewidth=0.5,
            xerr=err, error_kw={"elinewidth":0.9,"capsize":2,"ecolor":"black"})
    # Anchor inline R² labels past the upper CI end (with cap-clearance) so
    # they never collide with the error-bar whisker.
    label_x = np.maximum(np.clip(highs, 0, None), np.clip(vals, 0, None)) + 0.018
    for i, (_, r2, _) in enumerate(results):
        ax.text(label_x[i], y_pos[i], f"R² = {r2:+.3f}",
                va='center', ha='left', fontsize=8, color='black')
    ax.set_yticks(y_pos); ax.set_yticklabels([r[0] for r in results], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f'Out-of-fold R² for predicting BMI  '
                  f'(n = {n:,};  error bars = bootstrap 95% CI)', fontsize=9)
    ax.set_xlim(0, max(0.95, label_x.max() + 0.10))
    ax.xaxis.grid(True, alpha=0.30, linewidth=0.5, color='#888'); ax.set_axisbelow(True)
    black_box(ax); ax.tick_params(axis='y', length=0, pad=3)
    ax.text(1.0, -0.40,
            "BSA is a sufficient summary of body-size for predicting BMI (M2 ≈ M3).\n"
            "Volumes carry the same information indirectly (M4 ≫ M1) but add nothing\n"
            "once BSA is in the model — redundant, not absent.",
            transform=ax.transAxes, fontsize=7.0, ha='right', va='top', color='black')
    save_triplet(fig, OUT_DIR, "SuppFig_Xb_BMI_sensitivity")


# =============================================================================
# Figure S6 — Calibration plots for primary outcomes (C4)
# =============================================================================
def render_S6():
    apply_rcparams()
    with open(CKPT_MULTI, "rb") as f: state = pickle.load(f)
    n_pri = len(PRIMARY_OUTCOMES)
    fig, axes = plt.subplots(1, n_pri, figsize=(cm(6 * n_pri), cm(6.5)), squeeze=False)
    axes = axes.flatten()
    for ax, target in zip(axes, PRIMARY_OUTCOMES):
        r = state["results"][target]
        y = np.array(r["y"])
        p = np.array(r["pred_full"])
        ax.scatter(p, y, s=4, alpha=0.5, color=COL_DARKR, edgecolor='none')
        # Identity line
        lo = min(y.min(), p.min()); hi = max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], '--', color='black', linewidth=0.8)
        # OLS line
        sl = linregress(p, y)
        xs = np.linspace(p.min(), p.max(), 50)
        ax.plot(xs, sl.intercept + sl.slope * xs, color=COL_RED, linewidth=1.2)
        ax.set_xlabel(f'Predicted {r["display"]}', fontsize=9)
        ax.set_ylabel(f'Observed {r["display"]}', fontsize=9)
        ax.text(0.04, 0.96,
                f"R² = {r['r2_full']:+.3f}\nslope = {sl.slope:.2f}\nintercept = {sl.intercept:.2f}",
                transform=ax.transAxes, fontsize=7, va='top', ha='left')
        ax.set_aspect('auto')
        black_box(ax)
    fig.suptitle('Calibration on out-of-fold predictions — primary outcomes',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_triplet(fig, OUT_DIR, "SuppFig_Zb_calibration_primaries")


if __name__ == "__main__":
    print("─── Figure S4 ───"); render_S4()
    print("\n─── Figure S5 ───"); render_S5()
    print("\n─── Figure S6 ───"); render_S6()
    print("Done.")
