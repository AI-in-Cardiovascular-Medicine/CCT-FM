"""Render the main figures from the v2 checkpoints.

  Figure main (incremental value)  — 2-bar, BH-FDR-controlled
  SuppFig_Ya (pairwise SHAP-η, unadjusted)
  SuppFig_Yb (pairwise SHAP-η, adjusted)
  SuppFig_Yc (full-model joint SHAP-η)

The three SHAP-η panels (Y-a, Y-b, Y-c) now share a UNIFIED colour scale so
they are directly visually comparable. The scale is set to the smallest
"round 0.05" value that contains the maximum η across all three matrices.
Cells previously capped at 0.40 in Y-c (e.g. BSA → BMI at η ≈ 0.74) are now
shown at their true value, which is the more honest representation of the
BSA → BMI algebraic tautology.

Run after pairwise_analysis.py and multivariate_analysis.py have completed.
"""
import os, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from config import (
    RANDOM_STATE_LIST,
    OUT_DIR, CKPT_DIR, CLINICAL_TARGETS, TARGET_DISPLAY,
    INDEXED_VOLUMES, DEMOGRAPHICS, RAW_TO_INDEXED, RATIOS,
    FEATURE_DISPLAY, PRIMARY_OUTCOMES,
)
from figures import (
    apply_rcparams, black_box, cm, save_triplet,
    COL_GREY, COL_RED, COL_DARKR, COL_FLAG, COL_ORANGE,
    render_grouped_heatmap,
)

CKPT_PAIR  = os.path.join(CKPT_DIR, "pairwise_v2_checkpoint.pkl")
CKPT_MULTI = os.path.join(CKPT_DIR, "multivariate_v2_checkpoint.pkl")


# =============================================================================
# Supp Fig Y panels a, b, c — unified colour scale across all three
# =============================================================================
PAIR_ROW_GROUPS = [
    ("Demographics",    DEMOGRAPHICS),
    ("Indexed volumes", INDEXED_VOLUMES),
]
PAIR_GROUP_COLOR  = {"Demographics": COL_ORANGE, "Indexed volumes": COL_RED}

MULTI_ROW_GROUPS = [
    ("Demographics",         DEMOGRAPHICS),
    ("Substructure volumes", list(RAW_TO_INDEXED.keys())),
]
MULTI_GROUP_COLOR = {"Demographics": COL_ORANGE, "Substructure volumes": COL_RED}


def _build_pairwise_matrix(cfg_res, feature_order):
    """Build (n_features × n_targets) η matrix and target ordering for one
    pairwise configuration ('adjusted' or 'unadjusted')."""
    done = [t for t in CLINICAL_TARGETS if t in cfg_res and
            any(v.get("shap_eta") is not None for v in cfg_res[t].values())]

    def max_eta(t):
        return max((cfg_res[t][f].get("shap_eta", 0)
                    for f in feature_order if f in cfg_res[t]), default=0)
    sort_targets = sorted(done, key=max_eta, reverse=True)

    eta_mat = np.zeros((len(feature_order), len(sort_targets)))
    for j, t in enumerate(sort_targets):
        d = cfg_res[t]
        for i, f in enumerate(feature_order):
            eta_mat[i, j] = d.get(f, {}).get("shap_eta", 0.0)
    return eta_mat, sort_targets


def _build_joint_matrix(multi_res, feature_order):
    done = [t for t in CLINICAL_TARGETS if t in multi_res
            and not multi_res[t].get("skipped")]
    sort_targets = sorted(done, key=lambda t: -multi_res[t]["r2_full"])
    eta_mat = np.zeros((len(feature_order), len(sort_targets)))
    for j, t in enumerate(sort_targets):
        eta = multi_res[t]["eta_full"]
        for i, f in enumerate(feature_order):
            eta_mat[i, j] = eta.get(f, 0.0)
    return eta_mat, sort_targets


def render_supp_fig_Y():
    """Render Supp Fig Y panels a, b, c with a UNIFIED colour scale.

    No vmax_cap is applied to panel c. The BSA → BMI cell sits at its true
    value (η ≈ 0.74), which is the honest representation of the BSA-to-BMI
    algebraic tautology. All three panels share vmax = max-η across panels,
    rounded up to the nearest 0.05, so the colour intensity in any cell is
    directly comparable between Y-a, Y-b, and Y-c.
    """
    with open(CKPT_PAIR, "rb") as f:  pair_state  = pickle.load(f)
    with open(CKPT_MULTI, "rb") as f: multi_state = pickle.load(f)

    pair_feature_order  = [f for _, fs in PAIR_ROW_GROUPS  for f in fs]
    multi_feature_order = [f for _, fs in MULTI_ROW_GROUPS for f in fs]

    pair_unadj_mat, pair_unadj_targets = _build_pairwise_matrix(
        pair_state["results"]["unadjusted"], pair_feature_order)
    pair_adj_mat, pair_adj_targets = _build_pairwise_matrix(
        pair_state["results"]["adjusted"], pair_feature_order)
    multi_mat, multi_targets = _build_joint_matrix(
        multi_state["results"], multi_feature_order)

    # ── Unified vmax across all three matrices ──
    all_max = max(pair_unadj_mat.max(), pair_adj_mat.max(), multi_mat.max())
    unified_vmax = float(np.ceil(all_max * 20) / 20)  # round up to nearest 0.05
    print(f"  Unified SHAP-η colour scale: vmax = {unified_vmax:.2f}  "
          f"(max-η observed = {all_max:.3f}; from {('joint Y-c' if multi_mat.max() == all_max else 'pairwise Y-a/Y-b')})")

    # ── Render Y-a (unadjusted) ──
    render_grouped_heatmap(
        pair_unadj_mat, PAIR_ROW_GROUPS,
        [TARGET_DISPLAY[t] for t in pair_unadj_targets],
        OUT_DIR, "SuppFig_Ya_pairwise_SHAP_eta_unadjusted",
        feature_display=FEATURE_DISPLAY,
        group_color=PAIR_GROUP_COLOR,
        fig_h_cm=11.5,
        vmax_set=unified_vmax,
    )

    # ── Render Y-b (adjusted) ──
    render_grouped_heatmap(
        pair_adj_mat, PAIR_ROW_GROUPS,
        [TARGET_DISPLAY[t] for t in pair_adj_targets],
        OUT_DIR, "SuppFig_Yb_pairwise_SHAP_eta_adjusted",
        feature_display=FEATURE_DISPLAY,
        group_color=PAIR_GROUP_COLOR,
        fig_h_cm=11.5,
        vmax_set=unified_vmax,
    )

    # ── Render Y-c (joint full model, NO CAP) ──
    render_grouped_heatmap(
        multi_mat, MULTI_ROW_GROUPS,
        [TARGET_DISPLAY[t] for t in multi_targets],
        OUT_DIR, "SuppFig_Yc_full_model_SHAP_eta",
        feature_display=FEATURE_DISPLAY,
        group_color=MULTI_GROUP_COLOR,
        fig_h_cm=10.5,
        vmax_set=unified_vmax,
    )

    # ── Excel: pairwise η tables ──
    df_a = _eta_df(pair_state, "adjusted",   pair_feature_order)
    df_u = _eta_df(pair_state, "unadjusted", pair_feature_order)
    xlsx = os.path.join(OUT_DIR, "Pairwise_SHAP_Results.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as wr:
        df_a.to_excel(wr, sheet_name="adjusted_SHAP_eta")
        df_u.to_excel(wr, sheet_name="unadjusted_SHAP_eta")
    print(f"  Saved: {xlsx}")


def _eta_df(state, cfg, feature_order):
    cfg_res = state["results"][cfg]
    rows = []
    for f in feature_order:
        row = {"Feature": FEATURE_DISPLAY.get(f, f)}
        for t in CLINICAL_TARGETS:
            if t not in cfg_res: continue
            row[TARGET_DISPLAY[t]] = cfg_res[t].get(f, {}).get("shap_eta", np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Feature")


# =============================================================================
# Figure main — Incremental-value bar chart with BH-FDR significance
# =============================================================================
def render_fig3c():
    apply_rcparams()
    with open(CKPT_MULTI, "rb") as f: state = pickle.load(f)
    res = state["results"]

    # BMI is removed from this figure: its baseline R² reflects the algebraic
    # BSA → BMI link, not a cardiac-volume signal. BMI quantitative result is
    # in Supp Fig X panel b and in incremental_value_results.xlsx.
    done = [t for t in CLINICAL_TARGETS
            if t in res and not res[t].get("skipped")
            and t != "Base_BMI"]
    sort_targets = sorted(done, key=lambda t: -res[t]["r2_full"])
    n_t = len(sort_targets)

    # Defensive: ensure BH has been applied to all displayed targets.
    for t in sort_targets:
        assert "p_boot_bh" in res[t], (
            f"target {t} lacks p_boot_bh — _apply_bh was not run; "
            "rerun multivariate_analysis.run() to apply BH before rendering."
        )

    fig, ax = plt.subplots(figsize=(cm(18), cm(13.5)))
    y_pos = np.arange(n_t); bar_h = 0.36

    arr  = lambda k: np.array([res[t][k] for t in sort_targets])
    ci2  = lambda k, i: np.array([res[t][k][i] for t in sort_targets])
    r2b, r2f = arr("r2_base"), arr("r2_full")
    b_lo, b_hi = ci2("ci_base", 0), ci2("ci_base", 1)
    f_lo, f_hi = ci2("ci_full", 0), ci2("ci_full", 1)
    d, d_lo, d_hi = arr("delta_r2"), ci2("ci_delta", 0), ci2("ci_delta", 1)
    p_boot_bh = arr("p_boot_bh")
    p_t_bh    = arr("p_ttest_bh")
    bh_reject = arr("bh_reject")

    r2b_d = np.clip(r2b, 0, None); r2f_d = np.clip(r2f, 0, None)
    err_b = [np.clip(r2b_d - np.clip(b_lo, 0, None), 0, None),
             np.clip(np.clip(b_hi, 0, None) - r2b_d, 0, None)]
    err_f = [np.clip(r2f_d - np.clip(f_lo, 0, None), 0, None),
             np.clip(np.clip(f_hi, 0, None) - r2f_d, 0, None)]

    ax.barh(y_pos + bar_h/2, r2b_d, height=bar_h, color=COL_GREY,
            edgecolor='black', linewidth=0.5,
            xerr=err_b, error_kw={"elinewidth":0.8,"capsize":1.8,"ecolor":"black"},
            label="Baseline (Age + Sex + BSA)")
    ax.barh(y_pos - bar_h/2, r2f_d, height=bar_h, color=COL_RED,
            edgecolor='black', linewidth=0.5,
            xerr=err_f, error_kw={"elinewidth":0.8,"capsize":1.8,"ecolor":"black"},
            label="Full   (Baseline + segmentation)")

    # (No significance divider line: rows are sorted by full-model R², not by
    # FDR significance, so a single horizontal divider cannot cleanly separate
    # significant from non-significant targets. FDR significance is conveyed by
    # the asterisks/daggers on each target instead.)

    x_pair_max = np.maximum(np.clip(b_hi, 0, None), np.clip(f_hi, 0, None))
    xlim_max = min(0.95, max(0.80, x_pair_max.max() + 0.27))
    for i, t in enumerate(sort_targets):
        sig = ""
        if bh_reject[i]:
            sig = "**" if p_boot_bh[i] < 0.01 else "*"
        bold = bh_reject[i]
        if bh_reject[i] and p_t_bh[i] < 0.05:
            sig = sig + "†"
        txt = f"ΔR² = {d[i]:+.3f}{sig}"
        ax.text(x_pair_max[i] + 0.012, y_pos[i], txt,
                fontsize=7, va='center', ha='left',
                color=COL_DARKR if bold else 'black',
                fontweight='bold' if bold else 'normal')

    # Y labels with mathtext star / triangle glyphs
    ytick_labels = []
    for t in sort_targets:
        lbl = f"{res[t]['display']}  (n = {res[t]['n']:,})"
        if res[t]["display"] == "BMI":
            lbl += r"  $\blacktriangle$"
        if t in PRIMARY_OUTCOMES:
            lbl = r"$\bigstar$  " + lbl
        ytick_labels.append(lbl)
    ax.set_yticks(y_pos); ax.set_yticklabels(ytick_labels, fontsize=8)
    for tlab, t in zip(ax.get_yticklabels(), sort_targets):
        if res[t]["display"] == "BMI":
            tlab.set_color(COL_FLAG); tlab.set_fontstyle('italic')

    ax.invert_yaxis()
    ax.set_xlabel(
        f'Out-of-fold R²  (5-fold CV'
        f'{("" if len(RANDOM_STATE_LIST)==1 else f" × {len(RANDOM_STATE_LIST)} seeds")}'
        f'; error bars = BCa 95 % CI; seed = '
        f'{RANDOM_STATE_LIST[0] if len(RANDOM_STATE_LIST)==1 else "0,7,13,23,42"})',
        fontsize=9)
    ax.set_xlim(0, xlim_max)
    ax.xaxis.grid(True, alpha=0.30, linewidth=0.5, color='#888')
    ax.set_axisbelow(True)
    black_box(ax); ax.tick_params(axis='y', length=0, pad=3)

    leg = ax.legend(loc='lower right', frameon=True, fancybox=False,
                    edgecolor='black', fontsize=8, handlelength=1.5,
                    handleheight=0.9, borderpad=0.4, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.8)

    ax.text(1.0, -0.135,
            r"$\bigstar$ pre-specified primary outcome     "
            "*  BH-FDR-controlled bootstrap p ≤ 0.05     "
            "** ≤ 0.01     "
            "†  also significant by per-fold paired t-test",
            transform=ax.transAxes, fontsize=6.4, ha='right', va='top', color='black')

    save_triplet(fig, OUT_DIR, "Figure_main_incremental_value")

    # ── Excel — full performance table with p<0.001 floor ──
    # With N_BOOT = 1,000 the smallest non-zero bootstrap p the procedure can
    # resolve is 1/1000 = 0.001. Replace p == 0 with the string "<0.001" so
    # the table does not mislead the reader.
    perf = []
    for t in sort_targets:
        r = res[t]
        perf.append({
            "Target":              r["display"],
            "Primary":             r["primary"],
            "Tuned":               r.get("tuned", False),
            "N":                   r["n"],
            "R2_baseline":         r["r2_base"],
            "R2_baseline_CI_low":  r["ci_base"][0],
            "R2_baseline_CI_high": r["ci_base"][1],
            "R2_full":             r["r2_full"],
            "R2_full_CI_low":      r["ci_full"][0],
            "R2_full_CI_high":     r["ci_full"][1],
            "Delta_R2":            r["delta_r2"],
            "Delta_R2_CI_low":     r["ci_delta"][0],
            "Delta_R2_CI_high":    r["ci_delta"][1],
            "p_bootstrap":         _fmt_pval(r["p_boot"]),
            "p_bootstrap_BH":      _fmt_pval(r["p_boot_bh"]),
            "p_perfold_ttest":     _fmt_pval(r["p_ttest"]),
            "p_perfold_ttest_BH":  _fmt_pval(r["p_ttest_bh"]),
            "BH_reject":           r["bh_reject"],
            "MAE_baseline":        r["mae_base"],
            "MAE_full":            r["mae_full"],
            "Ridge_R2_baseline":   r["ridge_r2_base"],
            "Ridge_R2_full":       r["ridge_r2_full"],
        })
    pd.DataFrame(perf).to_excel(
        os.path.join(OUT_DIR, "incremental_value_results.xlsx"), index=False)
    print(f"  Saved: incremental_value_results.xlsx")


def _fmt_pval(p, floor=1e-3):
    """Format a bootstrap or t-test p-value for the export table.
    Replace exact 0 with '<0.001' (the resolution floor at N_BOOT = 1,000).
    NaN passes through as the string 'NA'."""
    if p is None or (isinstance(p, float) and p != p):  # NaN check
        return "NA"
    if p == 0:
        return f"<{floor:g}"
    return f"{p:.4g}"


# Back-compat alias so existing imports keep working
def render_fig2_and_S1():
    """Deprecated: use render_supp_fig_Y() — kept for backwards compatibility."""
    render_supp_fig_Y()


def render_fig3d():
    """Deprecated: rolled into render_supp_fig_Y() (joint matrix renders Y-c)."""
    render_supp_fig_Y()


if __name__ == "__main__":
    print("── Supp Fig Y (a, b, c) — unified colour scale ──")
    render_supp_fig_Y()
    print("\n── Figure main (incremental value) ──")
    render_fig3c()
    print("\nDone.")
