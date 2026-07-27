"""Figure 3c v3 - three-bar visual of out-of-fold R^2 for each clinical target."""
import os, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from config import (
    CKPT_DIR, OUT_DIR, CLINICAL_TARGETS, TARGET_DISPLAY, PRIMARY_OUTCOMES,
    RANDOM_STATE_LIST,
)
from figures import (
    apply_rcparams, black_box, cm, save_triplet,
    COL_GREY, COL_GREY2, COL_RED, COL_FLAG, COL_ORANGE, COL_DARKR,
)

CKPT = os.path.join(CKPT_DIR, "multivariate_v2_checkpoint.pkl")


def render():
    apply_rcparams()
    with open(CKPT, "rb") as f: state = pickle.load(f)
    res = state["results"]
    done = [t for t in CLINICAL_TARGETS
            if t in res and not res[t].get("skipped")
            and "r2_vol_only" in res[t]]
    sort_targets = sorted(done, key=lambda t: -res[t]["r2_full"])
    n_t = len(sort_targets)

    fig, ax = plt.subplots(figsize=(cm(18), cm(24)))
    y_pos = 1.6 * np.arange(n_t)
    bar_h = 0.34
    offset = bar_h

    arr  = lambda k: np.array([res[t][k] for t in sort_targets])
    ci2  = lambda k, i: np.array([res[t][k][i] for t in sort_targets])

    r2_b   = np.clip(arr("r2_base"),     0, None)
    r2_v   = np.clip(arr("r2_vol_only"), 0, None)
    r2_f   = np.clip(arr("r2_full"),     0, None)
    err_b  = [np.clip(r2_b - np.clip(ci2("ci_base",     0), 0, None), 0, None),
              np.clip(np.clip(ci2("ci_base",     1), 0, None) - r2_b, 0, None)]
    err_v  = [np.clip(r2_v - np.clip(ci2("ci_vol_only", 0), 0, None), 0, None),
              np.clip(np.clip(ci2("ci_vol_only", 1), 0, None) - r2_v, 0, None)]
    err_f  = [np.clip(r2_f - np.clip(ci2("ci_full",     0), 0, None), 0, None),
              np.clip(np.clip(ci2("ci_full",     1), 0, None) - r2_f, 0, None)]

    ax.barh(y_pos - offset, r2_b, height=bar_h, color=COL_GREY,   edgecolor='black',
            linewidth=0.4, xerr=err_b,
            error_kw={"elinewidth":0.6,"capsize":1.4,"ecolor":"black"},
            label="Baseline    (Age + Sex + BSA)")
    ax.barh(y_pos,          r2_v, height=bar_h, color=COL_ORANGE, edgecolor='black',
            linewidth=0.4, xerr=err_v,
            error_kw={"elinewidth":0.6,"capsize":1.4,"ecolor":"black"},
            label="Volume-only (8 raw substructure volumes)")
    ax.barh(y_pos + offset, r2_f, height=bar_h, color=COL_RED,    edgecolor='black',
            linewidth=0.4, xerr=err_f,
            error_kw={"elinewidth":0.6,"capsize":1.4,"ecolor":"black"},
            label="Full        (Baseline + 8 substructure volumes)")

    x_pair_max = np.maximum.reduce([
        np.clip(ci2("ci_base",     1), 0, None),
        np.clip(ci2("ci_vol_only", 1), 0, None),
        np.clip(ci2("ci_full",     1), 0, None),
    ])
    # Leave room for three inline R^2 labels at the right of each group.
    xlim_max = min(1.05, max(0.95, x_pair_max.max() + 0.22))

    # Three color-matched inline R^2 labels per target, one per bar, all at the
    # same x (just right of the longest bar/CI). The trio is fanned out vertically
    # — top label nudged up, bottom label nudged down — so the three never overlap
    # each other while staying visually grouped and clear of the neighbouring targets.
    LBL_FS = 7.8
    LBL_DX = 0.012
    LBL_SPREAD = 0.50   # vertical half-spread of the label trio (> bar offset 0.34)
    rows = [("r2_base",     -LBL_SPREAD, '#4a4a4a', 'normal'),
            ("r2_vol_only",  0.0,        '#8a4a2c', 'bold'),
            ("r2_full",      LBL_SPREAD,  COL_DARKR, 'normal')]
    for i in range(n_t):
        x_lbl = x_pair_max[i] + LBL_DX
        for k, dy, col, fw in rows:
            ax.text(x_lbl, y_pos[i] + dy,
                    rf"$R^2$ = {arr(k)[i]:+.3f}",
                    fontsize=LBL_FS, va='center', ha='left',
                    color=col, fontweight=fw)

    ytick_labels = []
    for t in sort_targets:
        lbl = f"{res[t]['display']}  (n = {res[t]['n']:,})"
        if res[t]["display"] == "BMI":
            lbl += r"  $\blacktriangle$"
        if t in PRIMARY_OUTCOMES:
            lbl = r"$\bigstar$  " + lbl
        ytick_labels.append(lbl)
    ax.set_yticks(y_pos); ax.set_yticklabels(ytick_labels, fontsize=10)
    for tlab, t in zip(ax.get_yticklabels(), sort_targets):
        if res[t]["display"] == "BMI":
            tlab.set_color(COL_FLAG); tlab.set_fontstyle('italic')
    ax.invert_yaxis()

    seed_lbl = (f"seed = {RANDOM_STATE_LIST[0]}" if len(RANDOM_STATE_LIST) == 1
                else f"{len(RANDOM_STATE_LIST)} seeds averaged")
    ax.set_xlabel('Out-of-fold R²', fontsize=11)
    ax.set_xlim(0, xlim_max)
    ax.xaxis.grid(True, alpha=0.30, linewidth=0.5, color='#888')
    ax.set_axisbelow(True)
    black_box(ax)
    ax.tick_params(axis='y', length=0, pad=3)

    leg = ax.legend(loc='lower right', frameon=True, fancybox=False,
                    edgecolor='black', fontsize=9.0, handlelength=1.5,
                    handleheight=0.9, borderpad=0.4, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.8)

    ax.text(1.0, -0.10,
            r"$\bigstar$ = pre-specified primary outcome   "
            r"$\blacktriangle$ = BMI: BSA $\to$ BMI tautology, not cardiac signal   "
            "CK / CK-MB shown on log-transformed scale   "
            "inter-chamber ratios excluded (algebraic transforms of raw volumes)\n"
            f"5-fold CV ({seed_lbl}); error bars = BCa 95 % CI; "
            "no formal significance testing between the three models",
            transform=ax.transAxes, fontsize=8.0, ha='right', va='top',
            color='black')

    save_triplet(fig, OUT_DIR, "SuppFig_Xa_three_models")


if __name__ == "__main__":
    render()
