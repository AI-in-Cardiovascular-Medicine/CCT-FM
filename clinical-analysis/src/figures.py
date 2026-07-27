"""Shared rcParams + palette + render helpers.

Single source of style truth for every figure in v2.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

DPI         = 500
COL_GREY    = '#C6D0D6'    # baseline / linear comparator
COL_GREY2   = '#B2B9BE'    # secondary baseline tier
COL_RED     = '#D98880'    # scientific red — full / signal
COL_DARKR   = '#B98079'    # dark red — significance annotation
COL_FLAG    = '#9C7672'    # very dark red — BMI tautology flag
COL_ORANGE  = '#EFB69F'    # demographics sidebar accent
CMAP_RED    = LinearSegmentedColormap.from_list("seg_red", ["#FCF1F0", "#D98880"])

def apply_rcparams():
    rcParams['font.family']      = 'sans-serif'
    rcParams['font.sans-serif']  = ['Arial Nova', 'Arial', 'Liberation Sans', 'DejaVu Sans']
    rcParams['font.size']        = 8
    rcParams['axes.titlesize']   = 9
    rcParams['axes.labelsize']   = 9
    rcParams['xtick.labelsize']  = 8
    rcParams['ytick.labelsize']  = 8
    rcParams['legend.fontsize']  = 8
    rcParams['axes.linewidth']   = 1.0
    rcParams['xtick.major.width']= 1.0
    rcParams['ytick.major.width']= 1.0
    rcParams['xtick.major.size'] = 3.5
    rcParams['ytick.major.size'] = 3.5
    rcParams['xtick.direction']  = 'out'
    rcParams['ytick.direction']  = 'out'
    rcParams['lines.linewidth']  = 1.0
    rcParams['figure.dpi']       = DPI
    rcParams['savefig.dpi']      = DPI
    rcParams['savefig.bbox']     = 'tight'
    rcParams['savefig.pad_inches'] = 0.06
    rcParams['pdf.fonttype']     = 42
    rcParams['ps.fonttype']      = 42
    rcParams['svg.fonttype']     = 'none'

def black_box(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(1.0)

def cm(v):
    return v / 2.54

def save_triplet(fig, save_dir, name):
    out_base = os.path.join(save_dir, name)
    for fmt in ("png", "pdf", "svg"):
        fig.savefig(f"{out_base}.{fmt}", format=fmt, dpi=DPI, bbox_inches='tight')
        print(f"  Saved: {out_base}.{fmt}")
    plt.close(fig)


# =============================================================================
# Heatmap (used by Figure 2, Figure S1, Figure 3d)
# =============================================================================
def render_grouped_heatmap(eta_mat, row_groups, target_labels, save_dir, name,
                           feature_display, group_color, fig_h_cm=14,
                           vmax_cap=None, vmax_set=None):
    """Render a grouped SHAP-η heatmap.

    Parameters
    ----------
    eta_mat : (n_features, n_targets) array of η values
    row_groups : list of (group_name, [feature_keys])
    target_labels : list of column labels in display order
    feature_display : dict mapping feature_key → display label
    group_color : dict mapping group_name → hex
    vmax_cap : optional float; if provided, colour scale is *capped* at this
               value and cells above it are labelled ">vmax_cap". Used to
               preserve dynamic range when one cell is much larger than the
               rest. Suppresses dynamic range information for capped cells.
    vmax_set : optional float; if provided, colour scale is *fixed* to
               exactly this value (the auto-computed and vmax_cap arguments
               are ignored). Used to harmonise the colour scale across
               multiple panels that should be visually comparable.
    """
    apply_rcparams()
    feature_order, grp_bounds = [], []
    cur = 0
    for gn, fs in row_groups:
        feature_order.extend(fs); cur += len(fs); grp_bounds.append((gn, cur))
    n_t = eta_mat.shape[1]
    if vmax_set is not None:
        vmax = float(vmax_set)
    else:
        vmax = max(0.10, np.ceil(eta_mat.max() * 20) / 20)
        if vmax_cap is not None:
            vmax = min(vmax, vmax_cap)

    fig = plt.figure(figsize=(cm(18), cm(fig_h_cm)))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.18, 1.0], wspace=0.0)
    ax_side = fig.add_subplot(gs[0, 0])
    ax_b    = fig.add_subplot(gs[0, 1])

    im = ax_b.imshow(eta_mat, cmap=CMAP_RED, aspect='auto', vmin=0, vmax=vmax)
    for i in range(eta_mat.shape[0]):
        for j in range(eta_mat.shape[1]):
            v = eta_mat[i, j]
            tc = 'white' if v > vmax * 0.55 else 'black'
            if v < 0.005:
                lbl = "—"
                tc = '#bbb'
            elif vmax_cap is not None and v > vmax:
                # Truly above an imposed cap → ">cap" notation
                lbl = f">{vmax:.2f}"
            else:
                # Show the actual value, even if it equals vmax (honest range)
                lbl = f".{v:.2f}"[1:] if v < 1 else f"{v:.2f}"
            ax_b.text(j, i, lbl, ha='center', va='center', fontsize=6.8, color=tc)
    ax_b.set_yticks([])
    ax_b.set_xticks(range(n_t))
    ax_b.set_xticklabels(target_labels, rotation=45, ha='right', fontsize=8)
    ax_b.tick_params(axis='x', length=3, width=1.0)
    for _, idx in grp_bounds[:-1]:
        ax_b.axhline(idx - 0.5, color='white', linewidth=2.4)
        ax_b.axhline(idx - 0.5, color='black', linewidth=1.0)
    black_box(ax_b)

    ax_side.set_xlim(0, 1); ax_side.set_ylim(-0.5, len(feature_order) - 0.5)
    ax_side.invert_yaxis(); ax_side.axis('off')

    prev = 0
    bar_x, bar_w = 0.32, 0.10
    for gn, idx in grp_bounds:
        mid = (prev + idx - 1) / 2.0
        ax_side.add_patch(Rectangle(
            (bar_x, prev - 0.40), bar_w, (idx - prev) - 0.20,
            facecolor=group_color[gn], edgecolor='none', clip_on=False,
        ))
        ax_side.text(0.10, mid, gn, rotation=90, ha='center', va='center',
                     fontsize=8, fontweight='bold', color='black')
        for row in range(prev, idx):
            ax_side.text(0.97, row, feature_display[feature_order[row]],
                         ha='right', va='center', fontsize=8,
                         color='black', fontweight='bold')
        prev = idx

    cax = ax_b.inset_axes([1.018, 0.0, 0.018, 1.0])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('SHAP η', fontsize=9)
    cbar.ax.tick_params(labelsize=7, width=1.0, length=3)
    cbar.outline.set_linewidth(1.0); cbar.outline.set_edgecolor('black')

    save_triplet(fig, save_dir, name)
