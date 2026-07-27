"""
Carbon footprint of model training/inference for THIS study (Supplementary Figure S2).
======================================================================================
Single authoritative producer of:
  - output/co2_panel_a_intensities.{png,pdf,svg}   (Supplementary Figure S2)
  - output/co2_methodology_and_results.md          (methodology + results prose)
  - output/co2_emissions_results.json              (machine-readable results)

Carbon footprint per task: CO2e (kg) = t × n_GPU × TDP × PUE × CI × 1e-6
(Green Algorithms; Lannelongue et al., Advanced Science, 2021).

Grid carbon intensities (Supplementary Figure S2) use a single consistent, cited source:
Ember, Global Electricity Review 2024 / European Electricity Review 2024 (2023 values),
mirrored by Our World in Data, "Carbon intensity of electricity":
  Switzerland 41, EU-27 242, United States 369, World (global average) 480 gCO2/kWh.
"""

import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ── Scientific figure standards ──────────────────────────────────────────
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['figure.dpi'] = 400
rcParams['savefig.dpi'] = 400
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.02

def cm_to_inch(cm):
    return cm / 2.54

def save_figure(fig, name, formats=('png', 'pdf', 'svg')):
    for fmt in formats:
        fig.savefig(f'{name}.{fmt}', format=fmt, dpi=400, bbox_inches='tight')
    print(f'Saved: {name}.png / .pdf / .svg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load config + specs ────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "hardware_and_grid_config.json")) as f:
    cfg = json.load(f)

TDP    = cfg["gpu"]["tdp_watts"]                          # A100 80GB PCIe = 300 W
N_GPU  = cfg["gpu"]["n_gpu"]
PUE    = cfg["facility"]["pue"]                           # 1.1
CI     = cfg["facility"]["carbon_intensity_gCO2_per_kWh"] # Switzerland = 41
EPOCHS = cfg["training"]["epochs"]                        # 1000
PT_GPU_H = cfg["self_supervised_pretraining"]["gpu_hours"] # 50
PT_N_GPU = cfg["self_supervised_pretraining"]["n_gpu"]
PT_TDP   = cfg["pretraining_gpu"]["tdp_watts"]            # H200 NVL = 600 W

# Four cited grid carbon intensities (Ember 2023) — single consistent source.
GRID = cfg["grid_scenarios_gCO2_per_kWh"]
ci_regions = {
    f"Switzerland ({GRID['Switzerland']})":            GRID["Switzerland"],
    f"EU-27 ({GRID['EU-27']})":                        GRID["EU-27"],
    f"United States ({GRID['United States']})":        GRID["United States"],
    f"Global avg ({GRID['World (global average)']})":  GRID["World (global average)"],
}

our_models = []
with open(os.path.join(DATA_DIR, "model_computational_specs.csv")) as f:
    for row in csv.DictReader(f):
        our_models.append({
            "model_name":               row["model_name"],
            "training_time_per_epoch_s": float(row["training_time_per_epoch_s"]),
            "model_parameters":         int(row["model_parameters"]),
            "inference_time_10cases_s": float(row["inference_time_10cases_s"]),
        })

# ── Compute our emissions ───────────────────────────────────────────────────
def co2_kg(runtime_h, tdp=TDP, n_gpu=N_GPU):
    return runtime_h * n_gpu * tdp * PUE * CI * 1e-6

pretrain_energy_kwh = PT_GPU_H * PT_N_GPU * PT_TDP * PUE / 1000.0   # incl. PUE
pretrain_co2_kg     = pretrain_energy_kwh * CI / 1000.0

our_results = []
for m in our_models:
    train_h    = m["training_time_per_epoch_s"] * EPOCHS / 3600.0
    energy_kwh = train_h * TDP * PUE / 1000.0
    co2_infer_1_kg = co2_kg(m["inference_time_10cases_s"] / 10.0 / 3600.0)
    our_results.append({
        "model_name":              m["model_name"],
        "model_parameters":        m["model_parameters"],
        "total_training_time_h":   round(train_h, 2),
        "energy_kwh":              round(energy_kwh, 2),
        "co2_training_kg":         round(co2_kg(train_h), 4),
        "co2_training_g":          round(co2_kg(train_h) * 1000, 1),
        "co2_inference_1case_g":   round(co2_infer_1_kg * 1000, 4),
    })

# ── Supplementary Figure S2 ─────────────────────────────────────────────────
REGION_COLORS = ['#E07B39', '#2A9D8F', '#E9C46A', '#2C4A6B']  # CH, EU, USA, World

model_names = [r["model_name"] for r in our_results]
model_short = [n.replace(" W CT-Aug", "\n(W Aug)").replace(" Wo CT-Aug", "\n(Wo Aug)")
               for n in model_names]
energy_kwh = np.array([r["energy_kwh"] for r in our_results])

a_labels = model_short + ["SSL pretrain\n(H200)"]
a_energy = np.append(energy_kwh, pretrain_energy_kwh)

REF = cfg["reference_comparison"]
ref_co2_kg = REF["distance_km"] * REF["car_factor_gCO2_per_km"] / 1000.0   # 100 km x 175 g/km = 17.5 kg
REF_COLOR = "#9AA3A8"  # neutral grey, distinct from the four grid colours

fig_a, ax_a = plt.subplots(figsize=(cm_to_inch(20.5), cm_to_inch(11.5)))
n_groups = len(a_labels)
x = np.arange(n_groups)
bar_width = 0.20
group_center = bar_width * 1.5
y_max_a = max(a_energy * max(ci_regions.values()) / 1000)

def _fmt_kg(v):
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"

for i, (region, ci_val) in enumerate(ci_regions.items()):
    co2_vals = a_energy * ci_val / 1000
    ax_a.bar(x + i * bar_width, co2_vals, bar_width,
             label=region, color=REGION_COLORS[i], edgecolor='white', linewidth=0.3)
    for xi, val in zip(x + i * bar_width, co2_vals):
        ax_a.text(xi, val + y_max_a * 0.012, _fmt_kg(val),
                  ha='center', va='bottom', rotation=90, fontsize=5.5, color='#222222')

# Single grid-independent reference bar in its own group on the far right.
x_ref = n_groups + 0.25
ax_a.bar(x_ref + group_center, ref_co2_kg, 0.46,
         color=REF_COLOR, edgecolor='white', linewidth=0.3, zorder=3)
ax_a.text(x_ref + group_center, ref_co2_kg + y_max_a * 0.012, _fmt_kg(ref_co2_kg),
          ha='center', va='bottom', rotation=90, fontsize=5.5, color='#222222')

ax_a.set_ylabel('CO$_2$e (kg)', fontsize=12)
all_ticks = list(x + group_center) + [x_ref + group_center]
all_lbls = a_labels + [f"{REF['distance_km']} km\nby car\n(reference)"]
ax_a.set_xticks(all_ticks)
ax_a.set_xticklabels(all_lbls, fontsize=9.5, ha='right', rotation=25)
ax_a.set_xlim(-0.5, x_ref + group_center + 0.5)
ax_a.set_ylim(0, y_max_a * 1.52)
ax_a.legend(title='Grid carbon intensity (gCO$_2$/kWh)',
            fontsize=10, title_fontsize=11, loc='upper left',
            bbox_to_anchor=(0.0, 1.0), frameon=False,
            handlelength=1.5)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
save_figure(fig_a, os.path.join(OUTPUT_DIR, "co2_panel_a_intensities"))
plt.close(fig_a)

# ── Methodology + Results text ──────────────────────────────────────────────
min_tr = min(our_results, key=lambda r: r["co2_training_kg"])
max_tr = max(our_results, key=lambda r: r["co2_training_kg"])
min_pp = min(r["co2_inference_1case_g"] for r in our_results)
max_pp = max(r["co2_inference_1case_g"] for r in our_results)
total_swiss_kg = sum(r["co2_training_kg"] for r in our_results) + pretrain_co2_kg
ref_equiv_km = total_swiss_kg / (REF["car_factor_gCO2_per_km"] / 1000.0)

methodology = f"""# Supplementary Note S3. Carbon footprint of model training and inference\n\n## Methodology

Carbon dioxide equivalent (CO₂e) emissions were estimated for model training and inference using the Green Algorithms framework (Lannelongue et al., Advanced Science, 2021), as CO₂e (kg) = t × n_GPU × TDP × PUE × CI × 10⁻⁶, where t is wall-clock runtime (hours), n_GPU the number of GPUs, TDP the GPU thermal design power, PUE the facility power usage effectiveness (set to {PUE} to be comparable with other foundation-model studies, e.g. DINOv2 (Oquab et al., 2024)), and CI the grid carbon intensity. Self-supervised pretraining was a one-time cost ({PT_GPU_H} GPU-hours on a single {cfg['pretraining_gpu']['name']}, TDP = {PT_TDP} W); each of the six benchmark models was then trained on a single {cfg['gpu']['name']} (TDP = {TDP} W) for {EPOCHS} epochs. Because CO₂e scales linearly with grid carbon intensity, each task is reported under four grid scenarios drawn from a single source (Ember, 2023 values): Switzerland ({GRID['Switzerland']} gCO₂/kWh, the low-carbon grid on which the models were trained), the EU-27 ({GRID['EU-27']}), the United States ({GRID['United States']}), and the global average ({GRID['World (global average)']}) (Supplementary Figure S2). Inference emissions are reported per patient. The Swiss intensity is production/territorial-based; a consumption-based value (accounting for imports) would be higher. Estimates capture GPU operational (Scope-2) energy only and exclude the CPU/RAM term of the full Green Algorithms model and embodied hardware emissions, so they should be read as approximate rather than as a strict upper bound.

## Results

On the low-carbon Swiss grid on which the models were trained (CI = {CI} gCO₂e/kWh), one-time self-supervised pretraining incurred {pretrain_co2_kg:.2f} kg CO₂e ({PT_GPU_H} GPU-hours on an {cfg['pretraining_gpu']['name']}), training each of the six benchmark models on a single {cfg['gpu']['name']} emitted between {min_tr['co2_training_kg']:.2f} kg ({min_tr['model_name']}) and {max_tr['co2_training_kg']:.2f} kg CO₂e ({max_tr['model_name']}), and inference cost {min_pp:.2f}–{max_pp:.2f} g CO₂e per patient (Supplementary Figure S2). Emissions were modest across all architectures, with nnU-Net the most efficient and the transformer- and Mamba-based models incurring higher cost; identical computation on more carbon-intensive grids would produce several-fold higher emissions (up to ≈{round(GRID['World (global average)']/CI)}× at the global average). To make these magnitudes tangible, pretraining plus training all six models on the Swiss grid (≈{total_swiss_kg:.0f} kg CO₂e in total) is equivalent to roughly {ref_equiv_km:.0f} km driven in an average passenger car — less than the {ref_co2_kg:.1f} kg CO₂e emitted by the single {REF['distance_km']} km car trip shown for reference in Supplementary Figure S2 (car factor ≈{REF['car_factor_gCO2_per_km']} gCO₂e/km; Green Algorithms, Lannelongue et al. 2021).
"""

# Two-sentence takeaway "heading" after Results, underlined by underscores whose
# count = round(2 x average sentence length). Keeping the rule short forces short sentences.
_s1 = (f"On the low-carbon Swiss grid used for development, pretraining and training all six "
       f"benchmark models together emitted \u2248{total_swiss_kg:.0f} kg CO\u2082e \u2014 less than a single "
       f"{REF['distance_km']} km car trip (Supplementary Figure S2).")
_s2 = ("Because CO\u2082e scales almost linearly with grid carbon intensity, where a model is run "
       "influences its footprint more than which architecture is chosen (full methods and results "
       "in Supplementary Note S3).")
_avg_chars = (len(_s1) + len(_s2)) / 2
_rule = "_" * round(_avg_chars * 2)
methodology += f"\n\n## Two-sentence summary (to include in main-text Results)\n\n**{_s1} {_s2}**\n{_rule}\n"

with open(os.path.join(OUTPUT_DIR, "co2_methodology_and_results.md"), "w") as f:
    f.write(methodology)
print("Saved: output/co2_methodology_and_results.md")

with open(os.path.join(OUTPUT_DIR, "co2_emissions_results.json"), "w") as f:
    json.dump({
        "metadata": {
            "training_gpu": cfg["gpu"]["name"], "training_tdp_watts": TDP,
            "pretraining_gpu": cfg["pretraining_gpu"]["name"], "pretraining_tdp_watts": PT_TDP,
            "pretraining_gpu_hours": PT_GPU_H, "n_gpu": N_GPU, "pue": PUE,
            "carbon_intensity_gCO2_kWh_switzerland": CI, "epochs": EPOCHS,
            "grid_scenarios_gCO2_per_kWh": GRID,
            "formula": cfg["_meta"]["formula"], "reference": cfg["reference"],
        },
        "self_supervised_pretraining": {
            "gpu": cfg["pretraining_gpu"]["name"], "tdp_watts": PT_TDP, "gpu_hours": PT_GPU_H,
            "energy_kwh": round(pretrain_energy_kwh, 2),
            "co2_kg_switzerland": round(pretrain_co2_kg, 4),
            "note": "One-time cost, shared across the study (not per model).",
        },
        "our_results": our_results,
        "reference_comparison": {
            "activity": REF["activity"], "distance_km": REF["distance_km"],
            "car_factor_gCO2_per_km": REF["car_factor_gCO2_per_km"],
            "co2_kg": round(ref_co2_kg, 2),
            "all_training_swiss_kg": round(total_swiss_kg, 2),
            "all_training_swiss_equiv_car_km": round(ref_equiv_km, 0),
            "source": REF["car_factor_source"],
        },
    }, f, indent=2)
print("Saved: output/co2_emissions_results.json")
print("Done.")
