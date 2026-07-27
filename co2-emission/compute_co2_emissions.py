"""
CO2 Emission Estimation for Deep Learning Model Training and Inference
======================================================================
Based on the Green Algorithms framework (Lannelongue et al., Advanced Science, 2021).

Formula: CO2e (kg) = t × n_GPU × TDP × PUE × CI × 1e-6

This script computes the per-model training and per-patient inference.
"""

import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "hardware_and_grid_config.json")) as f:
    config = json.load(f)

N_GPU  = config["gpu"]["n_gpu"]
TDP    = config["gpu"]["tdp_watts"]                 # A100 80GB PCIe = 300 W
PUE    = config["facility"]["pue"]                  # 1.1
CI     = config["facility"]["carbon_intensity_gCO2_per_kWh"]   # Switzerland 41
EPOCHS = config["training"]["epochs"]               # 1000

PT_GPU_H = config["self_supervised_pretraining"]["gpu_hours"]   # 50
PT_N_GPU = config["self_supervised_pretraining"]["n_gpu"]       # 1
PT_TDP   = config["pretraining_gpu"]["tdp_watts"]              # H200 NVL = 600 W

def calc_co2_kg(runtime_hours, n_gpu=N_GPU, tdp=TDP, pue=PUE, ci=CI):
    return runtime_hours * n_gpu * tdp * pue * ci * 1e-6

models = []
with open(os.path.join(DATA_DIR, "model_computational_specs.csv")) as f:
    for row in csv.DictReader(f):
        models.append({
            "model_name":               row["model_name"],
            "training_time_per_epoch_s": float(row["training_time_per_epoch_s"]),
            "size_mb":                  float(row["size_mb"]),
            "model_parameters":         int(row["model_parameters"]),
            "inference_time_10cases_s": float(row["inference_time_10cases_s"]),
        })

pretrain_co2_kg = calc_co2_kg(PT_GPU_H, n_gpu=PT_N_GPU, tdp=PT_TDP)

results = []
for m in models:
    total_train_h = m["training_time_per_epoch_s"] * EPOCHS / 3600.0
    co2_train_kg  = calc_co2_kg(total_train_h)
    infer_10_s = m["inference_time_10cases_s"]
    co2_infer_10_kg = calc_co2_kg(infer_10_s / 3600.0)
    infer_1_s = infer_10_s / 10.0
    co2_infer_1_kg = calc_co2_kg(infer_1_s / 3600.0)
    results.append({
        "model_name":               m["model_name"],
        "model_parameters":         m["model_parameters"],
        "size_mb":                  m["size_mb"],
        "total_training_time_h":    round(total_train_h, 2),
        "co2_training_kg":          round(co2_train_kg, 6),
        "co2_training_g":           round(co2_train_kg * 1000, 3),
        "inference_10cases_time_s": round(infer_10_s, 2),
        "co2_inference_10cases_g":  round(co2_infer_10_kg * 1000, 4),
        "inference_1case_time_s":   round(infer_1_s, 2),
        "co2_inference_1case_g":    round(co2_infer_1_kg * 1000, 4),
    })

print("=" * 100)
print("CO2 EMISSION ESTIMATES  —  Green Algorithms Framework")
print(f"Training GPU: {config['gpu']['name']}  |  TDP: {TDP} W  |  n_GPU: {N_GPU}")
print(f"Pretraining GPU: {config['pretraining_gpu']['name']}  |  TDP: {PT_TDP} W  |  {PT_GPU_H} GPU-h")
print(f"PUE: {PUE}  |  CI (Switzerland): {CI} gCO2e/kWh  |  Epochs: {EPOCHS}")
print("=" * 100)
print(f"Self-supervised pretraining (one-time): {pretrain_co2_kg:.3f} kg "
      f"({pretrain_co2_kg*1000:.1f} g) CO2e at CI={CI} gCO2e/kWh")
print("-" * 100)
header = f"{'Model':<25} {'Params':>12} {'Train(h)':>10} {'CO2 Train(g)':>14} {'CO2/patient(g)':>15}"
print(header)
print("-" * 100)
for r in results:
    print(f"{r['model_name']:<25} {r['model_parameters']:>12,} {r['total_training_time_h']:>10.2f} "
          f"{r['co2_training_g']:>14.2f} {r['co2_inference_1case_g']:>15.4f}")

output_csv = os.path.join(OUTPUT_DIR, "co2_emissions_results.csv")
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults saved to: {output_csv}")
