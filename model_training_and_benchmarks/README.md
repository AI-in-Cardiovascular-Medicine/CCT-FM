# Model training and benchmarks

Every training run behind CCT-FM, driven from a single entry point: self-supervised
pre-training on the unlabelled corpus, supervised training of the segmentation models,
the architecture and model benchmarks, and the downstream / label-efficiency grids. All
of it runs on [nnUZoo](https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo) — an
nnU-Net-based framework that exposes CNN, transformer, and Mamba (state-space)
architectures behind a single trainer interface, so every architecture is trained and
evaluated under an identical pipeline (same plans, same patch size, same schedule, same
metrics).

What is reported:

- **Architectures benchmark** — nnU-Net (with and without the cardiac CT augmentation,
  CT-Aug), U<sup>2</sup>-Net Small, U<sup>2</sup>-Net Large, SwinUNETR, and SwinUMamba,
  all trained on the 1000-case expert-annotated training set (14 cardiac structures) and
  evaluated on the five external test sets.
- **Models benchmark** — the CCT-FM model against the open-source tools TotalSegmentator,
  ATLAS, and MOOSE (external tools are run through their own CLIs; only the shared
  evaluation step, `nnUNetv2_evaluate_folder`, is reused).
- **Foundation model** — masked-autoencoder pre-training on 60k unlabelled cardiac CT.
- **Downstream tasks and label efficiency** — 7 label fractions × {scratch, pre-trained}
  per task, the matched grid behind the low-data-regime result.

## Contents

| Path | Purpose |
|------|---------|
| [train.py](train.py) | Single entry point for preprocessing, training and testing; wraps the nnUZoo CLI, sets the `nnUNet_*` environment variables, and chains prediction → (optional label reversal) → evaluation |
| [pyproject.toml](pyproject.toml) | Only the extras this repo adds on top of nnUZoo — the `train.py` imports and the test-metric dependency. nnUZoo itself provides everything else |

The figures and statistical tests computed on the metric JSONs this produces live in
[`figures-and-statistics/`](../figures-and-statistics), which needs neither torch nor
nnUZoo.

## Models

Trainer names as passed to `-tr`. All runs use the `3d_fullres` configuration with the
`nnUNetResEncUNetLPlans` plans (`-p`), i.e. the residual-encoder L preset.

| Model (manuscript) | Trainer (`-tr`) |
|---|---|
| nnU-Net **without** CT-Aug | `nnUNetTrainer` |
| nnU-Net **with** CT-Aug | `nnUNetTrainerMetalArtifactStepMotionNoMirror` |
| U<sup>2</sup>-Net Small | `nnUNetTrainerMetalArtifactStepMotionNoMirrorU2NetS` |
| U<sup>2</sup>-Net Large | `nnUNetTrainerMetalArtifactStepMotionNoMirrorU2Net` |
| SwinUNETR | `nnUNetTrainerMetalArtifactStepMotionNoMirrorSwUNETR` |
| SwinUMamba | `nnUNetTrainerMetalArtifactStepMotionNoMirrorSwinUMamba` <sup>†</sup> |

<sup>†</sup> needs the Mamba CUDA kernels in addition to nnUZoo — see
[Environment](#environment).

`MetalArtifactStepMotion` is the cardiac CT augmentation (CT-Aug) pipeline — simulation of
metal artifacts together with the step and motion artifacts; 
`NoMirror` disables mirroring, which is anatomically invalid for cardiac chambers.


## Environment

nnUZoo owns the dependency set: installing it brings the trainers, the architectures and
the `nnUNetv2_*` console scripts that `train.py` drives. Follow the
[nnUZoo installation guide](https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo#installation-guide)
— the install is not duplicated here.

For SwinUMamba, the `mamba-ssm` / `causal-conv1d` CUDA kernels are needed on top of that.
They are not installed by default; see
[`mamba_assets/`](https://github.com/AI-in-Cardiovascular-Medicine/nnUZoo/tree/main/mamba_assets)
in nnUZoo for the kernels and the matching instructions. With those in place SwinUMamba
trains under the same Python 3.12 environment as everything else, so no second environment
is required.

[requirements.txt](requirements.txt) in this folder declares **only** what this repository adds
on top of nnUZoo.
### Hardware

| Stage | GPU | Notes |
|---|---|---|
| Self-supervised pre-training (60k scans) | 1 × NVIDIA **H200 NVL** (141 GB) | ~50 GPU-hours, one-time, shared across the study |
| Supervised training of all six benchmark models | 1 × NVIDIA **A100 80 GB PCIe** | 1000 epochs each, single GPU per model |

Host: 12 of 288 CPU cores and ~1 TB system RAM available (`-n_proc 12`). All runs are
single-GPU (`-num_gpus 1`).

The 80 GB / 141 GB VRAM is what the released configuration was trained with, not a hard
floor: the `3d_fullres` `nnUNetResEncUNetLPlans` patch size drives the requirement, and a
24 GB card is enough for the CNN models at a reduced `-batch_size`. SwinUNETR and
SwinUMamba are the memory-heavy ones. Inference needs ≥ 8 GB system RAM; a GPU is
recommended but not required for the CNN models.

Exact runtimes, TDPs and the resulting carbon footprint per model are in
[`co2-emission/`](../co2-emission).

## Dataset

Everything lives in a single nnU-Net dataset, `Dataset051_CT_Cardio_FULL_Organs`: 1000
training cases (`imagesTr`/`labelsTr`) and the five external test sets as sibling
`<target>_imagesTs` / `<target>_labelsTs` folders. One CT input channel, 14 foreground
structures plus background:

| Label | Structure | | Label | Structure |
|---|---|---|---|---|
| 1 | Coronary Artery | | 8 | Pulmonary Arteries |
| 2 | LV Myocardium | | 9 | Precardial Fat |
| 3 | Left Atrium | | 10 | Epicardial Fat |
| 4 | Left Ventricle | | 11 | Pulmonary Vein |
| 5 | Right Atrium | | 12 | Superior Vena Cava |
| 6 | Right Ventricle | | 13 | Inferior Vena Cava |
| 7 | Aorta | | 14 | LA Appendage |

### External test sets (`${target}`)

The five `${target}` values used throughout the commands below, 598 cases in total:

| `${target}` | n | Modality / cohort |
|---|---|---|
| `ExtTest1` | 173 | Cardiac CT (CCT) |
| `ExtTest2` | 198 | Cardiac CT (CCT) |
| `ExtTest3` | 132 | Cardiac CT (CCT) |
| `ExtTest4` | 75 | Cardiac CT with metal artifacts |
| `ExtTest5` | 20 | CCTA — public MM-WHS challenge CT set |
| **total** | **598** | |


For the models benchmark, the predictions of the open-source tools are stored alongside the
reference labels in the same dataset folder — `<target>_ts_labels` (TotalSegmentator),
`<target>_atlas_labels` (ATLAS) and `<target>_moose_labels` (MOOSE), with
`_heart_labels` / `_coronary_labels` / `_general_labels` variants where a tool splits its
output across several models — and evaluated with the same
`nnUNetv2_evaluate_folder` call as the in-house models.

## Data layout

Datasets follow the nnU-Net v2 raw format. `train.py` resolves `nnUNet_raw`,
`nnUNet_preprocessed` and `nnUNet_results` under a single `-base_dir` (or set the three
individually with `-nnunet_raw`, `-nnunet_preprocessed`, `-nnunet_results`):

```
<base_dir>/
├── nnunet_raw/
│   └── Dataset051_CT_Cardio_FULL_Organs/
│       ├── dataset.json
│       ├── imagesTr/            case_0000.nii.gz, ...
│       ├── labelsTr/            case.nii.gz, ...
│       ├── <target>_imagesTs/   one folder per external test set
│       └── <target>_labelsTs/
├── nnunet_preprocessed/
└── nnunet_results/
```

`-test_input`, `-test_input_labels`, `-test_output` and `-pretrained_weights` are all
resolved **relative** to `nnunet_raw` / `nnunet_results` when the path does not exist as
given, so the commands below stay machine-independent. Each external test set lives in its
own `<target>_imagesTs` / `<target>_labelsTs` pair and is evaluated into its own
`<target>_test_summary.json`, which keeps the five external evaluations separate.

## Preprocessing

Every dataset is planned and preprocessed once, before any training. The result is shared
by every trainer and every architecture, so this is not repeated per model. Five datasets
are involved:

| Dataset | Role |
|---|---|
| `Dataset064_Cardiac_CT_FM` | self-supervised pre-training corpus, 60k unlabelled cardiac CT |
| `Dataset051_CT_Cardio_FULL_Organs` | main 14-structure task: 1000 training cases, 598 external test cases |
| `Dataset280_corronary_anomaly` | downstream — coronary artery, anomaly cohort |
| `Dataset091_TAVI_VNC` | downstream — non-contrast CT |
| `Dataset281_mm_wh` | downstream — whole heart, public MM-WHS |


```bash
python train.py \
  -dataset_name Dataset064_Cardiac_CT_FM \
  -model 3d_fullres \
  -pl nnUNetPlannerResEncL \
  --preprocess --verify_dataset_integrity \
```

Main 14-structure task:

```bash
python train.py \
  -dataset_name Dataset051_CT_Cardio_FULL_Organs \
  -model 3d_fullres \
  -pl nnUNetPlannerResEncL \
  --preprocess --verify_dataset_integrity
```

Downstream — coronary artery, anomaly cohort:

```bash
python train.py \
  -dataset_name Dataset280_corronary_anomaly \
  -model 3d_fullres \
  -pl nnUNetPlannerResEncL \
  --preprocess --verify_dataset_integrity
```

Downstream — non-contrast CT:

```bash
python train.py \
  -dataset_name Dataset091_TAVI_VNC \
  -model 3d_fullres \
  -pl nnUNetPlannerResEncL \
  --preprocess --verify_dataset_integrity
```

Downstream — whole heart, public MM-WHS:

```bash
python train.py \
  -dataset_name Dataset281_mm_wh \
  -model 3d_fullres \
  -pl nnUNetPlannerResEncL \
  --preprocess --verify_dataset_integrity
```

`-pl nnUNetPlannerResEncL` is the planner that writes the `nnUNetResEncUNetLPlans` plans
used by every training command below; the two must match.

* Add `--p_continue_process` to resume an interrupted run without
rewriting the cases already done; that matters mostly for the same corpus.
* `-np` / `-npfp` cap the preprocessing and fingerprinting workers if RAM is tight — each worker holds a full
volume.

## Foundation-model pre-training

Self-supervised masked-autoencoder pre-training on the unlabelled corpus. This is the
one-time run that produces the checkpoint every `_Pretrain_` fine-tuning consumes:

```bash
python train.py \
  -device 0 \
  -model 3d_fullres \
  -dataset_name Dataset064_Cardiac_CT_FM \
  -tr nnUNetTrainerCT3DSSLMAENNUNetArtifactAugP256_BS2 \
  -p nnUNetResEncUNetLPlans \
  --train --c \
  -num_epochs 1000
```

The trainer name encodes the setup: MAE objective on 3D CT, the CT-Aug artifact
augmentation inside the SSL pipeline (`ArtifactAug`), patch size 256 and batch size 2
(`P256_BS2`). This is the H200 run in the hardware table (~50 GPU-hours). It writes

```
nnunet_results/Dataset064_Cardiac_CT_FM/nnUNetTrainerCT3DSSLMAENNUNetArtifactAugP256_BS2__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth
```

## Supervised training

Training the benchmark architectures on the main task — repeat with each `-tr` from the
model table:

```bash
python train.py \
  -device 0 \
  -model 3d_fullres \
  -dataset_name Dataset051_CT_Cardio_FULL_Organs \
  -tr nnUNetTrainerMetalArtifactStepMotionNoMirror \
  -p nnUNetResEncUNetLPlans \
  --train --c
```

`-num_epochs` defaults to 1000, which is what all six benchmarked models use.

- `--c` resumes from the latest checkpoint and skips the run entirely if validation has
  already completed — safe to re-issue over a whole sweep. It is mutually exclusive with
  `--overwrite_train`, and it takes precedence over `-pretrained_weights` (resuming ignores
  the pre-trained weights, since they are already baked into the checkpoint).
- `-bin_path` points at the `bin/` of the environment holding the `nnUNetv2_*` console
  scripts; needed only when driving an environment other than the active one.

## Downstream tasks and label efficiency

Each downstream task is trained as a **7 × 2 grid**: seven label fractions × {from
scratch, from the foundation model} = **14 trainings per task**. The two arms use the same
architecture and label fraction and differ only in initialisation, lr_scheduler, and learning rate, so each pair
is a matched comparison.

| | Trainer | Initialisation |
|---|---|---|
| scratch | `nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_{pct}Percent` | random |
| pre-trained | `nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_{pct}Percent` | FM encoder, via `-pretrained_weights` + `--only_encoder` |

with `{pct}` ∈ `01 05 10 20 50 75 100` (percent of that task's labelled training set).

| Downstream dataset | Task | n (train / test) |
|---|---|---|
| `Dataset280_corronary_anomaly` | coronary artery, anomaly cohort | 29 (25 / 4) |
| `Dataset091_TAVI_VNC` | non-contrast CT | 64 (58 / 6) |
| `Dataset051_CT_Cardio_FULL_Organs` | whole heart | 1000 |

All 14 runs for all three tasks — 42 trainings:

```bash
FM_CKPT=Dataset064_Cardiac_CT_FM/nnUNetTrainerCT3DSSLMAENNUNetArtifactAugP256_BS2__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth

PCTS=(01 05 10 20 50 75 100)
EPOCHS=(50 50 100 100 150 200 250)

for ds in Dataset280_corronary_anomaly Dataset091_TAVI_VNC Dataset051_CT_Cardio_FULL_Organs; do
  for i in "${!PCTS[@]}"; do
    pct="${PCTS[$i]}"
    num_epochs="${EPOCHS[$i]}"

    # from scratch
    python train.py \
      -device 0 \
      -model 3d_fullres \
      -dataset_name "$ds" \
      -tr "nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_${pct}Percent" \
      -p nnUNetResEncUNetLPlans \
      --train --c \
      -num_epochs "$num_epochs"

    # from the foundation model, encoder only
    python train.py \
      -bin_path "$CONDA_PREFIX/bin" \
      -device 0 \
      -model 3d_fullres \
      -dataset_name "$ds" \
      -tr "nnUNetTrainerCT3DSSLMAEUNetNNUNetSegmentation_256_B2_Pretrain_${pct}Percent" \
      -p nnUNetResEncUNetLPlans \
      --train --c \
      -num_epochs "$num_epochs" \
      --only_encoder \
      -pretrained_weights "$FM_CKPT"

  done
done
```

- `--only_encoder` loads only the encoder from the checkpoint and leaves the decoder
  randomly initialised.
- `-pretrained_weights` is resolved relative to `nnunet_results`, so the checkpoint path
  stays machine-independent.
- The fine-tuning runs use a far shorter schedule than the 1000-epoch benchmarks: the
  epoch budget scales with the label fraction, from 50 epochs at 1–5 % up to 250 at 100 %.
- The label fraction is applied by the trainer itself, not by the data on disk: all 14 runs
  of a task read the same preprocessed dataset.

Running the same grid on `Dataset051_CT_Cardio_FULL_Organs` produces the label-efficiency
curves for the main 14-structure task, evaluated on the five external test sets.

## Testing

Predict on one external test set and evaluate against its reference labels:

```bash
target=ExtTest1   # or ExtTest2 ... ExtTest5

python train.py \
  -tr nnUNetTrainerMetalArtifactStepMotionNoMirrorU2NetS \
  -device 2 \
  -dataset_name Dataset051_CT_Cardio_FULL_Organs \
  -test_input      "${target}_imagesTs" \
  -test_input_labels "${target}_labelsTs" \
  -test_output     "${target}_imagesTs" \
  --test \
  -test_summary_name "${target}_test_summary.json"
```

`-test_output` is created under
`nnunet_results/<dataset>/<tr>__<plans>__<model>/fold_<fold>/`, and the summary is written
next to it. `--test` performs three steps: prediction (`code_inference.py`), optional label
reversal when `dataset.json` declares `reverse_class`, and metric computation
(`nnUNetv2_evaluate_folder`). Add `--add_hd` for Hausdorff distances, `--overwrite_test` to
discard existing predictions, and `--overwrite_test_evaluation` to recompute metrics from
existing predictions.

Sweeping all architectures over all external test sets:

```bash

for tr in nnUNetTrainer \
          nnUNetTrainerMetalArtifactStepMotionNoMirror \
          nnUNetTrainerMetalArtifactStepMotionNoMirrorU2NetS \
          nnUNetTrainerMetalArtifactStepMotionNoMirrorU2Net \
          nnUNetTrainerMetalArtifactStepMotionNoMirrorSwUNETR; do
    python train.py -tr "$tr" -device 0 \
        -dataset_name Dataset051_CT_Cardio_FULL_Organs --train
    for target in ExtTest1 ExtTest2 ExtTest3 ExtTest4 ExtTest5; do
       python train.py -tr "$tr" -device 0 \
        -dataset_name Dataset051_CT_Cardio_FULL_Organs
        --train 
        -test_input "${target}_imagesTs" \
        -test_input_labels "${target}_labelsTs" \
        -test_output "${target}_imagesTs" \
        --test -test_summary_name "${target}_test_summary.json"
  done
done
```

Run the same loop with `-tr nnUNetTrainerMetalArtifactStepMotionNoMirrorSwinUMamba` to add
SwinUMamba.

## Useful arguments

`python train.py --help` lists everything; the ones that matter for reproduction:

| Argument | Meaning |
|---|---|
| `-dataset_name`, `-fold`, `-model`, `-p`, `-pl` | dataset, cross-validation fold, configuration (`3d_fullres`), plans, planner |
| `-device`, `-num_gpus` | GPU index/indices (mapped to `CUDA_VISIBLE_DEVICES`) and number of GPUs |
| `-num_epochs`, `-batch_size`, `-val_size` | schedule length, batch-size override, train/val split regenerated with `-random_state` |
| `--preprocess`, `--train`, `--test` | stage selection; combinable in one call |
| `-np`, `-npfp`, `-e_np`, `-n_proc` | worker counts for preprocessing, fingerprinting, evaluation, and data augmentation |
| `--val`, `--val_best`, `-skip_val` | run validation only / validate with the best checkpoint / skip validation |
| `-test_checkpoint` | checkpoint used at test time (default `checkpoint_best.pth`) |
| `--must_finished` | abort evaluation unless `checkpoint_final.pth` exists, i.e. training really completed |

`nnUNet_compile` is forced off and `nnUNet_def_n_proc` / `nnUNet_n_proc_DA` are set from
`-n_proc` for every invocation.

## Notes

- Model weights are released separately; see the repository root [README](../README.md).
- The trainers, augmentation pipeline, and architectures live in nnUZoo.
- Main CTAug library is located in: [CTAug](https://github.com/ai-in-Cardiovascular-Medicine/ctaug)

## Citation

The benchmarking framework is nnUZoo; please cite it alongside the CCT-FM paper (see the
root [README](../README.md)):

```
@article{kazaj2025claims,
  author  = {Pooya Mohammadi Kazaj and Giovanni Baj and Yazdan Salimi and Anselm W. Stark and Waldo Valenzuela
             and George C. M. Siontis and Habib Zaidi and Mauricio Reyes and Christoph Graeni and Isaac Shiri},
  title   = {From Claims to Evidence: A Unified Framework and Critical Analysis of CNN vs. Transformer vs. Mamba in Medical Image Segmentation},
  journal = {arXiv preprint},
  volume  = {2503.01306},
  year    = {2025},
  doi     = {10.48550/arXiv.2503.01306},
  url     = {https://arxiv.org/abs/2503.01306}
}
```
