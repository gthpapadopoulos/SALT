<h1 align="center">SALT</h1>

<p align="center">
  <strong>State-Aligned Low-Rank Transfer for Single-Temporal Cross-Dataset<br>
  Structural Localization in Remote Sensing</strong>
</p>

<p align="center">
  <strong>Accepted at WACV 2027 &middot; GeoCV Workshop</strong>
</p>

Official PyTorch implementation of **SALT**, accepted at the **Third Workshop
on Computer Vision for Geospatial Image Analysis (GeoCV), WACV 2027**.

## Overview

Change detection in remote sensing conventionally relies on spatially aligned
bi-temporal imagery, which may be unavailable or unreliable in practical
scenarios. Recent single-temporal approaches address this limitation by
learning structural evidence, such as building regions, from a labeled source
dataset and transferring it to unseen target imagery using only the current
image. However, adaptation to the source dataset can over-specialize
pretrained features and reduce their cross-dataset generalization, a risk that
this paper argues is higher when the feature adaptation is parameterized
independently of the states used for prediction. To address this issue, the
current study introduces SALT, a State-Aligned Low-Rank Transfer framework.
SALT keeps a DINOv3 backbone frozen and learns negative and positive
structural states that both score each patch and, through State-Aligned
Low-Rank Adaptation (SALA), span the rank-2 subspace in which patch features
are corrected. Directional State Alignment supervises these corrections to
raise the evidence of positive patches and lower that of negative ones, while
Geometry Preservation limits changes to the pairwise relations among
pretrained patch features. Experiments on LEVIR-CD and the WHU aerial
building dataset show that, at the pixel level and in both transfer
directions, SALT outperforms four baselines reproduced under a common
source-only protocol, improving over the strongest reproduced baselines by
9.45 and 5.79 points in pixel-level AP and by 27.69 and 14.25 points in IoU
(LEVIR-CD to WHU and WHU to LEVIR-CD, respectively), with only 0.068M
trainable parameters and 40.2 ms per image.

<p align="center">
  <a href="assets/SALT_overview.pdf">
    <img src="assets/SALT_overview.png" width="100%"
         alt="SALT framework: a frozen DINOv3 backbone, state-aligned low-rank adaptation, directional supervision, geometry preservation, and target structural localization.">
  </a>
</p>

## Installation

Use Python 3.10 and a CUDA-capable PyTorch installation. The retained runs used
PyTorch 2.6.0 and torchvision 0.21.0. For CUDA 12.4:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
bash scripts/setup_backbone.sh
```

The setup script checks out the exact official DINOv3 code revision. Obtain
`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` through the
[official DINOv3 repository](https://github.com/facebookresearch/dinov3), subject
to its access and license terms, and put it in `weights/`. Commands verify its
full SHA-256 before use. Backbone weights are not included.

## Datasets

Obtain LEVIR-CD and the **aerial building extraction** subset of WHU. The WHU
dataset here uses building-footprint masks, not bi-temporal building-change
masks. Keep one current/post-event LEVIR image per original sample.

Arrange the data as follows, or provide an equivalent directory using
`--data-root`:

```text
data/
  LEVIR/
    train/{B,label}/
    val/{B,label}/
    test/{B,label}/
  WHU3/
    train/{image,label}/
    val/{image,label}/
    test/{image,label}/
```

`WHU3` is the folder identifier for the aerial WHU dataset. The exact filenames,
sample order, and image labels are supplied in `splits/`. Counts are:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| LEVIR-CD | 445 | 64 | 128 |
| WHU | 4,736 | 1,036 | 2,416 |

Validate all files and mask-derived image labels before training:

```bash
python scripts/prepare_data.py --data-root data --check-masks
```

Dataset images are not included. No additional crops or tiles are generated.

## Evaluate Released Checkpoints

```bash
python evaluate.py \
  --checkpoint checkpoints/levir_to_whu/seed111.pt \
  --data-root data \
  --weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output outputs/levir_to_whu/seed111 \
  --profile
```

For the reverse direction, use `checkpoints/whu_to_levir/seed111.pt`.
Checkpoints for seeds 222 and 333 are supplied in the same folders.
`--device cuda:0` is the default; CPU evaluation is supported with
`--device cpu`. `--dinov3-repo` accepts an existing pinned DINOv3 checkout.

Each output `result.json` contains all six pixel/image metrics and the source
threshold. `--save-predictions` retains the dense raw maps and masks; these
can consume several GiB. By default, temporary maps are removed after scoring.

## Train One Model

```bash
python train.py \
  --config configs/levir_to_whu.yaml \
  --seed 111 \
  --data-root data \
  --weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output runs/levir_to_whu/seed111
```

Use `configs/whu_to_levir.yaml` for WHU source training. Train for all ten
epochs, select the maximum source-validation P-AP checkpoint with P-AUROC as
the tie-breaker, and estimate a single source-validation F1 threshold. The
target test split is accessed only by `evaluate.py`.

`best.pt` includes the head, configuration, selected epoch, and source
threshold. `last.pt` also preserves optimizer and RNG states. Add `--resume`
to continue an interrupted run from its last completed epoch.

## Run Both Directions and All Seeds

The launcher queues one experiment per currently idle GPU. It runs training
and evaluation, skips completed evaluations, resumes interrupted training,
and writes a log for every stage.

```bash
bash scripts/run_all.sh \
  --data-root data \
  --weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output runs
```

To select GPU IDs explicitly, append `--gpus 0 1 2 3`. To continue after a
terminal disconnect:

```bash
nohup bash scripts/run_all.sh \
  --data-root data \
  --weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output runs > salt_run.log 2>&1 < /dev/null &
```

The launcher creates `runs/summary.csv` and `runs/summary.json` after all six
evaluations. To summarize your completed evaluations manually:

```bash
python scripts/summarize.py --root runs --output outputs/summary
```

## Visualize Predictions

```bash
python visualize.py \
  --checkpoint checkpoints/levir_to_whu/seed111.pt \
  --data-root data \
  --weights weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --index 1 \
  --output outputs/example
```

This exports the RGB image, ground truth, continuous heatmap, raw score array,
source-thresholded mask, and a combined PNG/PDF. `--index` selects a sample in
the fixed target manifest. Heatmap normalization is for display only.
