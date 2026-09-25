<p align="center">
  <img src="./assets/project-banner.svg" width="100%" alt="YOLO11 dangerous-person detection research project banner" />
</p>

<p align="center">
  <a href="https://github.com/Nambekai/dangerous-person-detection-yolo11/actions/workflows/quality.yml"><img src="https://github.com/Nambekai/dangerous-person-detection-yolo11/actions/workflows/quality.yml/badge.svg" alt="Repository quality status" /></a>
  <img src="https://img.shields.io/badge/status-research%20prototype-0B1F33" alt="Research prototype status" />
  <img src="https://img.shields.io/badge/model-YOLO11s-8C1D40" alt="Selected model YOLO11s" />
  <img src="https://img.shields.io/badge/dataset-3%2C140%20images-30475E" alt="Dataset with 3,140 images" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0-A67C00" alt="AGPL-3.0 license" />
</p>

## Overview

This graduation-thesis repository studies YOLO11-based detection of people who appear with a potentially dangerous object in public-area imagery. The detector localizes the person, not the object alone, using two contextual classes: `normal_person` and `potentially_dangerous_person`.

The study compared YOLO11n, YOLO11s, YOLO11s with a recorded preprocessing configuration, YOLO11m, and YOLO11l. YOLO11s provided the strongest overall balance and was selected for the demonstration.

> This is an academic research prototype, not an autonomous security decision system. A prediction is a prompt for human review. It does not establish identity, intent, guilt, or future behavior.

## Key Results

| Measure | YOLO11s result |
| --- | ---: |
| Precision | 0.87 |
| Recall | 0.69 |
| F1 score | 0.77 |
| mAP50 | 0.80 |
| mAP50-95 | 0.64 |
| Test images | 314 |

All reported measures come from the held-out test set described in the thesis. The repository does not claim independent replication beyond the supplied artifacts.

<p align="center">
  <img src="./assets/system-workflow.svg" width="100%" alt="Four-stage research workflow covering data, training, evaluation, and review" />
</p>

<p align="center"><sub>Figure 1. Research workflow used to prepare data, compare models, evaluate results, and review outputs.</sub></p>

## Dataset

The final YOLO-format dataset contains 3,140 images and 10,595 labeled person instances.

| Split | Images | Normal-person instances | Potentially-dangerous-person instances | Total instances |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,512 | 6,170 | 2,298 | 8,468 |
| Validation | 314 | 968 | 356 | 1,324 |
| Test | 314 | 539 | 264 | 803 |
| Total | 3,140 | 7,677 | 2,918 | 10,595 |

Dataset access:

- [Google Drive dataset folder](https://drive.google.com/drive/folders/1m-BzFvQwwWzvTCXApwJftQU3tRt0CmCC?usp=drive_link)
- [Versioned GitHub release](https://github.com/Nambekai/dangerous-person-detection-yolo11/releases/tag/v1.0.0)
- [Dataset card](./docs/DATASET_CARD.md)

The embedded Roboflow export notice identifies the dataset as CC BY 4.0. Review the dataset card and source notices before reuse.

## Model Comparison

<p align="center">
  <img src="./assets/model-comparison.svg" width="100%" alt="Comparison of precision, recall, F1, mAP50, and mAP50-95 across five YOLO11 variants" />
</p>

<p align="center"><sub>Figure 2. Held-out test-set comparison. YOLO11s led the aggregate F1, mAP50, and mAP50-95 measures.</sub></p>

<details>
<summary><strong>Training evidence</strong></summary>

<p align="center">
  <img src="./assets/results/yolo11s-training-curves.png" width="96%" alt="YOLO11s training and validation curves" />
</p>

<p align="center"><sub>Figure 3. YOLO11s mAP and loss curves over the recorded 132 training epochs.</sub></p>

<p align="center">
  <img src="./assets/results/yolo11s-training-results.png" width="96%" alt="YOLO11s detailed training results" />
</p>

<p align="center"><sub>Figure 4. Detailed YOLO11s training, validation, precision, recall, and mAP traces.</sub></p>

</details>

## Prediction Gallery

The following images are preserved as original model outputs for auditability. Their embedded class identifiers retain the original training labels; all repository captions and explanatory text are in US English.

<table>
  <tr>
    <td width="50%" align="center"><img src="./assets/predictions/test-0194-public-area.jpg" width="100%" alt="Crowded public-area test prediction" /><br /><sub>Figure 5. Crowded public-area test frame with multiple localized people.</sub></td>
    <td width="50%" align="center"><img src="./assets/predictions/test-0247-hard-negative.jpg" width="100%" alt="Hard-negative test prediction" /><br /><sub>Figure 6. Hard-negative example classified as a normal person.</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><img src="./assets/predictions/test-0124.jpg" width="100%" alt="Low-light corridor test prediction" /><br /><sub>Figure 7. Low-light corridor example with a potentially dangerous person.</sub></td>
    <td width="50%" align="center"><img src="./assets/predictions/test-0243.jpg" width="100%" alt="Public-building corridor prediction" /><br /><sub>Figure 8. Public-building corridor example with two localized people.</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center"><img src="./assets/predictions/street-scene.jpg" width="100%" alt="Street-scene prediction" /><br /><sub>Figure 9. External street scene used for qualitative review.</sub></td>
    <td width="50%" align="center"><img src="./assets/predictions/public-area-scene.jpg" width="100%" alt="Public-area prediction" /><br /><sub>Figure 10. External public-area scene used for qualitative review.</sub></td>
  </tr>
</table>

<details>
<summary><strong>Additional test-set galleries</strong></summary>

<p align="center"><img src="./assets/results/test-prediction-gallery-a.png" width="100%" alt="First extended test-prediction gallery" /></p>
<p align="center"><sub>Figure 11. Extended test-set review, group A.</sub></p>

<p align="center"><img src="./assets/results/test-prediction-gallery-b.png" width="100%" alt="Second extended test-prediction gallery" /></p>
<p align="center"><sub>Figure 12. Extended test-set review, group B.</sub></p>

<p align="center"><img src="./assets/results/test-prediction-gallery-c.png" width="100%" alt="Third extended test-prediction gallery" /></p>
<p align="center"><sub>Figure 13. Extended test-set review, group C.</sub></p>

</details>

## Quick Start

### 1. Clone and prepare Python

```bash
git clone https://github.com/Nambekai/dangerous-person-detection-yolo11.git
cd dangerous-person-detection-yolo11
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate `.venv` with the command appropriate to your operating system before installing dependencies.

### 2. Download a model

Download `weights-YOLO11s.zip` from the [v1.0.0 release](https://github.com/Nambekai/dangerous-person-detection-yolo11/releases/tag/v1.0.0), extract it, and locate `best.pt`.

### 3. Run inference

```bash
python scripts/inference.py \
  --model path/to/best.pt \
  --source path/to/image-or-video \
  --confidence 0.35
```

Outputs are written under `runs/predict/`. The inference script changes displayed class names to English without modifying the trained class indices.

### 4. Reconstruct a training run

Extract the dataset so that `data/dataset/train`, `data/dataset/valid`, and `data/dataset/test` exist, then run:

```bash
python scripts/train.py --variant yolo11s --data config/data.yaml
```

The training utility reconstructs settings recorded in the thesis. The original notebooks, package lock file, random seed, and automatically selected optimizer details were not present in the supplied folder, so exact numerical reproduction is not guaranteed. See the [reproducibility guide](./docs/REPRODUCIBILITY.md).

## Recorded Training Configuration

| Variant | Base model | Epochs set | Epochs completed | Patience | Batch | Image size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | `yolo11n.pt` | 150 | 109 | 35 | 8 | 832 |
| YOLO11s | `yolo11s.pt` | 150 | 132 | 35 | 8 | 832 |
| YOLO11s preprocessing variant | `yolo11s.pt` | 150 | 150 | 35 | 8 | 832 |
| YOLO11m | `yolo11m.pt` | 150 | 150 | 40 | 8 | 832 |
| YOLO11l | `yolo11l.pt` | 150 | 150 | 40 | 8 | 832 |

The recorded shared settings were optimizer `auto` and an NVIDIA Tesla T4 GPU. The preprocessing variant settings are preserved in [`config/experiments.yaml`](./config/experiments.yaml).

## Repository Map

```text
.
|-- assets/                 Figures and prediction examples
|-- config/                 Dataset and experiment configuration
|-- docs/                   Dataset card, model card, thesis, slides, and guides
|-- scripts/                Training, inference, and quality-validation utilities
|-- tests/                  Lightweight repository tests
|-- CITATION.cff            Machine-readable citation metadata
|-- LICENSE                 AGPL-3.0 license
`-- README.md               Project overview and entry point
```


The thesis and defense deck are preserved in their original Vietnamese submission language. All repository-facing documentation, labels, captions, configuration, and guidance are provided in United States English. The deck contains the 26-slide defense sequence followed by 14 original backup slides.

## Responsible Use

- Keep a trained human reviewer in the decision loop.
- Do not use the model for identity recognition, demographic inference, predictive policing, or automated punitive action.
- Validate performance in the actual deployment environment and document subgroup, lighting, distance, crowd-density, and camera-angle effects.
- Establish lawful data governance, retention, access control, incident review, and an appeal process before any field use.
- Treat false negatives as a safety risk and false positives as a potential harm to affected people.

