# Chest X-ray NORMAL vs PNEUMONIA Classification

[한국어](README.md) | **English**

PyTorch portfolio project for classifying chest X-ray images as `NORMAL` or
`PNEUMONIA`. The work emphasizes reproducible experiments, false-negative and
false-positive analysis, threshold trade-offs, and Grad-CAM review rather than
accuracy alone.

> This repository is for research and education only. It is not a clinical
> diagnostic system, medical device, or substitute for review by a qualified
> healthcare professional.

## Project highlights

- Custom CNN baseline followed by transfer-learning comparisons.
- DenseNet121, EfficientNet, ConvNeXt, and ensemble experiments.
- Duplicate-aware and stricter duplicate-group split audits.
- Accuracy, precision, sensitivity, specificity, F1-score, AUROC, confusion
  matrix, FN, and FP reporting.
- OOF threshold analysis with explicit sensitivity/specificity trade-offs.
- Grad-CAM review of correct predictions, false negatives, and false positives.

## Selected experimental results

| Experiment | Evaluation protocol | Accuracy | Sensitivity | Specificity | F1 | AUROC | FN | FP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom CNN, best checkpoint | 1,043-image validation split | 0.9770 | 0.9884 | 0.9440 | 0.9846 | 0.9960 | 9 | 15 |
| DenseNet121 fine-tune | Same validation split | 0.9760 | 0.9806 | 0.9627 | 0.9838 | 0.9962 | 15 | 10 |
| 384px ensemble, threshold 0.40 | 5-fold OOF, 5,216 images | 0.9919 | 0.9935 | 0.9873 | 0.9946 | 0.9995 | 25 | 17 |

The OOF ensemble result and the single validation results use different
evaluation protocols, so they must not be treated as a strict apples-to-apples
model ranking. Internal validation performance also does not establish clinical
validity or external generalization.

See the [final experiment report](reports/final_ensemble_pseudo_labeling_report.md)
for threshold analysis, confusion matrices, error analysis, and limitations.

## Repository layout

```text
.
├── data/                 # CSV metadata and reproducible splits; images excluded
├── docs/                 # Experiment and daily logs
├── kaggle/               # Kaggle-oriented training entry points
├── notebooks/            # Integrated notebook deliverables
├── reports/              # Portfolio reports, tables, and derived figures
├── scripts/              # Role-grouped runnable workflows
├── src/
│   ├── data/             # Audit, Dataset, transforms, and split utilities
│   ├── models/           # Custom CNN and transfer-model factories
│   └── metrics.py        # Binary metrics and FN/FP export helpers
├── DATASET.md            # Provenance, licensing limits, and data-use policy
├── PROJECT_STATUS.md
└── ROADMAP.md
```

Generated checkpoints, predictions, submissions, and Grad-CAM images are kept
under `outputs/` locally and are intentionally excluded from Git.

Small configs, metrics, OOF evidence, and FN/FP tables selected from those runs
are versioned under [`reports/artifacts/`](reports/artifacts/README.md). Exact
train/validation assignments are versioned under
[`data/splits/`](data/splits/README.md).

## Final submission artifacts

The project was submitted with these two authoritative artifacts:

- `notebooks/train_final_submission_ensemble.ipynb`: integrated final training,
  TTA inference, and pseudo-label fine-tuning notebook.
- `reports/submission/chest_xray_classification_report_2026-07-01.pdf`: final
  25-page project report.

The PDF remains local-only for now because it embeds source X-rays and Grad-CAM
contact sheets. The notebook is included in the repository without saved cell
outputs.

## Setup

Create a virtual environment, then install the shared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Google Colab-specific scripts additionally require the packages provided by the
Colab runtime.

## Data layout

The code expects the following local layout:

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── images/
    ├── train/
    └── test/
```

Image files are not distributed in this repository. Before reproducing the
experiments, obtain the dataset through an authorized channel and follow its
applicable terms. This project received the data from OZ Coding School as course
material. The original upstream publisher, immutable version, and image
redistribution license were not supplied, so the repository does not infer a
public dataset identity from matching class names or image counts. See
[`DATASET.md`](DATASET.md) for the known provenance and repository policy.

## Basic verification

Run the dataset audit and smoke checks from the repository root:

```bash
python -m src.data.audit
python -m src.data.duplicate_aware_split
python scripts/smoke/check_dataloader_forward.py
python scripts/smoke/check_custom_cnn_training.py
```

The commands require the local images. They write generated artifacts under
`outputs/` and `reports/`.

## Experiment entry points

- `kaggle/train_custom_cnn_baseline.py`: formal Custom CNN baseline.
- `kaggle/train_transfer_model.py`: frozen and fine-tuned transfer models.
- `scripts/training/train_transfer_ensemble.py`: grouped cross-validation ensemble.
- `scripts/evaluation/evaluate_ensemble_backbones_oof_384.py`: 384px OOF evaluation.
- `scripts/explainability/generate_ensemble_gradcam_384.py`: ensemble Grad-CAM export.
- `scripts/verification/verify_packaged_ensemble.py`: packaged ensemble verification.

See [scripts/README.md](scripts/README.md) for the role-based workflow index and
[kaggle/README.md](kaggle/README.md) for Kaggle entry points. The
[notebook guide](notebooks/README.md) identifies the integrated submission
notebook. Many later experiments were designed for Kaggle or Colab and retain
environment-specific defaults; review their arguments and path constants before
running them.

## Evaluation policy

`PNEUMONIA` is the positive class. In this project:

- FN: a PNEUMONIA image predicted as NORMAL.
- FP: a NORMAL image predicted as PNEUMONIA.
- Sensitivity: PNEUMONIA recall.
- Specificity: NORMAL recall.

FN is reviewed separately because it represents the most concerning error type
within this educational framing. Threshold selection must still consider the
sensitivity/specificity trade-off and must not be interpreted as a clinical
operating threshold.

## Limitations

- Patient identifiers are unavailable, so patient-level leakage cannot be fully
  ruled out even after duplicate-aware splitting.
- The dataset was supplied through OZ Coding School, but its original upstream
  publisher, version, and redistribution license remain unconfirmed.
- Some scripts contain Kaggle/Colab-specific path defaults.
- Model weights and generated artifacts are not versioned in Git.
- Grad-CAM can reveal model sensitivity but cannot validate lesion localization.
- Strong internal or OOF metrics do not prove external or clinical validity.

## Documentation

- [Project status](PROJECT_STATUS.md)
- [Roadmap](ROADMAP.md)
- [Dataset provenance and use](DATASET.md)
- [Experiment log](docs/EXPERIMENT_LOG.md)
- [Historical code references](docs/HISTORICAL_CODE_REFERENCES.md)
- [Report index](reports/README.md)
- [Curated experiment artifacts](reports/artifacts/README.md)
- [Reproducible data splits](data/splits/README.md)
- [Final ensemble and pseudo-labeling report](reports/final_ensemble_pseudo_labeling_report.md)

## License

No open-source license has been selected yet. Until a license is added, the code
should not be assumed to grant reuse or redistribution rights.
