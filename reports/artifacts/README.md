# Curated Experiment Artifacts

This directory contains the small, reviewable subset promoted from the ignored
`outputs/` directory. It is intended to make the reported results auditable
without committing checkpoints, submissions, or large generated files.

## Contents

### `baseline/`

- Reusable run configuration with repository-relative paths.
- Best and final metric snapshots.
- Confusion matrix, training history, FN list, and FP list.
- Generated baseline run report.

### `transfer_learning/`

Six locally available runs are preserved:

- DenseNet121 frozen and fine-tuned.
- ResNet50 frozen and fine-tuned.
- EfficientNet-B0 frozen and fine-tuned.

Each run includes its normalized config, best/final metrics, confusion matrix,
training history, FN list, and FP list. Absolute local paths were replaced with
repository-relative paths. `comparison.csv` provides the common metric table
for all six preserved runs. The additional ConvNeXt-Tiny and EfficientNet-B3
runs documented historically were not available in the current `outputs/`
worktree and were not reconstructed.

### `ensemble_384/`

- Ensemble and checkpoint manifests without model weights.
- Checkpoint-loading verification results.
- OOF ensemble configuration, fold metrics, and 5,216 OOF predictions.
- Threshold 0.30/0.40/0.50 metric trade-offs and FN/FP case tables.
- Model comparison, probability correlation, and error-overlap tables.
- Pseudo-label configuration and checkpoint-compatibility summary.

## Deliberately excluded

- `.pt`, `.pth`, and other checkpoint files.
- The 883 MB packaged-model archive.
- Submission files and raw test probabilities.
- Generated Grad-CAM images containing source X-rays.
- Repeated intermediate outputs and caches.

The source `outputs/` directory remains local and ignored. These curated copies
are the stable Git-facing evidence for reports and reproducibility checks.
