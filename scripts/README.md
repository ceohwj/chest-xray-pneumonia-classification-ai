# Workflow Script Guide

Runnable code is grouped by what it does. File names follow this order:

```text
<action>_<target>_<environment>_<image-size>.py
```

`environment` and `image-size` are included only when they distinguish the
workflow. Run package-based scripts from the repository root with `python -m`.

## Directory map

```text
scripts/
├── training/          # training, fine-tuning, and stacking
├── evaluation/        # OOF and checkpoint evaluation
├── inference/         # prediction and submission generation
├── explainability/    # Grad-CAM generation
├── reporting/         # report figure generation
├── verification/      # packaged checkpoint verification
└── smoke/             # fast pipeline checks
```

## Training

| Module | Purpose | Environment |
| --- | --- | --- |
| `scripts.training.train_transfer_ensemble` | Train the reusable grouped-CV transfer ensemble | Local/Kaggle wrapper |
| `scripts.training.fit_oof_stacking_colab` | Fit a logistic-regression stacker from saved OOF probabilities | Colab |
| `scripts.training.finetune_pseudolabels_colab_384_strict` | Fine-tune with strict high-confidence pseudo labels | Colab |
| `scripts.training.finetune_pseudolabels_colab_384_weighted` | Fine-tune with weighted pseudo labels | Colab |
| `scripts.training.finetune_progressive_colab_512` | Continue 384px checkpoints at 512px | Colab |

Reusable trainer example:

```bash
python -m scripts.training.train_transfer_ensemble --help
```

The Colab modules mount Google Drive at startup and should be run only after
reviewing their project paths and input checkpoint names.

## Evaluation

| Module | Purpose |
| --- | --- |
| `scripts.evaluation.evaluate_ensemble_backbones_oof_384` | Recreate 384px per-backbone OOF predictions and metrics |

```bash
python -m scripts.evaluation.evaluate_ensemble_backbones_oof_384 --help
```

## Inference

| Module | Purpose | Environment |
| --- | --- | --- |
| `scripts.inference.generate_custom_cnn_submission` | Generate a submission from the Custom CNN checkpoint | Local/Kaggle |
| `scripts.inference.create_threshold_submissions` | Convert saved probabilities into thresholded submissions | Local/Kaggle |
| `scripts.inference.run_ensemble_inference_colab` | Run ensemble TTA, weight selection, and submission generation | Colab |

```bash
python -m scripts.inference.generate_custom_cnn_submission --help
python -m scripts.inference.create_threshold_submissions --help
```

## Explainability

| Module | Purpose |
| --- | --- |
| `scripts.explainability.generate_densenet_cbam_gradcam_384` | DenseNet121-CBAM Grad-CAM generation |
| `scripts.explainability.generate_ensemble_gradcam_384` | Grad-CAM generation for all 384px ensemble backbones |

```bash
python -m scripts.explainability.generate_densenet_cbam_gradcam_384 --help
python -m scripts.explainability.generate_ensemble_gradcam_384 --help
```

## Reporting and verification

| Module | Purpose |
| --- | --- |
| `scripts.reporting.generate_final_report_figures` | Rebuild derived figures for the final report |
| `scripts.verification.verify_packaged_ensemble` | Load and smoke-test a packaged ensemble |

```bash
python -m scripts.reporting.generate_final_report_figures
python -m scripts.verification.verify_packaged_ensemble --help
```

## Smoke checks

- `smoke/check_dataloader_forward.py`: one-batch Dataset/DataLoader and forward
  pass verification.
- `smoke/check_custom_cnn_training.py`: short Custom CNN training and checkpoint
  verification.

```bash
python scripts/smoke/check_dataloader_forward.py --help
python scripts/smoke/check_custom_cnn_training.py --help
```

## Kaggle workflows

Kaggle-specific entry points are intentionally kept in [`../kaggle`](../kaggle)
instead of being mixed with local modules. See [`../kaggle/README.md`](../kaggle/README.md).

## Historical references

The experiment logs mention three earlier utilities that are not present in the
current worktree. They are documented explicitly in
[`../docs/HISTORICAL_CODE_REFERENCES.md`](../docs/HISTORICAL_CODE_REFERENCES.md)
and are not part of the runnable workflow list above.

## Before running a workflow

1. Confirm its expected CSV, image, and checkpoint paths.
2. Use a new output directory so earlier artifacts remain intact.
3. Record the seed, split, preprocessing, threshold, and dependency versions.
4. Do not compare metrics across different evaluation protocols as though they
   came from one experiment.
