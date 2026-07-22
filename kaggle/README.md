# Kaggle Workflow Guide

These files are Kaggle-facing entry points. Their names describe the action
first, followed by the experiment target.

| File | Purpose |
| --- | --- |
| [`train_custom_cnn_baseline.py`](train_custom_cnn_baseline.py) | Train the formal Custom CNN baseline |
| [`train_transfer_model.py`](train_transfer_model.py) | Train one frozen or fine-tuned transfer model |
| [`run_transfer_ensemble.py`](run_transfer_ensemble.py) | Run the reusable trainer in `scripts/training/` with Kaggle paths |
| [`train_ensemble_oof_384.py`](train_ensemble_oof_384.py) | Run the original 384px OOF ensemble experiment |
| [`train_strict_ensemble_384.py`](train_strict_ensemble_384.py) | Run the strict-split 384px ensemble ablation |
| [`finetune_pseudolabels_384.py`](finetune_pseudolabels_384.py) | Fine-tune the 384px ensemble with strict pseudo labels |

Supporting instructions:

- [Transfer ensemble setup](TRANSFER_ENSEMBLE_KAGGLE.md)
- [Additional transfer models](ADDITIONAL_MODELS_KAGGLE.md)

Most scripts retain Kaggle dataset mount defaults. Update the input dataset
name or pass the supported path arguments before running them. Generated files
must be written under `/kaggle/working`, not `/kaggle/input`.
