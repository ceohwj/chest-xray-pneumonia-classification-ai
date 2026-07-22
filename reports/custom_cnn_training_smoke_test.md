# Custom CNN Training Smoke Test

This one-epoch run verifies the PyTorch training pipeline only. It is not a model-selection result and must not be interpreted as clinical evidence.

## Purpose

- Confirm DataLoader, Custom CNN, loss, backward pass, optimizer step, validation loop, metric logging, and checkpoint saving work end to end.
- Avoid hyperparameter tuning, transfer learning, Grad-CAM, or performance conclusions.

## Run Configuration

- train split: `outputs/train_split.csv`
- validation split: `outputs/val_split.csv`
- model: `BaselineCNN`
- input shape: `[16, 3, 224, 224]`
- output shape: `[16, 2]`
- loss function: `CrossEntropyLoss`
- optimizer: `AdamW`
- learning rate: `0.001`
- batch size: `16`
- image size: `224`
- epochs: `1`
- device: `cpu`

## Metrics Produced

- train loss: `0.2883157194867684`
- validation loss: `0.22153761592887392`
- accuracy: `0.9146692233940557`
- sensitivity / PNEUMONIA recall: `0.9238709677419354`
- specificity / NORMAL recall: `0.8880597014925373`
- precision: `0.9597855227882037`
- F1-score: `0.9414858645627876`
- AUROC: `0.9722676937891189`
- FN count: `59`
- FP count: `30`
- confusion matrix [[TN, FP], [FN, TP]]: `[[238, 30], [59, 716]]`

## Output Files

- checkpoint: `outputs/checkpoints/custom_cnn_smoke_epoch1.pt`
- log: `outputs/logs/custom_cnn_smoke_train_log.csv`
- metrics: `outputs/metrics/custom_cnn_smoke_metrics.json`
- confusion matrix: `outputs/confusion_matrices/custom_cnn_smoke_confusion_matrix.csv`

## Verification

- one epoch completed: `True`
- backward pass and optimizer step completed: `True`
- validation loop completed: `True`
- checkpoint saved: `True`
- metrics saved: `True`
- no clinical validity claimed: `True`

## Limitations

- Patient-level identifiers are unavailable, so patient-level leakage cannot be fully excluded.
- One train/test near-duplicate risk remains reported as a limitation; the test set was not modified.
- One epoch is insufficient for performance conclusions.
- Accuracy alone is not sufficient evidence of model quality.