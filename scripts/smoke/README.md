# Smoke Checks

These scripts are lightweight pipeline checks, not model-selection experiments.

- `check_dataloader_forward.py`: checks split CSV loading, image transforms, one DataLoader batch, Custom CNN forward pass, and loss computation.
- `check_custom_cnn_training.py`: runs exactly one epoch to verify training, validation, metrics, confusion matrix, and checkpoint writing.

Example commands:

```bash
python scripts/smoke/check_dataloader_forward.py
python scripts/smoke/check_custom_cnn_training.py
```

Generated outputs are written under `outputs/` and `reports/` and should not be interpreted as clinical or final model performance.
