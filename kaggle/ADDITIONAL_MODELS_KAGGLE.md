# Additional Single-Model Experiments on Kaggle

This pipeline is for research/education portfolio experiments only. It is not a clinical diagnostic system.

## Models

`kaggle/train_transfer_model.py` now supports these additional single-model experiments without replacing the existing models:

- `densenet201` with input size `224`
- `efficientnet_v2_s` with input size `384`
- `resnext50_32x4d` with input size `224`

Existing `densenet121`, `efficientnet_b0`, and `resnet50` experiments remain available.

## Kaggle Inputs

Attach a Kaggle dataset containing:

```text
train_split.csv
val_split.csv
test.csv
sample_submission.csv
images/
```

The script keeps the existing `train_split.csv` and `val_split.csv` unchanged and checks that `duplicate_group_id` does not overlap between train and validation.

## Smoke Test

Run this first:

```bash
python /kaggle/input/<CODE_DATASET_NAME>/kaggle/train_transfer_model.py \
  --model-name densenet201 \
  --smoke-test \
  --batch-size 8
```

Smoke mode:

- trains for 1 epoch
- uses small train/validation samples
- checks train/validation/test DataLoader shapes
- prints label distribution and prediction probability range
- writes and validates `submission.csv`

## Full Training

Run one model at a time:

```bash
python /kaggle/input/<CODE_DATASET_NAME>/kaggle/train_transfer_model.py \
  --model-name densenet201 \
  --epochs 10 \
  --batch-size 32
```

```bash
python /kaggle/input/<CODE_DATASET_NAME>/kaggle/train_transfer_model.py \
  --model-name efficientnet_v2_s \
  --epochs 10 \
  --batch-size 16
```

```bash
python /kaggle/input/<CODE_DATASET_NAME>/kaggle/train_transfer_model.py \
  --model-name resnext50_32x4d \
  --epochs 10 \
  --batch-size 32
```

## Outputs

Each run writes to:

```text
/kaggle/working/outputs/{model_name}/
```

Required files include:

- `config.json`
- `train_log.csv`
- `best_metrics.json`
- `final_metrics.json`
- `confusion_matrix.csv`
- `false_negatives.csv`
- `false_positives.csv`
- `val_predictions.csv`
- `test_predictions.csv`
- `submission.csv`
- `best_model.pth`

`submission.csv` preserves the columns and row order from `sample_submission.csv`, and writes P(PNEUMONIA) probabilities as the submission values.
