# Transfer Ensemble on Kaggle

This is a research/education portfolio experiment, not a clinical diagnostic system.

## Required Kaggle Inputs

Attach a Kaggle dataset that contains the same structure as:

```text
data/
  images/
    train/
    test/
  test.csv
  sample_submission.csv
outputs/
  train_split.csv
  val_split.csv
```

The existing local archive is:

```text
kaggle_upload/chest_xray_kaggle_dataset_clean.zip
```

Upload that zip as a Kaggle Dataset, then attach it to the notebook.

Also attach this repository/code, or upload the files so this exists in Kaggle:

```text
scripts/training/train_transfer_ensemble.py
kaggle/run_transfer_ensemble.py
```

## Debug Smoke Test

Run this first on Kaggle GPU:

```bash
python /kaggle/input/<CODE_DATASET_NAME>/kaggle/run_transfer_ensemble.py \
  --debug \
  --models densenet121 \
  --image-size 224 \
  --epochs 2 \
  --num-folds 2 \
  --batch-size 8
```

If auto-detection fails, pass the data/code roots explicitly:

```bash
python /kaggle/input/<CODE_DATASET_NAME>/kaggle/run_transfer_ensemble.py \
  --data-root /kaggle/input/<DATASET_NAME> \
  --project-root /kaggle/input/<CODE_DATASET_NAME> \
  --debug \
  --models densenet121 \
  --image-size 224 \
  --epochs 2 \
  --num-folds 2 \
  --batch-size 8
```

## Full Training

After debug passes:

```bash
python /kaggle/input/<CODE_DATASET_NAME>/kaggle/run_transfer_ensemble.py \
  --models densenet121 convnext_tiny efficientnet_b3 \
  --image-size 384 \
  --epochs 10 \
  --num-folds 5 \
  --batch-size 16 \
  --seed 42
```

## Outputs

The runner writes to `/kaggle/working`:

```text
/kaggle/working/transfer_ensemble/
  oof_densenet121.csv
  oof_convnext_tiny.csv
  oof_efficientnet_b3.csv
  oof_ensemble.csv
  best_oof_ensemble_config.json
  test_probs.csv
/kaggle/working/metrics/
  transfer_ensemble_fold_metrics.csv
/kaggle/working/submissions/
  submission.csv
  submission_threshold_0.30.csv
  submission_threshold_0.40.csv
  submission_threshold_0.50.csv
  submission_threshold_0.60.csv
  submission_threshold_0.70.csv
```
