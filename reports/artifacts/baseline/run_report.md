# Custom CNN Baseline Report

This report is for a research/education portfolio project. It does not establish clinical diagnostic validity.

## 1. Purpose

This is the first formal Custom CNN baseline after dataset audit, duplicate-aware split creation, and smoke testing.

## 2. Dataset split

- train split size: 4173
- validation split size: 1043
- duplicate-aware split status: no duplicate_group_id should cross train/validation in the prepared split.
- patient-level leakage limitation: patient identifiers are unavailable, so patient-level leakage cannot be fully excluded.

### Train class distribution

| Label | Class | Count | Percent |
| --- | --- | --- | --- |
| 0 | NORMAL | 1073 | 25.71% |
| 1 | PNEUMONIA | 3100 | 74.29% |

### Validation class distribution

| Label | Class | Count | Percent |
| --- | --- | --- | --- |
| 0 | NORMAL | 268 | 25.70% |
| 1 | PNEUMONIA | 775 | 74.30% |

## 3. Model architecture

- input shape: [batch_size, 3, 224, 224]
- output shape: [batch_size, 2]
- CNN blocks: 4 x Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
- pooling: AdaptiveAvgPool2d
- dropout: 0.4
- trainable parameter count: 389,410

## 4. Training setup

- seed: 42
- image size: 224
- batch size: 32
- epochs configured: 25
- epochs run: 23
- optimizer: AdamW
- learning rate: 0.001
- scheduler: ReduceLROnPlateau
- loss function: CrossEntropyLoss
- class weighting: [1.9445480108261108, 0.6730645298957825]
- augmentation: resize, mild rotation, small translation, brightness/contrast adjustment; no vertical flip.

## 5. Validation metrics

- best epoch by validation F1-score: 18
- best validation loss: 0.07070759059602354
- best accuracy: 0.9769894534995206
- best precision: 0.9807938540332907
- best sensitivity / PNEUMONIA recall: 0.9883870967741936
- best specificity / NORMAL recall: 0.9440298507462687
- best F1-score: 0.9845758354755784
- best AUROC: 0.9960471834376505
- best PR-AUC: 0.9986248589516526
- final epoch metrics: {"epoch": 23, "train_loss": 0.08855247525589774, "val_loss": 0.06794782293101456, "accuracy": 0.975071907957814, "precision": 0.9844760672703752, "recall": 0.9819354838709677, "sensitivity_pneumonia_recall": 0.9819354838709677, "specificity_normal_recall": 0.9552238805970149, "f1": 0.9832041343669251, "auroc": 0.9966490129995185, "pr_auc": 0.9988526950600707, "fn": 14, "fp": 12, "lr": 6.25e-05, "epoch_seconds": 15.558913469314575, "confusion_matrix": [[256, 12], [14, 761]], "best_epoch": 18}

## 6. Confusion matrix

| Actual \ Predicted | NORMAL | PNEUMONIA |
| --- | --- | --- |
| NORMAL | 253 | 15 |
| PNEUMONIA | 9 | 766 |

## 7. FN/FP review

- FN count (true PNEUMONIA predicted NORMAL): 9
- FP count (true NORMAL predicted PNEUMONIA): 15
- Do not overinterpret individual samples without image review.

## 8. Limitations

- Custom CNN baseline only.
- Patient-level leakage cannot be fully excluded.
- Train/test near-duplicate limitation remains.
- Internal validation only.
- No clinical validity claim.
- External validation would be required for clinical use.
- Do not select a model based on accuracy alone.

## 9. Next steps

- ResNet18/ResNet50 transfer learning.
- DenseNet121 comparison.
- EfficientNet-B0/B1 comparison.
- Grad-CAM analysis.
- Threshold tuning on validation set.
- FN/FP image review.