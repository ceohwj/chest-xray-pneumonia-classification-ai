# Pre-baseline Dataset Audit Report

This project is for research and education only. This audit does not establish clinical diagnostic validity.

## 1. Dataset summary

- train rows: 5216
- test rows: 624
- sample submission rows: 624
- train columns: file_name, label
- test columns: file_name
- sample submission columns: file_name, label
- label mapping status: CONFIRMED. label 0 = NORMAL, label 1 = PNEUMONIA.
- image directory status: `data/images` exists = True
- file existence status: train missing=0, test missing=0
- unreadable/corrupted image status: 0 unreadable files

## 2. Class distribution

| label | class_name | count | ratio |
| --- | --- | --- | --- |
| 0 | NORMAL | 1341 | 0.2570935582822086 |
| 1 | PNEUMONIA | 3875 | 0.7429064417177914 |

- ratios: {'label_0 (NORMAL)': '0.257', 'label_1 (PNEUMONIA)': '0.743'}
- interpretation: Class imbalance is meaningful; use class-aware metrics and consider class weights, WeightedRandomSampler, threshold tuning, and recall/sensitivity-focused evaluation.
- Later evaluation must include accuracy, sensitivity/recall, specificity, precision, F1-score, AUROC, PR-AUC, and confusion matrix.

## 3. Patient-level leakage check

- patient_id availability: not available
- inferred grouping attempt: 0 non-null inferred groups
- patient-level split possible: False
- split decision: stratified
- limitation: Patient-level split cannot be verified from the provided CSV and image file names.

## 4. Duplicate and near-duplicate check

- exact duplicate hash groups: 9
- perceptual hash duplicate groups: 607
- train/validation leakage risk: True
- train/test leakage risk: True
- details saved to `reports/tables/duplicate_image_report.csv`.

## 5. Shortcut-learning artifact check

- Generated random sample grids, average images, average border/corner crops, image size plots, aspect ratio plots, brightness/contrast plots, and border intensity plots.
- These figures provide evidence for possible border, cropping, brightness, and resolution shortcuts, but manual review is still required before training.
- Shortcut probe classifier was not trained in this first audit deliverable, to avoid starting model training before review.

## 6. Recommended preprocessing

- Resize: start with 224x224.
- Channel handling: if images are grayscale, keep one-channel for Custom CNN if desired; convert to 3-channel for ImageNet pretrained models.
- Normalization: use ImageNet normalization for pretrained models first; consider dataset mean/std after reviewing split statistics.
- Augmentation: avoid vertical flip. Consider mild rotation, small translation, small scaling, and brightness/contrast adjustment. Use horizontal flip only if accepted for this dataset.
- If border/text/device artifacts appear label-associated, crop, mask, or use lung-region-focused preprocessing before model training.

## 7. Baseline-ready split decision

- selected split method: stratified
- train split count by label: label_0 (NORMAL): 1073, label_1 (PNEUMONIA): 3099
- validation split count by label: label_0 (NORMAL): 268, label_1 (PNEUMONIA): 776
- limitation: Patient-level split cannot be verified from the provided CSV and image file names.
- test.csv was not used for validation, early stopping, threshold tuning, or model selection.

## 8. Risk checklist

| Check item | Status | Evidence | Action |
| --- | --- | --- | --- |
| CSV files loaded | PASS | train=(5216, 2), test=(624, 1), sample_submission=(624, 2) | Continue. |
| Image files mapped to CSV | PASS | missing=0, unreadable=0 | Continue. |
| Patient-level split verified | WARN | no patient identifier | Patient-level split cannot be verified from the provided CSV and image file names. |
| Same patient across splits absent | WARN | Not independently verifiable without patient IDs. | Treat split as temporary if patient IDs are unavailable. |
| Class imbalance checked | PASS | counts=[{'label': '0', 'count': 1341}, {'label': '1', 'count': 3875}] | Use non-accuracy metrics and imbalance-aware training options. |
| Duplicate images checked | WARN | exact_groups=9 | Review duplicate_image_report.csv. |
| Near-duplicate images checked | WARN | perceptual_hash_groups=607 | Review perceptual hash groups before trusting validation metrics. |
| Border/text/device shortcut checked | WARN | Figures generated; manual radiographic artifact review is still required. | Inspect random samples, average images, and border/corner plots. |
| Test set not used for validation | PASS | Only train.csv was split into train/val. | Do not tune thresholds or early stopping on test.csv. |
| Label mapping confirmed | PASS | CONFIRMED. label 0 = NORMAL, label 1 = PNEUMONIA. | Use PNEUMONIA recall/sensitivity and NORMAL specificity in later evaluation. |
