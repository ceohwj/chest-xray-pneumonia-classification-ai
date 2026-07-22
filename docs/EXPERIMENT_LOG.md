# Experiment Log

> Historical paths are preserved for traceability. See
> [Historical Code References](HISTORICAL_CODE_REFERENCES.md) for utilities that
> are referenced by past runs but are not retained in the current worktree.

Reusable template for model experiments. Add one completed entry per experiment run.

## Experiment Template

```markdown
## Experiment ID

Date:
Model:
Config:
Dataset split:
Checkpoint:
Output directory:

### Purpose

- What question does this experiment answer?

### Setup

- Seed:
- Image size:
- Batch size:
- Epochs:
- Optimizer:
- Learning rate:
- Scheduler:
- Loss function:
- Class handling:

### Metrics

| Metric | Value |
| --- | --- |
| Accuracy | TBD |
| Sensitivity / PNEUMONIA recall | TBD |
| Specificity / NORMAL recall | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-score | TBD |
| AUROC | TBD |

### Confusion Matrix

|  | Predicted NORMAL | Predicted PNEUMONIA |
| --- | ---: | ---: |
| Actual NORMAL | TBD | TBD |
| Actual PNEUMONIA | TBD | TBD |

### FN/FP Review

- FN count:
- FP count:
- FN definition: PNEUMONIA predicted as NORMAL.
- FP definition: NORMAL predicted as PNEUMONIA.
- Notable FN samples:
- Notable FP samples:

### Grad-CAM Review

- Correct samples reviewed:
- FN samples reviewed:
- FP samples reviewed:
- Observed focus regions:
- Possible artifact attention:

### Notes

- What worked:
- What failed:
- Limitations:
- Next experiment:
```

## Experiments

## EXP-001-custom-cnn-baseline

Date: 2026-06-24\
Model: Custom CNN baseline\
Config: seed 42, image size 224, batch size 32, max epochs 25, AdamW, learning rate 1e-3, ReduceLROnPlateau, CrossEntropyLoss with class weights\
Dataset split: duplicate-aware split; train 4,173 images, validation 1,043 images\
Checkpoint: `/kaggle/working/custom_cnn_baseline/checkpoints/best_checkpoint.pt`\
Output directory: `/kaggle/working/custom_cnn_baseline`

### Purpose

- Establish the first formal Custom CNN baseline after pre-baseline audit, duplicate-aware split creation, DataLoader smoke test, and one-epoch pipeline smoke test.
- Produce internal validation metrics and FN/FP exports for cautious baseline review.
- This experiment is not a clinical diagnostic evaluation.

### Setup

- Seed: 42
- Image size: 224x224
- Batch size: 32
- Epochs: configured for 25; early stopping stopped at epoch 23
- Optimizer: AdamW
- Learning rate: 1e-3 initial learning rate; final logged learning rate 6.25e-05
- Scheduler: ReduceLROnPlateau
- Loss function: CrossEntropyLoss
- Class handling: class weights from train split imbalance, `[1.9445480108261108, 0.6730645298957825]`
- Device: Kaggle GPU, Tesla T4
- Trainable parameters: 389,410
- Train split class distribution: NORMAL 1,073; PNEUMONIA 3,100
- Validation split class distribution: NORMAL 268; PNEUMONIA 775

### Metrics

Best checkpoint selected by validation F1-score at epoch 18.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9769894535 |
| Sensitivity / PNEUMONIA recall | 0.9883870968 |
| Specificity / NORMAL recall | 0.9440298507 |
| Precision | 0.9807938540 |
| Recall | 0.9883870968 |
| F1-score | 0.9845758355 |
| AUROC | 0.9960471834 |
| PR-AUC | 0.9986248590 |
| Validation loss | 0.0707075906 |
| Train loss | 0.0937751048 |
| Public leaderboard score | 0.73237 |

Final epoch was epoch 23. Final epoch F1-score was `0.9832041344`, sensitivity/PNEUMONIA recall was `0.9819354839`, specificity/NORMAL recall was `0.9552238806`, AUROC was `0.9966490130`, and PR-AUC was `0.9988526951`.

The public leaderboard score was much lower than the internal validation metrics. This gap suggests that the internal validation result may not fully represent external/test-like generalization performance.

### Confusion Matrix

Best checkpoint confusion matrix. Rows are actual labels and columns are predicted labels.

|  | Predicted NORMAL | Predicted PNEUMONIA |
| --- | ---: | ---: |
| Actual NORMAL | 253 | 15 |
| Actual PNEUMONIA | 9 | 766 |

### FN/FP Review

- FN count: 9
- FP count: 15
- FN definition: PNEUMONIA predicted as NORMAL.
- FP definition: NORMAL predicted as PNEUMONIA.
- FN is the highest-risk error type in this project.
- Notable FN samples: Not reviewed yet.
- Notable FP samples: Not reviewed yet.
- Exported artifacts expected from Kaggle run: `false_negatives.csv`, `false_positives.csv`

### Grad-CAM Review

- Correct samples reviewed: Not performed
- FN samples reviewed: Not performed
- FP samples reviewed: Not performed
- Observed focus regions: Not available
- Possible artifact attention: Not evaluated yet

### Notes

- What worked:
  - The formal Custom CNN baseline trained successfully on Kaggle GPU.
  - Internal validation F1-score and PNEUMONIA recall were high for this baseline run.
  - Early stopping selected epoch 18 by validation F1-score.
  - Metrics include sensitivity, specificity, AUROC, PR-AUC, confusion matrix, FN count, and FP count rather than accuracy alone.
- What failed:
  - Grad-CAM was not performed yet.
  - FN/FP image review was not performed yet.
  - Internal validation metrics did not translate to a comparable public leaderboard score.
- Limitations:
  - This is a Custom CNN baseline, not the final model.
  - Patient-level leakage cannot be fully excluded because patient identifiers are unavailable.
  - Train/test near-duplicate limitation remains.
  - Internal validation does not prove clinical validity.
  - External validation is required before any clinical claim.
  - The result should be interpreted as a baseline result, not as evidence that the model can diagnose pneumonia.
- Artifacts generated:
  - `config.json`
  - `train_log.csv`
  - `best_metrics.json`
  - `final_metrics.json`
  - `confusion_matrix.csv`
  - `false_negatives.csv`
  - `false_positives.csv`
  - `custom_cnn_baseline_report.md`
  - `figures/confusion_matrix.png`
  - `figures/train_val_loss.png`
  - `figures/metric_curves.png`
  - `figures/roc_curve.png`
  - `figures/pr_curve.png`
  - `checkpoints/latest_checkpoint.pt`
  - `checkpoints/best_checkpoint.pt`
- Training observations:
  - Validation behavior fluctuated across epochs, especially specificity in several early/mid epochs.
  - Class weighting helped account for imbalance, but threshold behavior and FN/FP patterns still require review.
  - Best checkpoint balanced high PNEUMONIA recall with improved NORMAL specificity compared with several high-sensitivity/low-specificity epochs.
  - The public leaderboard score `0.73237` revealed a large gap from internal validation AUROC/F1-score.
  - Possible contributors include validation/test distribution shift, unresolved patient-level leakage risk, train/test near-duplicate limitation, shortcut learning from border/crop/brightness/text/device artifacts, class-imbalance threshold behavior, and limited feature extraction capacity of the Custom CNN baseline.
  - The key interpretation is not that the baseline simply failed, but that the experiment exposed a meaningful gap between internal validation and external/test-like evaluation.
- Next experiment:
  - Review FN/FP samples and run Grad-CAM analysis for correct, FN, and FP cases.
  - Compare against transfer learning baselines such as ResNet18, DenseNet121, and EfficientNet-B0.
  - Revisit validation design and threshold tuning after artifact review.

## EXP-002-transfer-learning-comparison

Date: 2026-06-30\
Model: Transfer learning comparison\
Config: seed 42, image size 224, batch size 32, max epochs 10, ReduceLROnPlateau, threshold 0.5\
Reproducible config file: `configs/exp002_transfer_learning_comparison.json`\
Dataset split: duplicate-aware split; train split from `outputs/train_split.csv`, validation split from `outputs/val_split.csv`\
Checkpoint: `outputs/transfer_learning/{experiment_name}/best_model.pth`\
Output directory: `outputs/transfer_learning`
Additional output directory: `outputs/transfer_learning_224_additional`
Merged comparison source: `outputs/transfer_learning_merged/comparison_summary.csv`
Merge script: `scripts/merge_transfer_learning_comparison.py`

### Purpose

- Compare available transfer learning runs using the same validation split and the same core metric set.
- Review frozen-head versus fine-tuning behavior where both are available.
- Prioritize PNEUMONIA recall, FN count, AUROC, and specificity rather than accuracy alone.
- This experiment is for research and portfolio comparison only, not clinical diagnostic validation.

### Setup

- Seed: 42
- Image size: 224x224
- Batch size: 32
- Epochs: 10
- Optimizer: AdamW, from `kaggle/train_transfer_model.py`
- Learning rate:
  - Frozen runs: head learning rate `1e-3`
  - Fine-tuning runs: backbone learning rate `1e-5`, head learning rate `1e-3`
- Scheduler: ReduceLROnPlateau
- Loss function: BCEWithLogitsLoss with `pos_weight` enabled by config field `use_pos_weight: true`, from `kaggle/train_transfer_model.py`
- Class handling: binary labels, NORMAL = 0 and PNEUMONIA = 1
- Threshold: 0.5
- Available completed runs included in the merged `comparison_summary.csv`:
  - `densenet121_frozen`
  - `densenet121_finetune`
  - `resnet50_frozen`
  - `resnet50_finetune`
  - `efficientnet_b0_frozen`
  - `efficientnet_b0_finetune`
  - `convnext_tiny_frozen`
  - `convnext_tiny_finetune`
  - `efficientnet_b3_frozen`
  - `efficientnet_b3_finetune`

### Metrics

Best checkpoint metrics from merged and refreshed `comparison_summary.csv` files.
Source files:
- `outputs/transfer_learning/comparison_summary.csv`
- `outputs/transfer_learning_224_additional/comparison_summary.csv`
- merged into `outputs/transfer_learning_merged/comparison_summary.csv`

| Experiment | Backbone | Regime | Best epoch | Accuracy | Sensitivity / PNEUMONIA recall | Specificity / NORMAL recall | Precision | F1-score | AUROC | FN | FP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| densenet121_finetune | DenseNet121 | finetune | 10 | 0.9760306807 | 0.9806451613 | 0.9626865672 | 0.9870129870 | 0.9838187702 | 0.9962060664 | 15 | 10 |
| efficientnet_b0_finetune | EfficientNet-B0 | finetune | 10 | 0.9702780441 | 0.9690322581 | 0.9738805970 | 0.9907651715 | 0.9797782127 | 0.9964756861 | 24 | 7 |
| convnext_tiny_finetune | ConvNeXt-Tiny | finetune | 10 | 0.9683604986 | 0.9612903226 | 0.9888059701 | 0.9959893048 | 0.9783322390 | 0.9967790082 | 30 | 3 |
| resnet50_finetune | ResNet50 | finetune | 9 | 0.9616490892 | 0.9729032258 | 0.9291044776 | 0.9754204398 | 0.9741602067 | 0.9917284545 | 21 | 19 |
| efficientnet_b3_finetune | EfficientNet-B3 | finetune | 10 | 0.9530201342 | 0.9432258065 | 0.9813432836 | 0.9932065217 | 0.9675711449 | 0.9943765046 | 44 | 5 |
| densenet121_frozen | DenseNet121 | frozen | 10 | 0.9511025887 | 0.9535483871 | 0.9440298507 | 0.9801061008 | 0.9666448659 | 0.9883630236 | 36 | 15 |
| resnet50_frozen | ResNet50 | frozen | 10 | 0.9367209971 | 0.9341935484 | 0.9440298507 | 0.9797023004 | 0.9564068692 | 0.9796726047 | 51 | 15 |
| efficientnet_b0_frozen | EfficientNet-B0 | frozen | 10 | 0.9300095877 | 0.9303225806 | 0.9291044776 | 0.9743243243 | 0.9518151815 | 0.9867886375 | 54 | 19 |
| efficientnet_b3_frozen | EfficientNet-B3 | frozen | 10 | 0.9252157239 | 0.9045161290 | 0.9850746269 | 0.9943262411 | 0.9472972973 | 0.9861001444 | 74 | 4 |
| convnext_tiny_frozen | ConvNeXt-Tiny | frozen | 6 | 0.9165867689 | 0.8954838710 | 0.9776119403 | 0.9914285714 | 0.9410169492 | 0.9857486760 | 81 | 6 |

### Confusion Matrix

Best checkpoint confusion matrices. Rows are actual labels and columns are predicted labels.

| Experiment | Actual NORMAL -> Pred NORMAL | Actual NORMAL -> Pred PNEUMONIA | Actual PNEUMONIA -> Pred NORMAL | Actual PNEUMONIA -> Pred PNEUMONIA |
| --- | ---: | ---: | ---: | ---: |
| densenet121_finetune | 258 | 10 | 15 | 760 |
| efficientnet_b0_finetune | 261 | 7 | 24 | 751 |
| convnext_tiny_finetune | 265 | 3 | 30 | 745 |
| resnet50_finetune | 249 | 19 | 21 | 754 |
| efficientnet_b3_finetune | 263 | 5 | 44 | 731 |
| densenet121_frozen | 253 | 15 | 36 | 739 |
| resnet50_frozen | 253 | 15 | 51 | 724 |
| efficientnet_b0_frozen | 249 | 19 | 54 | 721 |
| efficientnet_b3_frozen | 264 | 4 | 74 | 701 |
| convnext_tiny_frozen | 262 | 6 | 81 | 694 |

### FN/FP Review

- FN definition: PNEUMONIA predicted as NORMAL.
- FP definition: NORMAL predicted as PNEUMONIA.
- FN is the highest-risk error type in this project and should be reviewed separately.
- Lowest FN count among available transfer learning runs: `densenet121_finetune` with 15 FN.
- Lowest FP count among available transfer learning runs: `convnext_tiny_finetune` with 3 FP.
- Class-balanced Grad-CAM sample sets were generated for all 10 EXP-002 runs using the same sampling rule: 5 correct `NORMAL`, 5 correct `PNEUMONIA`, 3 FN, and 3 FP samples per run.
- Limited image-level review was performed for `densenet121_finetune`, `convnext_tiny_finetune`, and `efficientnet_b3_finetune` using Grad-CAM outputs and exported FN/FP lists.
- Reviewed error sample counts in this update:
  - `densenet121_finetune`: 3 FN, 3 FP
  - `convnext_tiny_finetune`: 3 FN, 3 FP
  - `efficientnet_b3_finetune`: 3 FN, 3 FP
- Available exported artifacts per completed run:
  - `false_negatives.csv`
  - `false_positives.csv`
  - `validation_predictions.csv`
  - `confusion_matrix.csv`
  - `best_metrics.json`
  - `final_metrics.json`
  - `train_log.csv`

### Grad-CAM Review

- Generated Grad-CAM outputs:
  - `outputs/gradcam/densenet121_frozen`
  - `outputs/gradcam/densenet121_finetune`
  - `outputs/gradcam/resnet50_frozen`
  - `outputs/gradcam/resnet50_finetune`
  - `outputs/gradcam/efficientnet_b0_frozen`
  - `outputs/gradcam/efficientnet_b0_finetune`
  - `outputs/gradcam/convnext_tiny_frozen`
  - `outputs/gradcam/convnext_tiny_finetune`
  - `outputs/gradcam/efficientnet_b3_frozen`
  - `outputs/gradcam/efficientnet_b3_finetune`
- Full-generation status:
  - All 10 EXP-002 runs now have matching `summary.json` and `metadata.csv` Grad-CAM artifacts under `outputs/gradcam/`.
- Qualitative review completed so far:
  - Detailed spot check:
    - `densenet121_finetune`
    - `convnext_tiny_finetune`
    - `efficientnet_b3_finetune`
  - Representative model-by-model pass:
    - `densenet121_frozen`
    - `resnet50_frozen`
    - `resnet50_finetune`
    - `efficientnet_b0_frozen`
    - `efficientnet_b0_finetune`
    - `convnext_tiny_frozen`
    - `efficientnet_b3_frozen`
- Correct samples reviewed:
  - `densenet121_finetune`: 5 NORMAL + 5 PNEUMONIA random correct samples, seed 42
  - `convnext_tiny_finetune`: 5 NORMAL + 5 PNEUMONIA random correct samples, seed 42
  - `efficientnet_b3_finetune`: 5 NORMAL + 5 PNEUMONIA random correct samples, seed 42
- FN samples reviewed:
  - `densenet121_finetune`: 3 samples
  - `convnext_tiny_finetune`: 3 samples
  - `efficientnet_b3_finetune`: 3 samples
- FP samples reviewed:
  - `densenet121_finetune`: 3 samples
  - `convnext_tiny_finetune`: 3 samples
  - `efficientnet_b3_finetune`: 3 samples
- Observed focus regions:
  - `densenet121_finetune` often showed broad lower-image or border-adjacent activation in the reviewed set. Some correct PNEUMONIA samples overlapped one lung field, but several correct and error cases remained diffuse rather than clearly lesion-centered.
  - `convnext_tiny_finetune` more often highlighted unilateral or bilateral lower-lung fields in reviewed PNEUMONIA and FN samples, although several correct and FP cases still leaned toward lateral chest or upper-lung regions rather than a clearly bounded opacity.
  - `efficientnet_b3_finetune` showed broader and less stable attention in the reviewed set, with several correct NORMAL, FN, and FP cases activating the upper image border, shoulder area, or diffuse central chest rather than a concentrated lung lesion region.
- Shared difficult case `train_4692.png` was a FN for all three reviewed models; `convnext_tiny_finetune` still activated both lower lungs, while `densenet121_finetune` and `efficientnet_b3_finetune` showed weaker or more off-lung emphasis with visible border activation.
- Expanded FN-only comparison:
  - Additional FN review was performed in `outputs/gradcam_fn_review/` for `densenet121_finetune` and `convnext_tiny_finetune`, using 8 FN samples per model.
  - Common reviewed FN cases included `train_4692.png`, `train_4264.png`, `train_5033.png`, `train_3865.png`, and `train_3070.png`.
  - Across these shared FN cases, `convnext_tiny_finetune` more repeatedly activated bilateral or unilateral lower-lung fields, while `densenet121_finetune` more often shifted toward the lower image margin, broad background regions, or weak diffuse activation.
  - This means `convnext_tiny_finetune` still failed these cases at the decision level, but its FN attention pattern looked more lung-centered than `densenet121_finetune` in the expanded spot check.
- Possible artifact attention:
  - `densenet121_finetune` repeatedly activated image borders or the lower black background margin in reviewed FN and FP samples such as `train_4692.png`, `train_1802.png`, and `train_0637.png`.
  - `convnext_tiny_finetune` occasionally appeared sensitive to lateral chest wall or shoulder-adjacent regions in FP cases such as `train_0637.png` and `train_1339.png`.
  - `efficientnet_b3_finetune` more repeatedly showed off-lung or border/corner attention in reviewed samples, which increases shortcut-learning concern despite its strong precision and specificity metrics.
- Model-by-model spot-check summary:

| Experiment | Grad-CAM spot-check summary | Shortcut-learning concern |
| --- | --- | --- |
| `densenet121_frozen` | Correct PNEUMONIA sometimes overlapped one lung field, but reviewed FN and FP cases were dominated by broad border or lower-margin activation. | High |
| `densenet121_finetune` | More stable than frozen, but still frequently broad and lower-border-adjacent rather than sharply lesion-centered. | Moderate to high |
| `resnet50_frozen` | Some correct PNEUMONIA overlap on one lung, but FN and FP examples still leaned strongly toward upper-border and non-lung regions. | High |
| `resnet50_finetune` | Fine-tuning improved confidence, but reviewed correct and FN samples still showed top-border or diffuse activation more than focal opacity localization. | Moderate to high |
| `efficientnet_b0_frozen` | Reviewed samples mixed lower-lung overlap with obvious lower black-margin and lateral off-lung activation. | Moderate to high |
| `efficientnet_b0_finetune` | Correct PNEUMONIA example overlapped a unilateral lung region more clearly than the frozen run, but FN and FP cases still showed weak or off-lung emphasis. | Moderate |
| `convnext_tiny_frozen` | Even frozen, it often highlighted bilateral lower-lung regions rather than only borders, though the heatmaps stayed coarse and over-broad. | Moderate |
| `convnext_tiny_finetune` | Most consistent lower-lung emphasis among reviewed models, including the shared difficult FN, but some FP cases still drifted toward lateral chest wall or upper-lung regions. | Moderate |
| `efficientnet_b3_frozen` | Narrow focal hotspots appeared, but they frequently landed on upper-border, central non-lung, or isolated off-lung regions. | High |
| `efficientnet_b3_finetune` | Attention remained broad and unstable, with repeated upper-border, shoulder, or diffuse central-chest activation. | High |
- Caution:
  - This was a small qualitative spot check, not a full Grad-CAM audit across all transfer learning runs.
  - Grad-CAM is supportive evidence for failure mode review and does not prove clinical validity.

### Notes

- What worked:
  - The comparison table was rebuilt from refreshed `comparison_summary.csv` files rather than manually copied metrics.
  - Transfer learning runs produced the required validation metrics, confusion matrices, and FN/FP exports across ten completed experiments.
  - Fine-tuning improved DenseNet121, ResNet50, ConvNeXt-Tiny, EfficientNet-B0, and EfficientNet-B3 over their corresponding frozen runs on F1-score and FN count.
  - `densenet121_finetune` remains strongest by F1-score, PNEUMONIA recall, and FN count.
  - `convnext_tiny_finetune` is strongest by AUROC, specificity, precision, and FP count among the merged comparison runs.
  - Across the reviewed Grad-CAM subset, ConvNeXt-Tiny showed the most consistent lung-region emphasis in both frozen and finetuned form, while EfficientNet-B3 showed the strongest repeated off-lung activation risk.
  - The expanded FN-only review reinforces that `convnext_tiny_finetune` may rely on lung-region structure more consistently than `densenet121_finetune` and `efficientnet_b3_finetune`, even though `densenet121_finetune` achieved the lowest FN count quantitatively.
- What failed:
  - The full 10-model pass is still sample-based; only three models received the deeper FN/FP spot check, and the remaining seven were reviewed with one representative correct/FN/FP example each.
  - FN/FP image-level review is still qualitative and should not be treated as a complete failure-mode audit.
- Limitations:
  - These are internal validation metrics only.
  - Patient-level leakage cannot be fully excluded because patient identifiers are unavailable.
  - A high AUROC or F1-score does not establish clinical diagnostic validity.
  - The validation split is useful for model comparison, but external validation is still required before making any medical claim.
  - Model ranking may change after additional operating-point analysis, Grad-CAM artifact review, or external/test-like evaluation.
  - The qualitative Grad-CAM comparison here used a small random subset and should not be overgeneralized.
- Next experiment:
  - Continue the operating-point study in EXP-003 for `densenet121_finetune` and `convnext_tiny_finetune` threshold sweeps.
  - Decide whether the project should report a fixed-threshold winner and a threshold-tuned winner separately.
  - Validate any chosen threshold on a different split or OOF-style setup before treating it as a stable operating point.

## EXP-003-transfer-learning-threshold-sweep

Date: 2026-06-30\
Model: Threshold sweep for leading EXP-002 transfer-learning candidates\
Config: validation thresholds `0.10` to `0.90` with `0.005` step; fixed-threshold reference `0.5`; specificity floor `0.95`\
Dataset split: same held-out validation split used in EXP-002; `outputs/val_split.csv`\
Source predictions:
- `outputs/transfer_learning/densenet121_finetune/validation_predictions.csv`
- `outputs/transfer_learning_224_additional/convnext_tiny_finetune/validation_predictions.csv`
Output directory: `outputs/threshold_sweeps/exp002_transfer_tuning`
Sweep script: `scripts/sweep_transfer_thresholds.py`

### Purpose

- Separate model-family comparison from operating-point tuning.
- Compare how threshold choice changes the FN/FP tradeoff for the two leading EXP-002 candidates.
- Check whether `convnext_tiny_finetune` can recover FN count while retaining stronger specificity and cleaner Grad-CAM behavior.

### Setup

- Compared models:
  - `densenet121_finetune`
  - `convnext_tiny_finetune`
- Threshold grid:
  - minimum `0.10`
  - maximum `0.90`
  - step `0.005`
- Reported operating points:
  - fixed threshold `0.5`
  - best-F1 threshold
  - best-recall threshold with specificity `>= 0.95`
  - lowest-FP threshold subject to FN `<= 15`, `<= 20`, and `<= 30`

### Metrics

Threshold summary from `outputs/threshold_sweeps/exp002_transfer_tuning/threshold_summary.csv`.

| Experiment | Selection | Threshold | Accuracy | F1-score | Sensitivity / PNEUMONIA recall | Specificity / NORMAL recall | Precision | FN | FP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| densenet121_finetune | fixed_threshold | 0.50 | 0.9760306807 | 0.9838187702 | 0.9806451613 | 0.9626865672 | 0.9870129870 | 15 | 10 |
| densenet121_finetune | best_f1 | 0.43 | 0.9798657718 | 0.9864428664 | 0.9858064516 | 0.9626865672 | 0.9870801034 | 11 | 10 |
| densenet121_finetune | best_recall_spec>=0.95 | 0.41 | 0.9789069990 | 0.9858247423 | 0.9870967742 | 0.9552238806 | 0.9845559846 | 10 | 12 |
| densenet121_finetune | lowest_fp_fn<=15 | 0.505 | 0.9769894535 | 0.9844559585 | 0.9806451613 | 0.9664179104 | 0.9882964889 | 15 | 9 |
| densenet121_finetune | lowest_fp_fn<=20 | 0.575 | 0.9769894535 | 0.9843953186 | 0.9767741935 | 0.9776119403 | 0.9921363041 | 18 | 6 |
| convnext_tiny_finetune | fixed_threshold | 0.50 | 0.9683604986 | 0.9783322390 | 0.9612903226 | 0.9888059701 | 0.9959893048 | 30 | 3 |
| convnext_tiny_finetune | best_f1 | 0.125 | 0.9817833174 | 0.9877656149 | 0.9896774194 | 0.9589552239 | 0.9858611825 | 8 | 11 |
| convnext_tiny_finetune | best_recall_spec>=0.95 | 0.11 | 0.9817833174 | 0.9877656149 | 0.9896774194 | 0.9589552239 | 0.9858611825 | 8 | 11 |
| convnext_tiny_finetune | lowest_fp_fn<=15 | 0.245 | 0.9798657718 | 0.9863724854 | 0.9806451613 | 0.9776119403 | 0.9921671018 | 15 | 6 |
| convnext_tiny_finetune | lowest_fp_fn<=20 | 0.32 | 0.9760306807 | 0.9837133550 | 0.9741935484 | 0.9813432836 | 0.9934210526 | 20 | 5 |

### Notes

- What worked:
  - Threshold tuning materially changed the ranking between the two leading EXP-002 candidates on the same validation split.
  - `densenet121_finetune` remained strongest at the fixed `0.5` threshold on FN count.
  - `convnext_tiny_finetune` became the stronger threshold-tuned operating-point candidate, reaching FN `15` with FP `6` and specificity `0.9776` at threshold `0.245`.
  - Best-F1 threshold was also slightly stronger for `convnext_tiny_finetune` than for `densenet121_finetune`.
  - Accuracy also improved at the tuned operating points, with `convnext_tiny_finetune` reaching `0.9818` at its best-F1 threshold versus `0.9799` for `densenet121_finetune`.
- What failed or remains incomplete:
  - These thresholds were tuned on the same held-out validation split used for model comparison.
  - No external or OOF-style validation has confirmed that the tuned operating points are stable.
- Limitations:
  - This is an internal operating-point analysis, not an external validation result.
  - Tuned thresholds may overfit the held-out validation split.
  - The result should not be used as a clinical decision threshold.
- Next experiment:
  - Decide whether the project should report separate fixed-threshold and threshold-tuned winners.
  - Validate `convnext_tiny_finetune` threshold candidates such as `0.245` on a different split or OOF-style setup.
  - Revisit whether threshold tuning changes which transfer model should be highlighted in the portfolio summary.
  - Before changing to a new backbone again, run controlled follow-up experiments on the same leading candidate with only one training factor changed at a time.
  - Highest-priority follow-up factors:
    - preprocessing such as CLAHE, contrast normalization, or lighter crop strategy
    - loss adjustment such as focal loss or BCEWithLogitsLoss weighting changes
    - optimizer and learning-rate schedule changes such as AdamW vs a lower backbone learning rate or cosine schedule
    - regularization and augmentation strength tuning such as mixup off/on, dropout, or weight decay refinement

## EXP-004-densenet-efficientnet-b3-convnext-384-ensemble

Date: 2026-06-30\
Model: DenseNet121-CBAM + EfficientNet-B3 + ConvNeXt-Tiny ensemble\
Config: seed 42, image size 384, batch size 16, 5-fold StratifiedGroupKFold, 10 epochs, SWA checkpoint averaging, multi-scale TTA submission inference\
Dataset split: duplicate-aware split; train split from `outputs/train_split.csv`, validation split from `outputs/val_split.csv`\
Checkpoint directory: `outputs/Densenet+efficientnet_B3+convnext_384`\
Submission file: `outputs/Densenet+efficientnet_B3+convnext_384/submission_384.csv`\
Source script: `kaggle/train_ensemble_oof_384.py`
Standalone rerun presets:
- `configs/convnext_384_single.json`
- `configs/efficientnet_b3_384_single.json`

### Purpose

- Preserve the completed 384px three-model ensemble run before running single-model comparisons.
- Compare a higher-capacity ensemble against prior single transfer-learning baselines later.
- Use this only as a research/portfolio experiment, not as a clinical diagnostic system.

### Setup

- Backbones:
  - DenseNet121 with CBAM attention head
  - ConvNeXt-Tiny
  - EfficientNet-B3
- Image size: 384x384
- Batch size:
  - DenseNet121-CBAM: 16
  - ConvNeXt-Tiny: 16
  - EfficientNet-B3: 16
- Folds: 5
- Epochs: 10
- Optimizer: AdamW
- Learning rate:
  - Head learning rate: `1e-3`
  - Backbone learning rate: `1e-4`
  - Backbone warmup learning rate: `1e-5`
- Scheduler: CosineAnnealingLR after backbone unfreeze
- Loss function: SmoothFocalLoss with alpha `0.25`, gamma `2.0`, smoothing `0.05`
- Augmentation: CLAHE, resize to 1.1x, center crop, horizontal flip, rotation, brightness/contrast jitter
- Checkpoint strategy: average last 3 epoch checkpoints per fold as SWA-like checkpoint
- Inference: 5 folds per model, 3 models total, 4-way multi-scale TTA
- Threshold: optimized from OOF predictions inside the script, but the selected threshold was not persisted to a metrics artifact.

### Saved Artifacts

The output directory contains 15 fold checkpoints and one submission file:

- `best_densenet121_384_fold0.pt` to `best_densenet121_384_fold4.pt`
- `best_convnext_384_fold0.pt` to `best_convnext_384_fold4.pt`
- `best_efficientnet_384_fold0.pt` to `best_efficientnet_384_fold4.pt`
- `submission_384.csv`

Submission file verification:

| File | Rows | Columns | NORMAL predictions | PNEUMONIA predictions |
| --- | ---: | --- | ---: | ---: |
| `outputs/Densenet+efficientnet_B3+convnext_384/submission_384.csv` | 624 | `file_name,label` | 211 | 413 |

### Metrics

- Internal validation metrics were not saved as a structured artifact for this run.
- OOF threshold search was performed by `kaggle/train_ensemble_oof_384.py`, but the optimal threshold, ensemble weights, OOF predictions, confusion matrix, FN/FP tables, AUROC, sensitivity, and specificity were not persisted.
- Because the metric artifacts are missing, this run should not be compared against EXP-002 by validation performance yet.

### Notes

- What worked:
  - The run produced all 15 model fold checkpoints.
  - The run produced a valid submission CSV with 624 rows and the expected `file_name,label` columns.
  - The model family is appropriate for a later ensemble-vs-single comparison because EfficientNet-B3 and ConvNeXt-Tiny checkpoints are available separately.
- What failed or remains incomplete:
  - No saved `best_metrics.json`, OOF prediction CSV, confusion matrix, FN/FP CSV, or AUROC summary was found for this ensemble run.
  - The selected OOF threshold and model weights need to be regenerated or recovered before portfolio comparison.
  - Grad-CAM was not performed for this 384px ensemble.
- Limitations:
  - Submission label distribution alone is not a performance metric.
  - Without saved OOF metrics or leaderboard score, this run is only recorded as a completed training/inference artifact.
  - Patient-level leakage cannot be fully excluded because patient identifiers are unavailable.
  - This experiment must not be described as clinical validation.
- Next experiment:
  - Evaluate `EfficientNet-B3` and `ConvNeXt-Tiny` as standalone 384px models using the saved fold checkpoints.
  - Regenerate OOF metrics for the three-model ensemble and save confusion matrix, sensitivity, specificity, AUROC, FN, and FP.
  - Add single-model and ensemble results to one comparison table once all metrics are available.

## EXP-005-384-single-model-oof-evaluation

Date: 2026-06-30\
Model: ConvNeXt-Tiny 384 and EfficientNet-B3 384 standalone checkpoint evaluation\
Config: seed 42, image size 384, batch size 16, threshold 0.5, crop TTA only, CPU inference\
Dataset split: recreated 5-fold StratifiedGroupKFold over combined `outputs/train_split.csv` + `outputs/val_split.csv`; 5,216 total labeled images\
Checkpoint directory: `outputs/Densenet+efficientnet_B3+convnext_384`\
Output directories:
- `outputs/single_model_384_eval_convnext`
- `outputs/single_model_384_eval_efficientnet`
Source script: `scripts/evaluation/evaluate_ensemble_backbones_oof_384.py`

### Purpose

- Evaluate the saved 384px single-model checkpoints from the prior ensemble workflow as standalone models.
- Recover structured metrics that were missing from EXP-004.
- Compare the model family against EXP-002 cautiously, while explicitly noting that the evaluation protocol is different.

### Setup

- Backbones:
  - ConvNeXt-Tiny 384
  - EfficientNet-B3 384
- Evaluation data:
  - Combined labeled rows from `outputs/train_split.csv` and `outputs/val_split.csv`
  - Recreated 5-fold StratifiedGroupKFold using `duplicate_group_id`
- Image size: 384x384
- Batch size: 16
- Threshold: 0.5 fixed threshold for reported metrics
- Additional threshold search:
  - Best-F1 threshold also searched from `0.10` to `0.90` by `0.02` step in the evaluation script
- TTA: crop only
- Device: local CPU

### Important Comparison Note

- These results are 5-fold out-of-fold metrics over 5,216 images.
- EXP-002 used a single held-out validation split with 1,043 validation images.
- Because the sample size and validation protocol differ, these numbers should not be treated as a strictly apples-to-apples replacement for EXP-002.
- The comparison is still useful as a directional model-family reference.

### Metrics

Fixed-threshold metrics at threshold `0.5`.

| Experiment | Accuracy | Sensitivity / PNEUMONIA recall | Specificity / NORMAL recall | Precision | F1-score | AUROC | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| convnext_tiny_384_oof | 0.9925230061 | 0.9909677419 | 0.9970171514 | 0.9989594173 | 0.9949475321 | 0.9998833802 | 35 | 4 |
| efficientnet_b3_384_oof | 0.9906058282 | 0.9886451613 | 0.9962714392 | 0.9986965589 | 0.9936454416 | 0.9997861971 | 44 | 5 |

Best-F1 threshold search summary:

| Experiment | Best-F1 threshold | Best F1-score | FN at best F1 | FP at best F1 |
| --- | ---: | ---: | ---: | ---: |
| convnext_tiny_384_oof | 0.34 | 0.9976762200 | 11 | 7 |
| efficientnet_b3_384_oof | 0.36 | 0.9957446809 | 14 | 19 |

Directional comparison against EXP-002:

- Both 384px standalone models scored higher than the best EXP-002 transfer-learning run on OOF F1-score, AUROC, sensitivity, specificity, precision, and FP count.
- Among the two 384px standalone models, `convnext_tiny_384_oof` was stronger than `efficientnet_b3_384_oof` on all reported fixed-threshold metrics and had fewer FN and FP.
- This does not prove that the 384px models are universally better than EXP-002 models, because the validation setup is not identical.

### Confusion Matrix

Rows are actual labels and columns are predicted labels at threshold `0.5`.

ConvNeXt-Tiny 384:

|  | Predicted NORMAL | Predicted PNEUMONIA |
| --- | ---: | ---: |
| Actual NORMAL | 1337 | 4 |
| Actual PNEUMONIA | 35 | 3840 |

EfficientNet-B3 384:

|  | Predicted NORMAL | Predicted PNEUMONIA |
| --- | ---: | ---: |
| Actual NORMAL | 1336 | 5 |
| Actual PNEUMONIA | 44 | 3831 |

### FN/FP Review

- FN definition: PNEUMONIA predicted as NORMAL.
- FP definition: NORMAL predicted as PNEUMONIA.
- Lowest FN count between the two standalone 384px models at threshold `0.5`: `convnext_tiny_384_oof` with 35 FN.
- Lowest FP count between the two standalone 384px models at threshold `0.5`: `convnext_tiny_384_oof` with 4 FP.
- Saved artifacts per run:
  - `metrics_*.json`
  - `summary.csv`
  - `summary.json`
  - `oof_*.csv`
  - `confusion_matrix_*.csv`
  - `false_negatives_*.csv`
  - `false_positives_*.csv`
  - `fold_metrics_*.csv`

### Grad-CAM Review

- Correct samples reviewed: Not performed
- FN samples reviewed: Not performed
- FP samples reviewed: Not performed
- Observed focus regions: Not available
- Possible artifact attention: Not evaluated yet

### Notes

- What worked:
  - The saved EXP-004 checkpoints were successfully evaluated as standalone models.
  - Structured OOF metrics, confusion matrices, and FN/FP exports were recovered for both 384px backbones.
  - `ConvNeXt-Tiny 384` was the stronger of the two standalone runs at the fixed threshold.
- What failed or remains incomplete:
  - These results are not on the exact EXP-002 held-out validation split.
  - Grad-CAM and image-level FN/FP review are still missing.
- Limitations:
  - OOF evaluation across 5,216 images is not directly comparable to the 1,043-image single validation split from EXP-002.
  - Threshold tuning can materially change FN/FP tradeoffs, so the fixed-threshold and best-F1 views should be interpreted together.
  - These results remain research/portfolio metrics and do not establish clinical validity.
- Next experiment:
  - Decide whether to add a dedicated 384px single-model training run on the exact EXP-002 validation protocol for fairer comparison.
  - Run Grad-CAM on `convnext_tiny_384_oof` and compare against `efficientnet_b3_384_oof`.
  - Regenerate the full three-model ensemble OOF metrics so EXP-004 can join the same comparison table.
  - After the 384px comparison is stabilized, prioritize controlled training-setting experiments over another immediate backbone swap.

## EXP-006-convnext-training-setting-ablation-plan

Date: Planned after EXP-003 threshold validation\
Base model: `convnext_tiny_finetune`\
Base reference: EXP-002 fixed-threshold run and EXP-003 threshold sweep\
Target protocol: keep the same train/validation split and the same core evaluation rules as EXP-002 unless the experiment itself is explicitly about protocol change

### Purpose

- Improve the strongest transfer-learning candidate without immediately switching to another backbone.
- Test whether preprocessing, loss, optimizer or scheduler, and learning-rate choices can reduce FN while preserving the specificity and Grad-CAM stability already seen in `convnext_tiny_finetune`.
- Change one factor at a time so the effect of each adjustment stays interpretable.

### Base Lock

Keep these fixed unless the individual ablation explicitly changes them:

- Backbone: `convnext_tiny`
- Regime: finetune
- Image size: `224x224`
- Batch size: `32`
- Validation split: same EXP-002 split
- Core metrics: accuracy, sensitivity, specificity, precision, F1-score, AUROC, FN, FP
- Error review: keep FN/FP export and Grad-CAM spot check workflow

### Planned Order

1. Preprocessing ablation
2. Loss-function ablation
3. Optimizer or scheduler ablation
4. Learning-rate and regularization ablation

### EXP-006A Preprocessing Ablation

- Goal:
  - Test whether image normalization and contrast handling improve lesion visibility without increasing artifact focus.
- Candidate changes:
  - CLAHE on / off
  - contrast normalization or histogram stabilization
  - lighter crop strategy to reduce border sensitivity
- Keep fixed:
  - existing optimizer, loss, and learning rates from EXP-002
- Success signal:
  - lower FN or better sensitivity at similar specificity
  - Grad-CAM remains inside lung regions rather than shifting to borders or text markers

Current first-pass candidate:

- Experiment name:
  - `exp006a_convnext_no_color_jitter`
- Config:
  - `configs/exp006a_convnext_no_color_jitter.json`
- Single change from the EXP-002 ConvNeXt baseline:
  - disable training-time brightness and contrast jitter by setting:
    - `train_brightness: 0.0`
    - `train_contrast: 0.0`
- Rationale:
  - This is the smallest preprocessing or augmentation change already supported by the current pipeline and keeps the ablation easy to interpret.

Implementation status:

- Added config-driven preprocessing controls in `kaggle/train_transfer_model.py` for:
  - brightness and contrast jitter strength
  - resize-plus-center-crop scale
  - autocontrast and histogram equalization toggles
- Smoke-test status:
  - Passed on a reduced subset with `convnext_tiny_frozen` and `convnext_tiny_finetune`.
  - Smoke summary artifact:
    - `outputs/exp006a_convnext_no_color_jitter/comparison_summary.csv`
- Full training status:
  - Not started yet
  - Keep `num_workers=0` for local sandbox execution because multi-worker shared-memory startup failed in this environment

### EXP-006B Loss Ablation

- Goal:
  - Improve recall for difficult PNEUMONIA cases without collapsing specificity.
- Candidate changes:
  - BCEWithLogitsLoss `pos_weight` retuning
  - focal loss
  - smoothing or asymmetric loss only if needed after the simpler tests
- Keep fixed:
  - best preprocessing choice from EXP-006A
  - same optimizer and baseline learning rates
- Success signal:
  - reduced FN count
  - acceptable FP increase
  - no obvious Grad-CAM drift toward background artifacts

### EXP-006C Optimizer / Scheduler Ablation

- Goal:
  - Test whether optimization dynamics, not model family, are limiting performance.
- Candidate changes:
  - AdamW baseline vs cosine schedule refinement
  - lower backbone learning rate
  - backbone/head LR gap adjustment
- Keep fixed:
  - best preprocessing and loss settings selected earlier
- Success signal:
  - smoother validation behavior
  - improved best checkpoint F1 or sensitivity without unstable FP spikes

### EXP-006D Learning-Rate / Regularization Ablation

- Goal:
  - Refine generalization once the larger levers are tested.
- Candidate changes:
  - weight decay adjustment
  - dropout change
  - augmentation strength adjustment
  - optional mixup on/off only if it fits the current pipeline cleanly
- Keep fixed:
  - best earlier settings from EXP-006A to EXP-006C
- Success signal:
  - small but reproducible gains
  - no decline in Grad-CAM plausibility

### Evaluation Rule

- Do not combine multiple new changes in the same first-pass ablation.
- For each sub-experiment, record:
  - fixed threshold `0.5` metrics
  - threshold-tuned summary if relevant
  - FN/FP counts
  - short Grad-CAM note for representative correct, FN, and FP cases
- If one ablation clearly worsens FN and Grad-CAM focus together, stop that branch early.

### Decision Rule

- Prefer changes that improve PNEUMONIA recall or FN count without a disproportionate specificity collapse.
- If two settings are close numerically, prefer the one with more lung-centered Grad-CAM behavior and fewer repeated border or background activations.
- Keep clinical framing conservative: this is internal model refinement, not diagnostic validation.

## EXP-007-strict-split-revalidation-plan

Date: Executed 2026-06-30 for `convnext_tiny_frozen` and `convnext_tiny_finetune`\
Protocol change: replace `outputs/train_split.csv` and `outputs/val_split.csv` with `outputs/train_split_strict.csv` and `outputs/val_split_strict.csv`\
Reference report: `reports/strict_duplicate_split_report.md`

### Purpose

- Re-check leading transfer-learning settings on a stricter duplicate-aware validation protocol.
- Measure how much current EXP-002 and EXP-003 conclusions depend on the original split definition.
- Use strict split as a robustness check, not as a direct replacement for the original comparison history.

### Why This Matters

- The strict split keeps the same train and validation row counts, but changes grouping behavior for oversized perceptual clusters.
- In the strict report:
  - original groups: `3210`
  - strict groups: `4081`
  - largest validation strict group: `17`
  - `873` images were reassigned under `split_large` policy across the full labeled set
- This makes validation less dominated by very large connected components and should produce a tougher, more trustworthy internal comparison.

### Recommended Scope

- First strict-split rerun candidates:
  - `convnext_tiny_finetune`
  - `densenet121_finetune`
- Optional follow-up:
  - best EXP-006 ablation winner once one setting clearly stands out

### Execution Status

- Completed runs:
  - `convnext_tiny_frozen`
  - `convnext_tiny_finetune`
  - `densenet121_frozen`
  - `densenet121_finetune`
- Configs:
  - `configs/exp007_convnext_strict_revalidation.json`
  - `configs/exp007_densenet_strict_revalidation.json`
- Output roots:
  - `outputs/transfer_learning_strict_convnext`
  - `outputs/transfer_learning_strict_densenet`
- Local execution note:
  - The first full run failed under `num_workers=2` because this sandbox blocked PyTorch shared-memory worker startup.
  - Re-running with `num_workers=0` completed successfully.
- Validation safety note:
  - The duplicate-group validation helper was updated to prefer `strict_duplicate_group_id` when the strict split files provide it.

### Current Results

Best-checkpoint summary at threshold `0.5`:

| Setting | Accuracy | Sensitivity | Specificity | Precision | F1-score | AUROC | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original split `densenet121_finetune` | 0.9760 | 0.9806 | 0.9627 | 0.9870 | 0.9838 | 0.9962 | 15 | 10 |
| Original split `convnext_tiny_finetune` | 0.9684 | 0.9613 | 0.9888 | 0.9960 | 0.9783 | 0.9968 | 30 | 3 |
| Strict split `densenet121_frozen` | 0.9338 | 0.9406 | 0.9142 | 0.9694 | 0.9548 | 0.9838 | 46 | 23 |
| Strict split `convnext_tiny_frozen` | 0.9607 | 0.9600 | 0.9627 | 0.9867 | 0.9732 | 0.9938 | 31 | 10 |
| Strict split `densenet121_finetune` | 0.9616 | 0.9703 | 0.9366 | 0.9779 | 0.9741 | 0.9945 | 23 | 17 |
| Strict split `convnext_tiny_finetune` | 0.9751 | 0.9832 | 0.9515 | 0.9832 | 0.9832 | 0.9978 | 13 | 13 |

Source artifact:

- `outputs/transfer_learning_strict_convnext/comparison_summary.csv`
- `outputs/transfer_learning_strict_densenet/comparison_summary.csv`

### Interpretation

- The strict split did not cause a collapse for either leading fine-tuned transfer-learning candidate.
- `convnext_tiny_finetune` remained the stronger strict-split result overall:
  - higher accuracy
  - higher sensitivity / PNEUMONIA recall
  - higher specificity
  - higher F1-score
  - higher AUROC
  - lower FN
  - lower FP
- Compared with each model's original-split result:
  - `convnext_tiny_finetune` improved recall, F1-score, AUROC, and FN count, but lost specificity and added FP
  - `densenet121_finetune` worsened on accuracy, recall, specificity, F1-score, AUROC, FN, and FP
- This makes the strict split more supportive of ConvNeXt than the original split did, because the original EXP-002 fixed-threshold comparison had DenseNet ahead on FN count while ConvNeXt was stronger on specificity and Grad-CAM consistency.
- Because the validation protocol changed, these rows should stay in a separate robustness table rather than being merged into EXP-002 directly.

### Evaluation Rule

- Keep backbone and training recipe fixed when moving to strict split.
- Treat the split change itself as the only experimental change in the first rerun.
- Report the same metric set:
  - accuracy
  - sensitivity / PNEUMONIA recall
  - specificity / NORMAL recall
  - precision
  - F1-score
  - AUROC
  - FN
  - FP
- Keep FN/FP export and Grad-CAM spot-check workflow unchanged.

### Interpretation Rule

- Do not merge strict-split results into EXP-002 tables as if they were the same validation protocol.
- Present strict-split results as a robustness or stress-test companion to the original split results.
- If a model keeps its advantage on both the original split and strict split, confidence in that ranking increases.
- If rankings flip sharply, prioritize the stricter validation result in discussion and call out the split sensitivity explicitly.

### Next Experiment

- Start the full `EXP-006A` run with `configs/exp006a_convnext_no_color_jitter.json` and compare it first against the original-split ConvNeXt baseline, not against the strict-split result.
- If EXP-006A changes the operating point meaningfully, repeat the winning variant on the strict split before moving deeper into loss or optimizer ablations.
