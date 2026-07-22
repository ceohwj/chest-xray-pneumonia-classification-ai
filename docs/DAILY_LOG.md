# Daily Log

> Historical commands are preserved as executed. Some referenced utilities are
> no longer present; see [Historical Code References](HISTORICAL_CODE_REFERENCES.md).

Chronological project progress log. Keep entries short and factual.

## 2026-06-16

- Added repository documentation scaffolding.
- Established research/educational framing for the Chest X-ray `NORMAL` vs `PNEUMONIA` project.
- Defined evaluation emphasis beyond accuracy, including sensitivity, specificity, AUROC, confusion matrix, FN/FP analysis, and Grad-CAM.
- Confirmed no model code has been modified during documentation setup.

## 2026-06-24

- Work completed:
  - Implemented and ran the pre-baseline dataset audit pipeline before model training.
  - Added CSV/image mapping checks, image metadata extraction, class imbalance analysis, duplicate and perceptual-hash checks, shortcut-risk figures, and temporary baseline split generation.
  - Added baseline-ready PyTorch structure under `src/`, including dataset, split, baseline CNN scaffold, transfer-model factory, training placeholder, and evaluation metrics helpers.
  - Generated `reports/pre_baseline_audit.md`, audit tables, audit figures, `outputs/train_split.csv`, and `outputs/val_split.csv`.
- Verification:
  - Ran `python3 src/data/audit.py --seed 42`.
  - Ran `python3 -m compileall src`.
  - Ran import and metric smoke checks for the new modules.
  - Confirmed all 5,216 train images and 624 test images mapped successfully, with 0 missing and 0 unreadable files.
- Notes or blockers:
  - Label mapping was confirmed after the initial audit: `label 0 = NORMAL`, `label 1 = PNEUMONIA`.
  - Class imbalance exists: `label_0` = 1,341 images (25.7%), `label_1` = 3,875 images (74.3%).
  - Patient-level split cannot be verified from the current CSV columns or sequential file names.
  - Duplicate/near-duplicate checks found leakage risk across the temporary train/validation split and one train/test near-duplicate risk; validation metrics may be inflated if this is not handled.
  - `torchvision` is not installed, so transfer-learning scaffolding is present but not runnable yet.
- Next step:
  - Review `reports/tables/duplicate_image_report.csv` and create a duplicate-aware split before training the Custom CNN baseline.

## 2026-06-24 Duplicate-aware split update

- Work completed:
  - Created duplicate-aware train/validation split using the existing exact and near-duplicate report.
  - Built duplicate groups as graph connected components from `exact_sha256`, `near_identical_ahash`, and `near_identical_dhash` rows.
  - Regenerated `outputs/train_split.csv` and `outputs/val_split.csv` with `duplicate_group_id` and confirmed class names.
  - Generated `reports/duplicate_aware_split_report.md`.
- Verification:
  - Confirmed label mapping: `NORMAL=0`, `PNEUMONIA=1`.
  - Confirmed no `duplicate_group_id` appears in both train and validation.
  - Confirmed all original training images are assigned exactly once.
  - Confirmed model training has not started yet.
- Notes or blockers:
  - Patient-level identifiers are still unavailable, so patient-level leakage cannot be fully excluded.
  - One train/test near-duplicate risk remains reported as a limitation; the test set was not modified.
- Next step:
  - Review the duplicate-aware split report, then run a one-batch DataLoader and forward-pass smoke test before starting the Custom CNN baseline.

## 2026-06-24 Pre-training smoke test

- Work completed:
  - Added deterministic baseline transform utilities without requiring `torchvision`.
  - Ran one-batch train DataLoader and validation DataLoader smoke tests using the duplicate-aware split.
  - Ran one Custom CNN forward pass and one `CrossEntropyLoss` computation.
  - Generated `reports/dataloader_forward_smoke_test.md`.
- Verification:
  - Confirmed train batch shape: `[16, 3, 224, 224]`, labels `[16]`.
  - Confirmed validation batch shape: `[16, 3, 224, 224]`, labels `[16]`.
  - Confirmed Custom CNN output shape: `[16, 2]`.
  - Confirmed loss computation succeeded and was finite.
  - Confirmed model training has not started yet.
- Notes or blockers:
  - Validation first batch contained only NORMAL samples because validation loading is deterministic and unshuffled for this smoke test; this is not a performance evaluation.
- Next step:
  - Implement a minimal one-epoch Custom CNN training smoke test with metrics logging, checkpoint saving, and no model-selection claims.

## 2026-06-24 Custom CNN one-epoch smoke test

- Work completed:
  - Implemented and ran exactly one epoch of Custom CNN training as a pipeline smoke test.
  - Used duplicate-aware `outputs/train_split.csv` and `outputs/val_split.csv`.
  - Saved train log, validation metrics, confusion matrix, checkpoint, and smoke-test report.
- Verification:
  - Confirmed forward pass, loss computation, backward pass, optimizer step, validation loop, and checkpoint saving completed.
  - Confirmed checkpoint contains model state, optimizer state, epoch, config, and validation metrics.
  - Confirmed no clinical validity or model-selection claim was made.
- Notes or blockers:
  - This was not a performance experiment; one epoch is insufficient for conclusions.
  - Patient-level leakage cannot be fully excluded and the train/test near-duplicate limitation remains.
- Next step:
  - Review smoke-test outputs, then implement the formal Custom CNN baseline training script with reproducible config and careful metric reporting.

## 2026-06-24 Kaggle Custom CNN baseline

- Work completed:
  - Ran the formal Custom CNN baseline training code on Kaggle GPU (`Tesla T4`).
  - Used duplicate-aware train/validation splits with confirmed label mapping: `NORMAL=0`, `PNEUMONIA=1`.
  - Saved baseline artifacts under `/kaggle/working/custom_cnn_baseline`.
- Verification:
  - Training ran with CUDA and stopped early at epoch 23; best validation F1-score was at epoch 18.
  - Best internal validation metrics included F1-score `0.9846`, PNEUMONIA recall/sensitivity `0.9884`, NORMAL recall/specificity `0.9440`, AUROC `0.9960`, and PR-AUC `0.9986`.
  - Best confusion matrix was `[[253, 15], [9, 766]]`, so FN count was `9` and FP count was `15`.
- Notes or blockers:
  - This is a Custom CNN baseline result, not a final model or clinical diagnostic result.
  - Patient-level leakage cannot be fully excluded, and the train/test near-duplicate limitation remains.
  - Internal validation requires cautious interpretation and should be followed by Grad-CAM and FN/FP image review.
- Next step:
  - Review false negative and false positive samples, then compare with transfer learning baselines such as ResNet18, DenseNet121, and EfficientNet-B0.

## 2026-06-24 Custom CNN submission file

- Work completed:
  - Generated hackathon submission CSV from the Custom CNN baseline checkpoint.
  - Stored the file at `outputs/submissions/submission_custom_cnn_baseline.csv`.
- Verification:
  - Confirmed submission shape is 624 rows and 2 columns.
  - Confirmed columns are exactly `file_name,label`.
  - Confirmed labels contain only `0` and `1`.
  - Confirmed file order matches `sample_submission.csv`.
- Notes or blockers:
  - Submission label distribution is NORMAL `379`, PNEUMONIA `245`.
  - This submission is for hackathon scoring only and is not a clinical diagnostic output.
- Next step:
  - Submit `outputs/submissions/submission_custom_cnn_baseline.csv` to the hackathon platform and record the public/private score when available.

## 2026-06-24 Custom CNN leaderboard result

- Work completed:
  - Submitted the Custom CNN baseline submission file to the hackathon leaderboard.
  - Recorded public leaderboard score `0.73237`.
- Verification:
  - Compared public leaderboard score with internal validation metrics from the best checkpoint.
  - Confirmed a large gap between internal validation performance and leaderboard performance.
- Notes or blockers:
  - Internal validation metrics were high, but the leaderboard result suggests limited external/test-like generalization for this baseline.
  - Possible contributors include validation/test distribution shift, unresolved patient-level leakage risk, train/test near-duplicate limitation, shortcut learning, threshold/calibration behavior, and the limited feature extraction capacity of a Custom CNN.
  - This result should be framed as discovery of an internal-validation/generalization gap, not simply as a failed baseline.
  - No clinical diagnostic claim should be made from this result.
- Next step:
  - Review FN/FP samples and Grad-CAM attention patterns, then compare against transfer learning baselines with ResNet18, DenseNet121, and EfficientNet-B0.

## 2026-06-30 384px ensemble artifact log

- Work completed:
  - Recorded the completed `Densenet+efficientnet_B3+convnext_384` ensemble run in `docs/EXPERIMENT_LOG.md`.
  - Confirmed the output directory contains 15 fold checkpoints: 5 DenseNet121-CBAM, 5 ConvNeXt-Tiny, and 5 EfficientNet-B3 checkpoints.
  - Confirmed `submission_384.csv` exists.
- Verification:
  - Confirmed submission shape is 624 rows and 2 columns.
  - Confirmed columns are exactly `file_name,label`.
  - Confirmed submission label distribution is NORMAL `211`, PNEUMONIA `413`.
- Notes or blockers:
  - Structured internal metrics were not found for this run.
  - OOF threshold, ensemble weights, confusion matrix, AUROC, sensitivity, specificity, FN, and FP need to be regenerated before comparing this ensemble with single models.
- Next step:
  - Evaluate EfficientNet-B3 and ConvNeXt-Tiny standalone 384px checkpoints, then regenerate ensemble OOF metrics for a fair comparison.

## 2026-06-30 Transfer learning comparison refresh

- Work completed:
  - Added terminal-ready transfer learning experiment support for `convnext_tiny_frozen`, `convnext_tiny_finetune`, `efficientnet_b3_frozen`, and `efficientnet_b3_finetune`.
  - Aligned the new ConvNeXt-Tiny and EfficientNet-B3 comparison runs to the same core EXP-002 setting of `224x224` image size and batch size `32`.
  - Added `configs/transfer_learning_convnext_efficientnet_b3.json` for the additional 4-run comparison set.
  - Rebuilt `outputs/transfer_learning/comparison_summary.csv` from per-run `run_summary.json` files and merged it with `outputs/transfer_learning_224_additional/comparison_summary.csv`.
  - Added `scripts/merge_transfer_learning_comparison.py` and updated `docs/EXPERIMENT_LOG.md` so EXP-002 now reflects the merged comparison table source.
- Verification:
  - Ran `python3 -m compileall kaggle/train_transfer_model.py`.
  - Ran `python3 -m compileall scripts/merge_transfer_learning_comparison.py`.
  - Confirmed smoke tests used `images=(32, 3, 224, 224)` for the additional 224px transfer-learning path.
  - Ran `.venv/bin/python scripts/merge_transfer_learning_comparison.py --roots outputs/transfer_learning outputs/transfer_learning_224_additional --merged-output-root outputs/transfer_learning_merged`.
  - Confirmed merged artifacts were created: `comparison_summary.csv`, `comparison_summary.json`, and `comparison_summary.md`.
- Notes or blockers:
  - The original `outputs/transfer_learning/comparison_summary.csv` was incomplete, so it had to be regenerated from `run_summary.json` files before merging.
  - Pretrained ConvNeXt-Tiny and EfficientNet-B3 weight downloads required local SSL certificate handling and cached weights in `outputs/torch_cache`.
  - Grad-CAM and image-level FN/FP review are still missing for the expanded transfer-learning comparison set.
- Next step:
  - Use `outputs/transfer_learning_merged/comparison_summary.csv` as the source of truth for future EXP-002 table refreshes.
  - Run Grad-CAM and FN/FP case review for `densenet121_finetune` and `convnext_tiny_finetune`.

## 2026-06-30 Transfer learning Grad-CAM spot check

- Work completed:
  - Extended `scripts/generate_transfer_gradcam.py` so it can sample random class-balanced correct cases with separate `NORMAL` and `PNEUMONIA` counts.
  - Generated Grad-CAM outputs for `convnext_tiny_finetune` and `efficientnet_b3_finetune`.
  - Generated the same class-balanced Grad-CAM outputs for `densenet121_finetune`.
  - Reviewed 5 random correct `NORMAL` samples, 5 random correct `PNEUMONIA` samples, 3 FN samples, and 3 FP samples for each reviewed model.
  - Updated `docs/EXPERIMENT_LOG.md` EXP-002 with a qualitative Grad-CAM comparison note.
- Verification:
  - Ran `python3 -m compileall scripts/generate_transfer_gradcam.py`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning_224_additional/convnext_tiny_finetune --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning_224_additional/efficientnet_b3_finetune --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning/densenet121_finetune --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Confirmed `summary.json` and `metadata.csv` were created under `outputs/gradcam/densenet121_finetune`, `outputs/gradcam/convnext_tiny_finetune`, and `outputs/gradcam/efficientnet_b3_finetune`.
- Notes or blockers:
  - This was a small qualitative sample review, not a full Grad-CAM audit of all transfer learning runs.
  - In the reviewed subset, `convnext_tiny_finetune` appeared to attend to lung regions more consistently, while `densenet121_finetune` and `efficientnet_b3_finetune` more often showed diffuse lower-border or off-lung activation.
  - Shared difficult FN case `train_4692.png` remained concerning because all three reviewed models failed it, and `densenet121_finetune` plus `efficientnet_b3_finetune` showed especially weak lung-focused activation there.
- Next step:
  - Extend the same class-balanced Grad-CAM review to the remaining transfer learning models.
  - Compare whether `densenet121_finetune`'s lower FN count remains preferable after broader artifact-focused review against `convnext_tiny_finetune`.

## 2026-06-30 Transfer learning Grad-CAM full generation

- Work completed:
  - Generated the same class-balanced Grad-CAM sample sets for the remaining EXP-002 runs: `densenet121_frozen`, `resnet50_frozen`, `resnet50_finetune`, `efficientnet_b0_frozen`, `efficientnet_b0_finetune`, `convnext_tiny_frozen`, and `efficientnet_b3_frozen`.
  - Standardized every EXP-002 Grad-CAM output to 5 correct `NORMAL`, 5 correct `PNEUMONIA`, 3 FN, and 3 FP samples per run.
  - Updated `docs/EXPERIMENT_LOG.md` so EXP-002 now distinguishes full artifact generation from partial qualitative review.
- Verification:
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning/densenet121_frozen --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning/resnet50_frozen --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning/resnet50_finetune --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning/efficientnet_b0_frozen --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning/efficientnet_b0_finetune --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning_224_additional/convnext_tiny_frozen --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning_224_additional/efficientnet_b3_frozen --device cpu --num-correct-normal 5 --num-correct-pneumonia 5 --num-fn 3 --num-fp 3 --correct-selection random --seed 42`.
  - Confirmed all 10 EXP-002 runs now have `summary.json` under `outputs/gradcam/<experiment_name>/summary.json`.
- Notes or blockers:
  - Full artifact generation is complete, but qualitative Grad-CAM interpretation is still limited to `densenet121_finetune`, `convnext_tiny_finetune`, and `efficientnet_b3_finetune`.
  - Reviewing all 10 runs carefully will take additional qualitative pass time because Grad-CAM should be interpreted conservatively and not overclaimed from a few images.
- Next step:
  - Review the remaining frozen and non-top-performing finetune runs for repeated artifact patterns.
  - Use the full Grad-CAM set to support later threshold-tuning and model-selection discussion for the transfer-learning experiments.

## 2026-06-30 Transfer learning Grad-CAM comparison pass

- Work completed:
  - Reviewed one representative `correct_normal`, `correct_pneumonia`, `fn`, and `fp` Grad-CAM image for each of the remaining seven EXP-002 runs.
  - Added a 10-model Grad-CAM spot-check summary table to `docs/EXPERIMENT_LOG.md`.
  - Extended the EXP-002 note so it now distinguishes:
    - detailed spot check for `densenet121_finetune`, `convnext_tiny_finetune`, and `efficientnet_b3_finetune`
    - representative model-by-model pass for the other seven runs
- Verification:
  - Confirmed representative images existed for each remaining run via `outputs/gradcam/<experiment_name>/metadata.csv`.
  - Visually reviewed representative Grad-CAM cards for:
    - `densenet121_frozen`
    - `resnet50_frozen`
    - `resnet50_finetune`
    - `efficientnet_b0_frozen`
    - `efficientnet_b0_finetune`
    - `convnext_tiny_frozen`
    - `efficientnet_b3_frozen`
- Notes or blockers:
  - In this representative pass, ConvNeXt-Tiny remained the most consistently lung-focused family, even in the frozen regime.
  - EfficientNet-B3 showed the most repeated off-lung or upper-border activation across frozen and finetuned checks.
  - The full 10-model comparison is still a spot check and should support, not replace, metric-based selection.
- Next step:
  - Review additional FN cards from `densenet121_finetune` and `convnext_tiny_finetune`.
  - Decide whether EXP-002 should prioritize `densenet121_finetune` for lower FN count or give more weight to `convnext_tiny_finetune` for Grad-CAM consistency plus stronger specificity/FP behavior.

## 2026-06-30 Transfer learning FN deep-dive

- Work completed:
  - Generated expanded FN-only Grad-CAM review sets under `outputs/gradcam_fn_review/` for `densenet121_finetune` and `convnext_tiny_finetune`.
  - Reviewed 8 FN samples per model to compare repeated failure patterns more directly.
  - Updated EXP-002 so the Grad-CAM section now includes an explicit FN-only comparison note.
- Verification:
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning/densenet121_finetune --output-root outputs/gradcam_fn_review --device cpu --num-correct 0 --num-fn 8 --num-fp 0 --correct-selection random --seed 42`.
  - Ran `.venv/bin/python scripts/generate_transfer_gradcam.py --experiment-dir outputs/transfer_learning_224_additional/convnext_tiny_finetune --output-root outputs/gradcam_fn_review --device cpu --num-correct 0 --num-fn 8 --num-fp 0 --correct-selection random --seed 42`.
  - Confirmed `metadata.csv` files were created for both expanded FN review folders.
- Notes or blockers:
  - Shared FN cases such as `train_4692.png`, `train_4264.png`, `train_5033.png`, `train_3865.png`, and `train_3070.png` showed a consistent split:
    - `convnext_tiny_finetune` usually remained inside lower-lung regions
    - `densenet121_finetune` more often drifted toward lower border or diffuse background activation
  - This does not erase `densenet121_finetune`'s lower FN count, but it does make the final EXP-002 choice less one-sided than metrics alone suggest.
- Next step:
  - Decide whether the project should prefer lower FN count directly or give additional weight to attention consistency and artifact risk.
  - If needed, run threshold tuning on both models and see whether `convnext_tiny_finetune` can recover FN count without losing its specificity advantage.

## 2026-06-30 Transfer learning threshold sweep

- Work completed:
  - Added `scripts/sweep_transfer_thresholds.py` to run reproducible validation-threshold sweeps from saved `validation_predictions.csv`.
  - Ran the sweep for `densenet121_finetune` and `convnext_tiny_finetune`.
  - Split threshold tuning into its own experiment entry and recorded the operating-point comparison as EXP-003.
- Verification:
  - Ran `python3 -m compileall scripts/sweep_transfer_thresholds.py`.
  - Ran `.venv/bin/python scripts/sweep_transfer_thresholds.py --experiment-dirs outputs/transfer_learning/densenet121_finetune outputs/transfer_learning_224_additional/convnext_tiny_finetune --output-root outputs/threshold_sweeps/exp002_transfer_tuning --threshold-min 0.10 --threshold-max 0.90 --threshold-step 0.005 --fixed-threshold 0.5 --min-specificity 0.95 --target-fn 15 20 30`.
  - Confirmed threshold artifacts were created:
    - `outputs/threshold_sweeps/exp002_transfer_tuning/threshold_summary.csv`
    - `outputs/threshold_sweeps/exp002_transfer_tuning/threshold_summary.md`
- Notes or blockers:
  - `convnext_tiny_finetune` improved dramatically once threshold was tuned, reaching FN `15` with FP `6` at threshold `0.245`, compared with FN `30` and FP `3` at the fixed threshold `0.5`.
  - On the same validation split, `convnext_tiny_finetune` best-F1 threshold `0.125` outperformed `densenet121_finetune` best-F1 threshold `0.43` on F1-score while also reducing FN more aggressively.
  - These tuned thresholds were selected on the validation split itself, so they are useful for internal operating-point analysis but may not generalize without further validation.
- Next step:
  - Decide whether the transfer-learning results should present separate fixed-threshold and threshold-tuned winners.
  - If the project wants to prefer `convnext_tiny_finetune`, validate its tuned threshold on a different split or OOF-style setup first.
  - After threshold validation, move to controlled training-setting experiments rather than another immediate backbone replacement.
  - Priority order: preprocessing, loss adjustment, optimizer or scheduler tuning, then learning-rate or regularization refinement.

## 2026-06-30 EXP-006 planning

- Work completed:
  - Added a planned `EXP-006-convnext-training-setting-ablation-plan` entry to `docs/EXPERIMENT_LOG.md`.
  - Fixed the next refinement stage around `convnext_tiny_finetune` instead of introducing another new backbone immediately.
  - Broke the next stage into one-factor-at-a-time ablations: preprocessing, loss, optimizer or scheduler, and learning-rate or regularization refinement.
- Verification:
  - Confirmed the new EXP-006 section was appended after the current completed experiment entries.
- Notes or blockers:
  - This is a planning entry only; no new training run has started yet.
  - The plan intentionally keeps the same EXP-002 split and metric set so later comparisons stay interpretable.
- Next step:
  - Start with `EXP-006A` preprocessing ablation before changing loss or optimizer settings.

## 2026-06-30 Strict split revalidation planning

- Work completed:
  - Reviewed `outputs/train_split_strict.csv` and `outputs/val_split_strict.csv` as a candidate follow-up validation protocol.
  - Added `EXP-007-strict-split-revalidation-plan` to `docs/EXPERIMENT_LOG.md`.
  - Positioned strict split as a robustness check for the current transfer-learning winner rather than a direct replacement for EXP-002.
- Verification:
  - Confirmed strict split keeps the same row counts as the current split:
    - train `4173`
    - validation `1043`
  - Confirmed additional duplicate-aware audit fields exist in the strict CSV files:
    - `duplicate_group_size`
    - `strict_group_policy`
    - `strict_duplicate_group_id`
    - `strict_audit_flag`
  - Reviewed `reports/strict_duplicate_split_report.md` and confirmed:
    - original groups `3210`
    - strict groups `4081`
    - largest validation strict group `17`
    - train/validation strict-group overlap `0`
- Notes or blockers:
  - Because the validation protocol changes, strict-split results should not be merged into the EXP-002 model-comparison table as if they were the same experiment.
  - The best first rerun candidates remain `convnext_tiny_finetune` and `densenet121_finetune`.
- Next step:
  - Re-run the current leading transfer-learning setting on `train_split_strict.csv` and `val_split_strict.csv`.
  - If ranking remains stable, use that to strengthen confidence before broader EXP-006 ablations continue.

## 2026-06-30 EXP-006A preprocessing setup

- Work completed:
  - Added config-driven preprocessing controls to `kaggle/train_transfer_model.py`.
  - Opened the following tunable fields for transfer-learning runs:
    - brightness and contrast jitter strength
    - resize-plus-center-crop scale
    - autocontrast toggle
    - histogram equalization toggle
  - Added the first concrete preprocessing-ablation config:
    - `configs/exp006a_convnext_no_color_jitter.json`
  - Fixed the first-pass EXP-006A choice to a single interpretable change:
    - disable training brightness and contrast jitter for `convnext_tiny`
- Verification:
  - Ran `python3 -m compileall kaggle/train_transfer_model.py`.
  - Ran `.venv/bin/python kaggle/train_transfer_model.py --config configs/exp006a_convnext_no_color_jitter.json --smoke-test --skip-test-inference`.
  - Confirmed smoke artifacts were created under `outputs/exp006a_convnext_no_color_jitter/`.
- Notes or blockers:
  - Full EXP-006A training has not started yet.
  - Local full runs should use `num_workers=0` in this sandbox because PyTorch shared-memory worker startup is blocked.
- Next step:
  - Run the full `exp006a_convnext_no_color_jitter` experiment on the original EXP-002 split and compare FN or FP tradeoffs against the existing ConvNeXt baseline.

## 2026-06-30 Strict split ConvNeXt revalidation

- Work completed:
  - Ran the planned strict-split revalidation for:
    - `convnext_tiny_frozen`
    - `convnext_tiny_finetune`
  - Added a results summary to `docs/EXPERIMENT_LOG.md` under `EXP-007-strict-split-revalidation-plan`.
  - Updated duplicate-group validation so strict-split runs prefer `strict_duplicate_group_id` over the older `duplicate_group_id` field when both exist.
- Verification:
  - Ran `.venv/bin/python kaggle/train_transfer_model.py --config configs/exp007_convnext_strict_revalidation.json --smoke-test --skip-test-inference`.
  - Ran `.venv/bin/python kaggle/train_transfer_model.py --config configs/exp007_convnext_strict_revalidation.json --num-workers 0`.
  - Confirmed strict comparison artifact:
    - `outputs/transfer_learning_strict_convnext/comparison_summary.csv`
  - Confirmed best strict finetune metrics:
    - accuracy `0.9751`
    - sensitivity `0.9832`
    - specificity `0.9515`
    - precision `0.9832`
    - F1-score `0.9832`
    - AUROC `0.9978`
    - FN `13`
    - FP `13`
- Notes or blockers:
  - The first full attempt with `num_workers=2` failed because the sandbox blocked `torch_shm_manager`.
  - Re-running with `num_workers=0` completed successfully.
  - The strict-split ConvNeXt result improved recall and FN count compared with the original split baseline, but it also increased FP and reduced specificity.
- Next step:
  - Re-run `densenet121_finetune` on the same strict split so the robustness comparison covers more than one leading backbone.
  - Start the full EXP-006A original-split preprocessing ablation.

## 2026-06-30 Strict split DenseNet revalidation

- Work completed:
  - Ran the planned strict-split revalidation for:
    - `densenet121_frozen`
    - `densenet121_finetune`
  - Added the DenseNet strict-split results into the EXP-007 robustness-comparison section in `docs/EXPERIMENT_LOG.md`.
  - Completed the first side-by-side strict comparison between the two leading transfer-learning families:
    - `densenet121_finetune`
    - `convnext_tiny_finetune`
- Verification:
  - Added `configs/exp007_densenet_strict_revalidation.json`.
  - Ran `.venv/bin/python kaggle/train_transfer_model.py --config configs/exp007_densenet_strict_revalidation.json --smoke-test --skip-test-inference`.
  - Ran `.venv/bin/python kaggle/train_transfer_model.py --config configs/exp007_densenet_strict_revalidation.json --num-workers 0`.
  - Confirmed strict DenseNet artifact:
    - `outputs/transfer_learning_strict_densenet/comparison_summary.csv`
  - Confirmed best strict finetune metrics:
    - accuracy `0.9616`
    - sensitivity `0.9703`
    - specificity `0.9366`
    - precision `0.9779`
    - F1-score `0.9741`
    - AUROC `0.9945`
    - FN `23`
    - FP `17`
- Notes or blockers:
  - Under the strict split, `convnext_tiny_finetune` stayed ahead of `densenet121_finetune` on every tracked threshold-0.5 metric and also had lower FN and FP.
  - This is a stronger robustness result for ConvNeXt than the original EXP-002 fixed-threshold comparison, where DenseNet had the lower FN count.
  - Full runs still require `num_workers=0` in this sandbox because multi-worker shared-memory startup is blocked.
- Next step:
  - Start the full EXP-006A original-split preprocessing ablation with `configs/exp006a_convnext_no_color_jitter.json`.
  - If EXP-006A produces a promising ConvNeXt variant, rerun that winner on the strict split before moving on to loss ablations.

## Entry Template

```markdown
## YYYY-MM-DD

- Work completed:
- Verification:
- Notes or blockers:
- Next step:
```
