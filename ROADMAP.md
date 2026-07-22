# Roadmap

This roadmap separates completed modeling work from the remaining publication
and reproducibility work. The project remains research/education-only.

## Completed experiment milestones

- [x] Dataset audit and label verification.
- [x] Reproducible Dataset, DataLoader, transforms, and split utilities.
- [x] Custom CNN baseline and smoke training workflow.
- [x] Accuracy, precision, recall, F1, AUROC, sensitivity, specificity,
  confusion matrix, FN, and FP reporting.
- [x] Transfer-learning comparison across frozen and fine-tuned models.
- [x] Duplicate-aware and strict duplicate-group split review.
- [x] 384px multi-model ensemble and 5-fold OOF evaluation.
- [x] Threshold sensitivity/specificity and FN/FP trade-off analysis.
- [x] Grad-CAM generation for correct, FN, and FP samples.
- [x] Pseudo-labeling experiment with limitations documented.

## Publication preparation

- [x] Align README and project status with the actual implementation.
- [x] Exclude local images, checkpoints, generated outputs, and credentials.
- [x] Add dependency, script, and report indexes.
- [x] Group runnable code by responsibility and apply action-first file names.
- [x] Promote small reproducibility artifacts from ignored experiment outputs.
- [ ] Add exact dataset URL, citation, version, and redistribution terms.
- [ ] Choose and add a repository license.
- [ ] Confirm that every committed figure is permitted for redistribution.
- [ ] Decide whether to publish model weights as a separate release artifact.

## Reproducibility hardening

- [ ] Recover or recreate missing experiment config files referenced by logs.
- [ ] Replace hard-coded Kaggle/Colab paths with argparse or config values.
- [ ] Consolidate the canonical training and evaluation entry points.
- [ ] Add a synthetic-data smoke test that does not require the private dataset.
- [ ] Add unit tests for metric edge cases and FN/FP mapping.
- [ ] Add CI for syntax, imports, and tests.
- [ ] Record a tested Python and dependency lock file for the reference run.
- [ ] Document checkpoint provenance, hashes, and expected preprocessing.

## Modeling follow-up

- [ ] Run independent external validation if an appropriately licensed dataset
  is available.
- [ ] Investigate patient-level grouping if identifiers can be obtained lawfully.
- [ ] Quantify calibration and consider reliability diagrams or Brier score.
- [ ] Review more Grad-CAM cases, especially border/marker-focused errors.
- [ ] Test artifact-mitigation changes one at a time using the same protocol.

## Reporting follow-up

- [ ] Keep single-split and OOF results clearly separated.
- [ ] Add a compact model card with intended use, out-of-scope use, data,
  metrics, ethical considerations, and limitations.
- [ ] Preserve the statement that performance is not evidence of clinical
  diagnostic validity.
