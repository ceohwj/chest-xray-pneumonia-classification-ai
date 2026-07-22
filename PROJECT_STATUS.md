# Project Status

Last updated: 2026-07-22

## Summary

The project has progressed from a Custom CNN baseline to transfer-learning,
grouped cross-validation, 384px ensemble, pseudo-labeling, threshold analysis,
and Grad-CAM experiments. The current phase is repository publication cleanup
and reproducibility hardening.

This is a research and education portfolio project, not a clinically validated
diagnostic system.

## Completed

- Audited the image metadata, class balance, dimensions, and possible artifacts.
- Implemented reusable Dataset, transform, split, model, metric, and seed helpers.
- Created duplicate-aware and stricter duplicate-group train/validation splits.
- Verified zero file overlap and zero strict duplicate-group overlap in the
  documented strict split.
- Trained and evaluated a Custom CNN baseline.
- Compared frozen and fine-tuned transfer-learning models.
- Evaluated a 384px DenseNet121-CBAM, EfficientNet-B3, and ConvNeXt-Tiny
  ensemble with 5-fold OOF predictions.
- Reported sensitivity, specificity, precision, recall, F1-score, AUROC,
  confusion matrices, FN, and FP.
- Analyzed threshold-dependent FN/FP trade-offs.
- Generated and reviewed Grad-CAM examples for correct, FN, and FP cases.
- Documented pseudo-labeling experiments and their limitations.

## Current reference result

The documented 384px ensemble at OOF threshold `0.40` reported:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9919 |
| Sensitivity / PNEUMONIA recall | 0.9935 |
| Specificity / NORMAL recall | 0.9873 |
| Precision | 0.9956 |
| F1-score | 0.9946 |
| AUROC | 0.9995 |
| FN | 25 |
| FP | 17 |

These are 5-fold OOF results over 5,216 labeled images. They are internal
experimental results and do not demonstrate external or clinical validity.

## Publication readiness

Completed in the current cleanup:

- Root README aligned with the implemented code and documented results.
- Dependencies listed in `requirements.txt`.
- Large data, checkpoints, generated outputs, credentials, and raw X-ray sample
  sheets excluded from Git.
- Script, Kaggle workflow, report, and historical-reference indexes added.
- Runnable code reorganized by training, evaluation, inference, explainability,
  reporting, verification, and smoke-test responsibility.
- The integrated final notebook is documented under `notebooks/`; the final PDF
  report is filed under `reports/submission/` and kept local pending image
  redistribution review.
- These notebook and PDF files are the artifacts used for the actual final
  submission, not reconstructed portfolio-only versions.

Still required:

- Add the exact dataset source, version, citation, and license terms.
- Choose and add a code license.
- Decide whether selected model weights should be published separately as a
  GitHub Release or other artifact, with provenance and checksums.
- Parameterize the remaining hard-coded Kaggle/Colab paths.
- Restore or regenerate referenced experiment configs that are not currently in
  the repository.
- Add automated tests or CI for imports, metrics, and small synthetic-data smoke
  checks.

## Known risks and limitations

- Patient-level identifiers are unavailable, so patient-level leakage cannot be
  fully excluded.
- Some perceptual duplicate groups contain mixed labels, indicating ambiguity or
  limitations in the duplicate heuristic.
- The labeled validation/OOF data comes from the same overall dataset family;
  independent external validation has not been performed.
- Grad-CAM review found attention on borders, markers, and other possible
  shortcut features in some samples.
- Pseudo-label confidence changes cannot establish better test performance when
  test labels are unavailable.
- Runnable workflows are grouped by role under `scripts/`; Kaggle entry points
  are kept separately under `kaggle/`.
