# Notebook Guide

| Notebook | Purpose | Environment |
| --- | --- | --- |
| [`train_final_submission_ensemble.ipynb`](train_final_submission_ensemble.ipynb) | Authoritative final-submission training, TTA inference, and pseudo-label fine-tuning workflow | Kaggle |

This is the notebook used for the final submission. It is an all-in-one artifact and contains no
saved cell outputs, but it retains Kaggle dataset mount defaults. For reusable
development, prefer the role-based Python modules under `scripts/`; use this
notebook when a single Kaggle execution document is required.

Before running it:

1. Update the Kaggle dataset and sample-submission paths.
2. Confirm the expected train, validation, test, and checkpoint inputs.
3. Use a fresh `/kaggle/working` output directory.
4. Record the seed, threshold, model weights, and preprocessing settings.
