# Report Index

Reports document the experiment sequence and should be interpreted as internal
research/education results, not clinical validation.

## Main reports

- [Final ensemble and pseudo-labeling report](final_ensemble_pseudo_labeling_report.md):
  final 384px ensemble, OOF metrics, threshold trade-offs, FN/FP analysis,
  pseudo-labeling, and Grad-CAM observations.
- [Mission 1 baseline report](mission1_baseline_report.md): dataset review and
  Custom CNN baseline.
- [Mission 1 data summary](mission1_data_summary.md): compact data and baseline
  evidence summary.

## Data and split validation

- [Pre-baseline audit](pre_baseline_audit.md)
- [Duplicate-aware split report](duplicate_aware_split_report.md)
- [Strict duplicate split report](strict_duplicate_split_report.md)
- [DataLoader forward smoke test](dataloader_forward_smoke_test.md)
- [Custom CNN training smoke test](custom_cnn_training_smoke_test.md)

## Supporting assets

- `figures/`: derived plots used by the reports.
- `tables/`: metadata and duplicate-audit CSVs.
- [`artifacts/`](artifacts/README.md): tracked configs, metrics, OOF evidence,
  and FN/FP tables selected from ignored experiment outputs.
- `submission/chest_xray_classification_report_2026-07-01.pdf`: authoritative
  25-page final submission report. It is excluded from Git because it embeds source X-rays;
  publish it only after confirming the dataset redistribution terms.

Raw X-ray sample sheets and local Grad-CAM outputs are excluded from Git until
the dataset redistribution terms are documented. Consequently, some historical
reports may reference local-only image artifacts under `outputs/`.

## Interpretation guardrails

- Compare results only when their split and evaluation protocols match.
- Review sensitivity, specificity, AUROC, confusion matrices, FN, and FP rather
  than accuracy alone.
- Patient-level leakage cannot be ruled out without patient identifiers.
- Grad-CAM is an interpretability aid, not lesion-localization validation.
- No result in these reports establishes clinical usefulness.
