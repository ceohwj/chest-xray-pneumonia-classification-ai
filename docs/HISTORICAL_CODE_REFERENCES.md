# Historical Code References

The experiment logs preserve commands from earlier local iterations. Three
utility scripts referenced by those logs are not present in the current
worktree:

| Historical path | Recorded purpose | Current status |
| --- | --- | --- |
| `scripts/merge_transfer_learning_comparison.py` | Merge transfer-model result tables | Not retained |
| `scripts/generate_transfer_gradcam.py` | Generate Grad-CAM for 224px transfer runs | Not retained |
| `scripts/sweep_transfer_thresholds.py` | Sweep thresholds from saved validation probabilities | Not retained |

These names appear only as historical evidence in `DAILY_LOG.md` and
`EXPERIMENT_LOG.md`; they are not advertised as runnable entry points. If the
utilities are recovered or reimplemented, place them under the role-based
locations below:

```text
scripts/reporting/merge_transfer_model_results.py
scripts/explainability/generate_transfer_model_gradcam.py
scripts/evaluation/sweep_classification_thresholds.py
```

Do not infer the missing implementations from metric tables alone. Any
replacement should be verified against the saved CSV schemas and documented as
a reconstruction rather than the original experiment code.
