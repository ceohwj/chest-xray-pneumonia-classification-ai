# Reproducible Data Splits

These CSV files preserve the exact train/validation assignments used by the
documented experiments. Image files are not included.

| File | Rows | Purpose |
| --- | ---: | --- |
| `train_duplicate_aware.csv` | 4,173 | Original duplicate-aware training split |
| `val_duplicate_aware.csv` | 1,043 | Original duplicate-aware validation split |
| `train_strict_duplicate_aware.csv` | 4,173 | Strict duplicate-group training split |
| `val_strict_duplicate_aware.csv` | 1,043 | Strict duplicate-group validation split |

Label mapping:

- `0`: NORMAL
- `1`: PNEUMONIA

The strict split records both `duplicate_group_id` and
`strict_duplicate_group_id`. The documented verification found zero file overlap
and zero strict duplicate-group overlap between train and validation. Patient
identifiers are unavailable, so patient-level leakage cannot be ruled out.

The original generated copies remain under ignored `outputs/`. New experiments
should read these tracked files from `data/splits/` so a clone uses the same
assignments.
