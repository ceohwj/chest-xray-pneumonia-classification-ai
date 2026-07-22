# Strict Duplicate Split Report

This split is for research and education only. It is not evidence of clinical diagnostic validity.

## Goal

- Preserve leakage prevention for exact duplicates.
- Reduce validation/OOF calibration distortion from oversized aHash/dHash connected components.
- Keep the original `duplicate_group_id` column unchanged and add `strict_duplicate_group_id`.

## Strict group policy

- `group_size <= 20`: keep original group.
- `21 <= group_size <= 50`: audit flag, default keep.
- `group_size > 50`: split to file-level groups unless exact duplicate evidence links files.
- Split audit groups option enabled: False

### Policy image counts

| policy | image_count |
| --- | --- |
| keep | 4264 |
| split_large | 873 |
| audit_keep | 79 |

### Policy strict group counts

| policy | strict_group_count |
| --- | --- |
| keep | 3206 |
| split_large | 873 |
| audit_keep | 2 |

## Split comparison

| split | label_0 | label_1 | rows |
| --- | --- | --- | --- |
| old_train | 1073 | 3100 | 4173 |
| old_val | 268 | 775 | 1043 |
| strict_train | 1073 | 3100 | 4173 |
| strict_val | 268 | 775 | 1043 |

- Original groups: 3210
- Strict groups: 4081
- Original mixed-label groups: 89
- Strict mixed-label groups: 87
- Selected strict fold index: 4
- Strict split score: 0.000356
- Largest validation strict group: 17

## Verification

- Train/val file overlap count: 0
- Strict duplicate group train/val overlap count: 0
- All files assigned exactly once: True
- Missing from split count: 0
- Extra in split count: 0
- Duplicate assignment count: 0

## Class summaries

### Strict train

| label | class_name | count | percent |
| --- | --- | --- | --- |
| 0 | NORMAL | 1073 | 25.712916367121974 |
| 1 | PNEUMONIA | 3100 | 74.28708363287802 |

### Strict validation

| label | class_name | count | percent |
| --- | --- | --- | --- |
| 0 | NORMAL | 268 | 25.69511025886865 |
| 1 | PNEUMONIA | 775 | 74.30488974113135 |

## Original validation top duplicate groups

| duplicate_group_id | size | normal | pneumonia |
| --- | --- | --- | --- |
| dup_group_00554 | 785 | 86 | 699 |
| dup_group_00422 | 5 | 2 | 3 |
| dup_group_00246 | 4 | 2 | 2 |
| dup_group_00435 | 3 | 2 | 1 |
| dup_group_00280 | 2 | 1 | 1 |
| dup_group_00531 | 2 | 1 | 1 |
| dup_group_00594 | 2 | 1 | 1 |
| dup_group_00696 | 2 | 1 | 1 |
| dup_group_00851 | 2 | 1 | 1 |
| dup_group_00918 | 2 | 1 | 1 |

## Strict group size top 20

| strict_duplicate_group_id | size | normal | pneumonia |
| --- | --- | --- | --- |
| strict_dup_group_01322 | 41 | 0 | 41 |
| strict_dup_group_00993 | 38 | 0 | 38 |
| strict_dup_group_00069 | 18 | 3 | 15 |
| strict_dup_group_01042 | 18 | 1 | 17 |
| strict_dup_group_00070 | 17 | 13 | 4 |
| strict_dup_group_01057 | 16 | 0 | 16 |
| strict_dup_group_01137 | 16 | 0 | 16 |
| strict_dup_group_01367 | 15 | 0 | 15 |
| strict_dup_group_01420 | 14 | 0 | 14 |
| strict_dup_group_00029 | 13 | 8 | 5 |
| strict_dup_group_00115 | 13 | 7 | 6 |
| strict_dup_group_00968 | 13 | 1 | 12 |
| strict_dup_group_01204 | 13 | 0 | 13 |
| strict_dup_group_02878 | 13 | 0 | 13 |
| strict_dup_group_00700 | 12 | 1 | 11 |
| strict_dup_group_01375 | 12 | 0 | 12 |
| strict_dup_group_00989 | 11 | 0 | 11 |
| strict_dup_group_01207 | 11 | 0 | 11 |
| strict_dup_group_01357 | 11 | 0 | 11 |
| strict_dup_group_00158 | 10 | 1 | 9 |

## Notes

- This is a split improvement experiment, not a final claim about model quality.
- Patient-level identifiers are still unavailable, so patient-level leakage cannot be fully excluded.
- Large perceptual clusters are split because connected-component chaining can merge visually similar but non-identical studies.
- Exact duplicate evidence remains grouped under `strict_duplicate_group_id`.