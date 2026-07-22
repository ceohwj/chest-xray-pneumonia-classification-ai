# DataLoader and Forward-pass Smoke Test

This is a pre-training smoke test for a research/education project. It does not evaluate model performance.

## Inputs

- train split: `outputs/train_split.csv`
- validation split: `outputs/val_split.csv`
- image directory: `data/images`
- batch size: `16`
- image size: `224x224`
- label mapping: `0=NORMAL`, `1=PNEUMONIA`

## Transform Check

- Transform: deterministic RGB conversion, resize to 224x224, convert to float tensor in [0, 1].
- No augmentation was applied.
- No vertical flip or medically inappropriate augmentation was used.

## Train Batch

- image_shape: `[16, 3, 224, 224]`
- label_shape: `[16]`
- image_dtype: `torch.float32`
- label_dtype: `torch.int64`
- image_min: `0.0`
- image_max: `1.0`
- label_distribution: `{'0 (NORMAL)': 1, '1 (PNEUMONIA)': 15}`
- labels_are_integer: `True`
- labels_in_expected_set: `True`

## Validation Batch

- image_shape: `[16, 3, 224, 224]`
- label_shape: `[16]`
- image_dtype: `torch.float32`
- label_dtype: `torch.int64`
- image_min: `0.0`
- image_max: `0.9921568632125854`
- label_distribution: `{'0 (NORMAL)': 16}`
- labels_are_integer: `True`
- labels_in_expected_set: `True`

## Forward-pass Check

- model: `BaselineCNN`
- output_shape: `[16, 2]`
- expected_output_shape: `[16, 2]`
- label_shape_for_loss: `[16]`
- loss_function: `CrossEntropyLoss`
- loss_value: `0.7091091275215149`
- loss_is_finite: `True`

## Verification

- train_path_check: `{'checked': 32, 'missing': [], 'passes': True}`
- val_path_check: `{'checked': 32, 'missing': [], 'passes': True}`
- train_split_rows: `4173`
- val_split_rows: `1043`
- custom_cnn_forward_pass_succeeded: `True`
- training_started: `False`

## Issues

- No loading, shape, or loss-computation issues were found.
- Full model training has not started yet.