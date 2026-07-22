# Mission 1 Data Summary

이 문서는 `Mission 1. 데이터 이해 및 Baseline 구축` 보고서에 사용한 데이터 확인 결과를 요약한 파일이다. 이 프로젝트는 연구 및 교육 목적의 흉부 X-ray NORMAL/PNEUMONIA 분류 프로젝트이며, 실제 의료 진단을 대체하지 않는다.

## 확인한 입력 파일

* `data/splits/train_strict_duplicate_aware.csv`
* `data/splits/val_strict_duplicate_aware.csv`
* `reports/artifacts/baseline/config.json`
* `reports/artifacts/baseline/best_metrics.json`
* `reports/artifacts/baseline/final_metrics.json`
* `reports/artifacts/baseline/confusion_matrix.csv`
* `reports/artifacts/baseline/false_negatives.csv`
* `reports/artifacts/baseline/false_positives.csv`
* `reports/artifacts/baseline/run_report.md`
* `kaggle/train_custom_cnn_baseline.py`
* `src/models/baseline_cnn.py`
* `src/data/transforms.py`
* `src/data/dataset.py`

## 데이터 구조

| 항목 | 값 |
| --- | ---: |
| Train rows | 4,173 |
| Validation rows | 1,043 |
| Total rows | 5,216 |
| Train/validation file overlap | 0 |
| `duplicate_group_id` overlap | 2 |
| `strict_duplicate_group_id` overlap | 0 |

주요 컬럼:

* `file_name`
* `label`
* `duplicate_group_id`
* `class_name`
* `duplicate_group_size`
* `strict_group_policy`
* `strict_duplicate_group_id`
* `strict_audit_flag`

Label mapping:

* NORMAL = 0
* PNEUMONIA = 1

## 클래스 분포

| Split | NORMAL 수 | PNEUMONIA 수 | 전체 수 | NORMAL 비율 | PNEUMONIA 비율 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,073 | 3,100 | 4,173 | 25.71% | 74.29% |
| Validation | 268 | 775 | 1,043 | 25.70% | 74.30% |
| Total | 1,341 | 3,875 | 5,216 | 25.71% | 74.29% |

현재 데이터는 PNEUMONIA가 NORMAL보다 많은 불균형 데이터이다. 따라서 Accuracy만으로 성능을 판단하면 안 되며, Sensitivity, Specificity, F1-score, AUROC, Confusion Matrix, FN/FP 개수를 함께 확인해야 한다.

## 샘플 이미지 확인

생성 파일:

* `reports/sample_images_grid.png`

확인한 샘플 이미지 정보:

* 원본 이미지 모드: `L` grayscale
* 샘플 이미지 크기: `(224, 224)`
* 모델 입력 전 처리: RGB 변환 후 tensor 변환
* Baseline 입력 shape: `[batch_size, 3, 224, 224]`

## Baseline 성능 요약

Best checkpoint 기준:

| Metric | Value |
| --- | ---: |
| Best epoch | 18 |
| Accuracy | 0.9769894535 |
| Precision | 0.9807938540 |
| Recall / Sensitivity | 0.9883870968 |
| Specificity | 0.9440298507 |
| F1-score | 0.9845758355 |
| AUROC | 0.9960471834 |
| PR-AUC | 0.9986248590 |
| Validation loss | 0.0707075906 |

Confusion Matrix:

| | Predicted NORMAL | Predicted PNEUMONIA |
| --- | ---: | ---: |
| Actual NORMAL | 253 | 15 |
| Actual PNEUMONIA | 9 | 766 |

오류 개수:

* False Negative: 9
* False Positive: 15

## 주의 사항

* patient-level identifier는 현재 파일만으로는 확인하기 어렵다.
* strict duplicate group 기준 train/validation overlap은 0개지만, 환자 단위 leakage 가능성은 완전히 배제하기 어렵다.
* `custom_cnn_baseline_report.md`에는 public leaderboard score 0.73237이 기록되어 있어 internal validation 성능과 외부/test-like 성능 차이 가능성을 고려해야 한다.
* Grad-CAM 분석은 아직 수행되지 않아 모델이 폐 영역을 보고 판단했는지 확인하기 어렵다.
* 이 결과는 Baseline 실험 정리이며, 임상적 진단 성능으로 해석하면 안 된다.
