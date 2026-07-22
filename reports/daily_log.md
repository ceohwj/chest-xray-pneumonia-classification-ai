# Daily Log - Mission 1

## 오늘 수행한 작업

* 데이터 구조 확인
* train/validation split 확인
* 클래스 분포 분석
* 샘플 이미지 확인
* Baseline Custom CNN 학습 결과 정리
* Confusion Matrix 기반 오류 분석
* Baseline 한계 및 개선 방향 정리

## 확인한 결과

* train 데이터 수: 4,173
* validation 데이터 수: 1,043
* NORMAL 수: train 1,073 / validation 268 / total 1,341
* PNEUMONIA 수: train 3,100 / validation 775 / total 3,875
* best validation accuracy: 0.9769894535
* best validation F1-score: 0.9845758355
* best validation AUROC: 0.9960471834
* confusion matrix: TN 253, FP 15, FN 9, TP 766
* false negative 수: 9
* false positive 수: 15

## 발견한 문제점

* patient-level identifier가 없어 환자 단위 leakage 가능성을 완전히 배제하기 어렵다.
* 내부 validation 성능은 높지만 `custom_cnn_baseline_report.md`에 public leaderboard score 0.73237이 기록되어 있어 internal validation과 test-like 성능 차이 가능성이 확인되었다.
* Grad-CAM 분석이 아직 수행되지 않아 모델이 폐 영역을 보고 판단했는지 확인하기 어렵다.
* FN/FP 샘플은 CSV로 추출되었지만 이미지 단위 정성 검토는 아직 수행되지 않았다.

## 다음 실험 계획

* DenseNet121, ResNet50, EfficientNet-B0/B1 기반 transfer learning을 같은 split에서 비교한다.
* classifier head-only training 후 낮은 learning rate로 fine-tuning을 수행한다.
* class weight, focal loss, sampler, threshold tuning을 통해 FN 감소 가능성을 확인한다.
* AUROC, F1-score, sensitivity, specificity, confusion matrix를 함께 비교한다.
* FN/FP 샘플과 Grad-CAM을 함께 검토해 shortcut feature 학습 가능성을 점검한다.

## 주의할 점

* 이 모델은 연구 및 교육 목적의 Baseline이며 실제 의료 진단을 대체하지 않는다.
* validation set은 1,043장으로 모델 비교에는 사용할 수 있지만 임상적 유효성을 주장하기에는 부족하다.
* patient-level split을 보장할 식별자가 없어 leakage 가능성이 남아 있다.
* leaderboard와 internal validation 성능 차이가 발생할 수 있으므로, 내부 지표만으로 일반화 성능을 단정하면 안 된다.
