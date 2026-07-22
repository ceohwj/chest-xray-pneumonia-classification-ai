# Mission 1. 데이터 이해 및 Baseline 구축

## 1. 데이터 구조 및 클래스 분포 분석

### 1.1 데이터 구조

실제 확인 파일:

* `data/splits/train_strict_duplicate_aware.csv`
* `data/splits/val_strict_duplicate_aware.csv`
* `reports/artifacts/baseline/config.json`
* `reports/artifacts/baseline/best_metrics.json`
* `reports/artifacts/baseline/final_metrics.json`
* `reports/artifacts/baseline/confusion_matrix.csv`
* `reports/artifacts/baseline/false_negatives.csv`
* `reports/artifacts/baseline/false_positives.csv`
* `reports/artifacts/baseline/run_report.md`

데이터 구조 확인 결과:

* 전체 데이터 수: 5,216장
* train 데이터 수: 4,173장
* validation 데이터 수: 1,043장
* 이미지 경로 컬럼: `file_name`
* label 컬럼: `label`
* label mapping:
  * NORMAL = 0
  * PNEUMONIA = 1
* `data/splits/train_strict_duplicate_aware.csv`와 `data/splits/val_strict_duplicate_aware.csv`에는 `duplicate_group_id`, `strict_duplicate_group_id` 컬럼이 있다.
* 원본 `duplicate_group_id` 기준으로는 train/validation 사이에 겹치는 그룹이 2개 확인되었다.
* strict split에서 사용하는 `strict_duplicate_group_id` 기준으로는 train/validation 사이에 겹치는 그룹이 0개 확인되었다.
* train/validation 사이의 파일명 overlap은 0개로 확인되었다.

따라서 이 Mission 1 보고서에서는 strict duplicate-aware split 기준으로 Baseline 결과를 정리한다. 다만 patient-level identifier는 파일에서 확인되지 않았으므로, 같은 환자 단위의 leakage 가능성을 완전히 배제하기는 어렵다.

### 1.2 클래스 분포

`data/splits/train_strict_duplicate_aware.csv`와 `data/splits/val_strict_duplicate_aware.csv` 기준 클래스 분포는 다음과 같다.

| Split      | NORMAL 수 | PNEUMONIA 수 | 전체 수 | NORMAL 비율 | PNEUMONIA 비율 |
| ---------- | -------: | ----------: | ---: | --------: | -----------: |
| Train      | 1,073 | 3,100 | 4,173 | 25.71% | 74.29% |
| Validation | 268 | 775 | 1,043 | 25.70% | 74.30% |
| Total      | 1,341 | 3,875 | 5,216 | 25.71% | 74.29% |

### 1.3 클래스 불균형 여부

현재 데이터는 PNEUMONIA가 NORMAL보다 훨씬 많은 불균형 데이터이다. 전체 5,216장 중 NORMAL은 1,341장(25.71%), PNEUMONIA는 3,875장(74.29%)이다.

이런 상황에서 Accuracy만 사용하면 문제가 생길 수 있다. 다수 클래스인 PNEUMONIA 위주로 예측해도 전체 정답률이 높게 보일 수 있고, 반대로 NORMAL specificity나 minority class 관점의 오류가 가려질 수 있다. 의료영상 분류에서는 실제 PNEUMONIA를 NORMAL로 예측하는 False Negative가 특히 위험하므로, Accuracy뿐 아니라 PNEUMONIA recall/sensitivity, NORMAL specificity, F1-score, AUROC, Confusion Matrix, FN/FP 개수를 함께 확인해야 한다.

현재 Baseline에서는 class weight를 적용한 `CrossEntropyLoss`를 사용했지만, 클래스 불균형으로 인한 threshold 편향이나 FN/FP trade-off 가능성은 계속 점검해야 한다.

---

## 2. 샘플 이미지 확인

생성한 샘플 이미지 파일:

* `reports/sample_images_grid.png`

확인 결과:

* 원본 샘플 이미지 모드: `L`, 즉 grayscale 이미지로 확인되었다.
* 샘플 이미지 크기: 확인한 샘플 기준 `(224, 224)`였다.
* Dataset 및 transform 코드에서는 이미지를 `RGB`로 변환해 모델 입력에 사용한다.
* Baseline 모델 입력 shape은 `[batch_size, 3, 224, 224]`이다.
* `kaggle/train_custom_cnn_baseline.py` 기준 train transform은 RGB 변환, 224x224 resize, ±7도 rotation, 최대 4% translation, brightness/contrast 각각 ±0.08 범위 조정을 포함한다.
* validation transform은 RGB 변환과 224x224 resize만 적용한다.
* vertical flip은 사용되지 않았다.

NORMAL 샘플 이미지는 전반적으로 grayscale 흉부 X-ray 형태이며, 폐 영역과 주변 흉곽 구조가 보인다. PNEUMONIA 샘플 이미지도 같은 형식의 grayscale 흉부 X-ray이며, 육안상 밝기, 촬영 조건, crop, 가장자리, 마커 등 비질병적 차이가 섞여 보일 수 있다.

주의할 점은 샘플 이미지만 보고 "폐렴이 명확하다"고 단정할 수 없다는 것이다. 흉부 X-ray에서는 클래스 간 육안상 차이가 작을 수 있고, 모델이 폐 영역 자체가 아니라 배경, 테두리, 밝기, crop, 텍스트 마커 같은 shortcut feature를 학습할 가능성이 있다. 따라서 Grad-CAM과 FN/FP 이미지 리뷰가 필요하다.

---

## 3. Baseline 모델 학습

### 3.1 Baseline 모델 구조

현재 Baseline 모델은 Custom CNN이다. 실제 실행 스크립트 `kaggle/train_custom_cnn_baseline.py`와 리포트 `reports/artifacts/baseline/run_report.md` 기준 구조는 다음과 같다.

* 모델명: Custom CNN
* 입력 shape: `[batch_size, 3, 224, 224]`
* 출력 shape: `[batch_size, 2]`
* binary classification 방식: 2-class logits 출력 후 `CrossEntropyLoss` 사용
* convolution block 구성:
  * Conv2d(3 -> 32), BatchNorm2d, ReLU, MaxPool2d
  * Conv2d(32 -> 64), BatchNorm2d, ReLU, MaxPool2d
  * Conv2d(64 -> 128), BatchNorm2d, ReLU, MaxPool2d
  * Conv2d(128 -> 256), BatchNorm2d, ReLU, MaxPool2d
* activation function: ReLU
* pooling layer: 각 block의 MaxPool2d, 마지막 AdaptiveAvgPool2d((1, 1))
* classifier head: Flatten -> Dropout(0.4) -> Linear(256, 2)
* trainable parameter count: 389,410
* loss function: class weight가 적용된 `CrossEntropyLoss`
* optimizer: AdamW
* scheduler 사용 여부: ReduceLROnPlateau 사용
* batch size: 32
* learning rate: 0.001
* epoch 수: 설정 25 epochs, 실제 early stopping으로 23 epochs까지 실행
* early stopping 사용 여부: validation F1-score 기준 patience 5로 사용
* best checkpoint 기준: validation F1-score
* best epoch: 18

### 3.2 이미지 처리 방식

`reports/artifacts/baseline/config.json` 및 `kaggle/train_custom_cnn_baseline.py` 기준 이미지 처리 방식은 다음과 같다.

* image resize 크기: 224x224
* normalization 방식: pixel 값을 `[0, 1]` 범위로 스케일링한다. ImageNet mean/std normalization은 Baseline Custom CNN 실행 코드에서 확인되지 않았다.
* grayscale 이미지를 3채널로 변환했는지 여부: 원본 샘플은 `L` 모드 grayscale이지만, Dataset과 transform에서 `RGB`로 변환해 3채널 입력으로 사용한다.
* train transform:
  * RGB 변환
  * 224x224 resize
  * ±7도 rotation
  * 최대 4% translation
  * brightness ±0.08
  * contrast ±0.08
  * tensor 변환 및 `[0, 1]` scaling
* validation transform:
  * RGB 변환
  * 224x224 resize
  * tensor 변환 및 `[0, 1]` scaling
* augmentation 종류:
  * mild rotation
  * small translation
  * brightness/contrast jitter
* vertical flip 사용 여부: 사용되지 않았다.

의료영상에서 vertical flip은 해부학적 방향성을 훼손할 수 있어 부적절할 수 있다. 이번 Baseline에서는 vertical flip이 확인되지 않았다. Horizontal flip도 현재 Baseline 코드에서는 확인되지 않았다. ±7도 rotation, 작은 translation, brightness/contrast jitter는 비교적 mild augmentation으로 볼 수 있지만, augmentation은 해부학적 의미를 훼손하지 않는 범위에서만 사용해야 한다.

---

## 4. 데일리 로그 작성

데일리 로그는 `reports/daily_log.md`에 작성했다.

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
* 내부 validation 성능이 높지만, Custom CNN baseline report에는 public leaderboard score 0.73237이 기록되어 있어 internal validation과 외부/test-like 성능 차이 가능성이 확인되었다.
* Grad-CAM 분석이 아직 수행되지 않아 모델이 폐 영역을 보고 판단했는지 확인하기 어렵다.
* FN/FP 샘플이 CSV로 추출되었지만 이미지 단위 정성 검토는 아직 수행되지 않았다.

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

---

## 5. Baseline 모델 성능 기록

`reports/artifacts/baseline/best_metrics.json`, `final_metrics.json`, `confusion_matrix.csv`, `training_history.csv` 기준 성능을 정리했다. 아래 주요 표는 best checkpoint, 즉 validation F1-score가 가장 높았던 epoch 18 기준이다.

### 5.1 주요 성능 지표

| Metric               | Value | 의미                                  |
| -------------------- | ----: | ----------------------------------- |
| Accuracy             | 0.9769894535 | 전체 샘플 중 맞게 분류한 비율                   |
| Precision            | 0.9807938540 | PNEUMONIA로 예측한 샘플 중 실제 PNEUMONIA 비율 |
| Recall / Sensitivity | 0.9883870968 | 실제 PNEUMONIA 중 모델이 PNEUMONIA로 맞춘 비율 |
| Specificity          | 0.9440298507 | 실제 NORMAL 중 모델이 NORMAL로 맞춘 비율       |
| F1-score             | 0.9845758355 | Precision과 Recall의 조화 평균            |
| AUROC                | 0.9960471834 | threshold 변화에 따른 분류 성능              |

보조 기록:

* best epoch: 18
* best validation loss: 0.0707075906
* PR-AUC: 0.9986248590
* final epoch: 23
* final epoch F1-score: 0.9832041344
* final epoch AUROC: 0.9966490130

### 5.2 Confusion Matrix

`reports/artifacts/baseline/confusion_matrix.csv` 기준 best checkpoint confusion matrix는 다음과 같다.

|                  | Predicted NORMAL | Predicted PNEUMONIA |
| ---------------- | ---------------: | ------------------: |
| Actual NORMAL    | 253 | 15 |
| Actual PNEUMONIA | 9 | 766 |

### 5.3 FN/FP 오류 분석

`reports/artifacts/baseline/false_negatives.csv`와 `reports/artifacts/baseline/false_positives.csv` 기준 오류 개수는 다음과 같다.

* False Negative 개수: 9
* False Positive 개수: 15

FN은 실제 PNEUMONIA 이미지를 NORMAL로 예측한 경우이다. 의료영상 분류 맥락에서는 질환 의심 사례를 정상으로 놓칠 수 있으므로 가장 위험하게 다뤄야 하는 오류 유형이다. 이 프로젝트는 실제 진단 시스템이 아니지만, 연구 평가 관점에서도 FN은 별도로 추적하고 이미지 단위로 검토해야 한다.

FP는 실제 NORMAL 이미지를 PNEUMONIA로 예측한 경우이다. 실제 의료 환경으로 일반화해 표현하면, FP는 과잉 의심, 불필요한 추가 검사 또는 불안 증가로 이어질 수 있다. 다만 이 프로젝트는 임상 시스템이 아니므로, FP 역시 모델 오류 유형으로만 해석해야 한다.

현재 Baseline은 best checkpoint에서 FN 9개, FP 15개로 FP가 더 많다. 그러나 의료영상 AI 평가에서는 FN의 위험도가 더 크므로, 단순히 FP가 더 많다는 사실만으로 모델이 안전하다고 말할 수 없다. 또한 FN/FP 대부분이 어떤 시각적 특징에서 발생했는지는 현재 CSV만으로는 확인하기 어렵고, 이미지 리뷰와 Grad-CAM 분석이 필요하다.

---

## 6. 문제점 2가지 이상 도출

1. Custom CNN의 feature representation 한계가 있다.

   현재 Baseline은 4개 convolution block으로 구성된 작은 Custom CNN이다. trainable parameter count는 389,410으로 확인되었다. Baseline으로는 적절하지만, pretrained backbone에 비해 X-ray의 복잡한 texture, subtle opacity, 촬영 조건 차이를 충분히 표현하는 데 한계가 있을 가능성이 있다.

2. 내부 validation 성능과 실제 test/leaderboard 성능 차이 가능성이 확인되었다.

   `custom_cnn_baseline_report.md`에는 internal validation 기준 F1-score 0.9845758355, AUROC 0.9960471834가 기록되어 있다. 그러나 같은 리포트에 public leaderboard score 0.73237이 기록되어 있어 internal validation 성능이 외부/test-like 성능을 그대로 보장하지 않는다는 문제가 확인되었다.

3. patient-level identifier 부재로 인한 leakage 가능성을 완전히 배제하기 어렵다.

   split 파일에는 `duplicate_group_id`와 `strict_duplicate_group_id`가 있지만, patient-level identifier는 확인되지 않았다. strict duplicate group overlap은 0개로 확인되었지만, 같은 환자의 다른 촬영 이미지가 train/validation에 나뉘었는지는 현재 파일만으로는 확인하기 어렵다.

4. False Negative 발생 위험이 있다.

   best checkpoint 기준 FN은 9개 확인되었다. 수치상 recall/sensitivity는 0.9883870968로 높지만, 의료영상 분류에서는 PNEUMONIA를 NORMAL로 예측하는 FN을 별도로 검토해야 한다.

5. train/test near-duplicate 가능성이 있다.

   `reports/duplicate_aware_split_report.md`에는 train/test near-duplicate risk로 `train_0781.png/test_0066.png` 1건이 기록되어 있다. 이는 validation split 자체의 strict group overlap과는 별개로, test-like 평가 해석에 주의가 필요하다는 의미이다.

6. Grad-CAM 기반 해석이 아직 없다.

   현재 Baseline 결과에는 Grad-CAM 이미지가 포함되어 있지 않다. 따라서 모델이 폐 영역 또는 병변 의심 부위를 보고 판단했는지, 아니면 배경, 가장자리, 텍스트 마커 같은 shortcut feature를 봤는지는 현재 파일만으로는 확인하기 어렵다.

7. X-ray의 배경, 가장자리, 텍스트 마커 등을 shortcut feature로 학습할 가능성이 있다.

   `reports/pre_baseline_audit.md`는 border, crop, brightness, resolution shortcut 가능성을 점검하기 위한 figure가 생성되었다고 기록한다. 실제 shortcut 학습 여부는 Grad-CAM과 오류 샘플 리뷰가 필요하므로, 현재 단계에서는 가능성으로 표현하는 것이 적절하다.

---

## 7. 개선 방향 가설 수립

Baseline 이후 개선 방향을 실험 가능한 가설 형태로 정리한다.

### 가설 1. Pretrained Transfer Learning 적용

Custom CNN은 X-ray의 복잡한 texture와 병변 패턴을 충분히 학습하기 어려울 수 있다. ImageNet pretrained model을 사용하면 일반적인 edge, texture, shape feature를 초기값으로 활용할 수 있으므로, 작은 의료영상 데이터셋에서도 일반화 성능이 개선될 가능성이 있다.

실험 후보:

* DenseNet121
* ResNet50
* EfficientNet-B0/B1

실험 방식:

* classifier head만 먼저 학습
* 이후 backbone 일부 또는 전체를 낮은 learning rate로 fine-tuning
* Custom CNN baseline과 동일한 split에서 비교

### 가설 2. Threshold-independent metric 기반 평가

Accuracy는 특정 threshold에서의 정답률만 보여주기 때문에, 클래스 불균형이나 threshold 선택에 민감하다. AUROC는 threshold 변화에 따른 분류 성능을 평가할 수 있으므로, Baseline 모델의 전반적인 분리 능력을 확인하는 데 적절하다.

추가 평가 지표:

* AUROC
* F1-score
* PNEUMONIA recall/sensitivity
* NORMAL recall/specificity
* Confusion matrix

### 가설 3. FN/FP 중심 오류 분석

의료영상 분류에서는 PNEUMONIA 환자를 NORMAL로 예측하는 False Negative가 특히 위험하다. 따라서 단순 accuracy 개선보다 FN을 줄이는 방향으로 threshold 조정, class weight, focal loss, sampler를 검토한다.

실험 후보:

* class weight 적용
* WeightedRandomSampler 적용
* Focal Loss 적용
* threshold tuning
* recall 중심 model selection

### 가설 4. Grad-CAM 기반 해석 가능성 검토

성능이 비슷한 모델이라도 판단 근거는 다를 수 있다. Grad-CAM을 통해 모델이 폐 영역과 병변 의심 부위를 보고 판단하는지, 아니면 배경/마커/이미지 가장자리 같은 shortcut feature를 보는지 확인한다.

분석 대상:

* True Positive
* True Negative
* False Positive
* False Negative

---

# Reflection Questions

## Q1. Baseline으로 사용한 모델은 어떤 구조이며, 이미지 데이터를 어떻게 처리하나요?

Baseline 모델은 4개의 convolution block을 가진 Custom CNN이다. 입력 이미지는 224x224로 resize되고, 원본 grayscale 이미지는 RGB로 변환되어 `[batch_size, 3, 224, 224]` 형태로 모델에 들어간다. 각 convolution block은 Conv2d, BatchNorm2d, ReLU, MaxPool2d로 구성되며, 마지막에는 AdaptiveAvgPool2d를 적용한다. classifier는 Flatten, Dropout(0.4), Linear(256, 2)로 구성되어 NORMAL/PNEUMONIA 2개 class logit을 출력한다. 학습에는 class weight가 적용된 CrossEntropyLoss를 사용했다.

## Q2. 현재 사용한 평가 지표는 무엇이며, 왜 해당 지표가 적절한가요?

현재 사용한 평가 지표는 Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score, AUROC, Confusion Matrix이다. Accuracy는 전체 정답률을 보여주지만 클래스 불균형과 threshold 선택의 영향을 받는다. Precision은 PNEUMONIA로 예측한 샘플의 신뢰도를, Recall/Sensitivity는 실제 PNEUMONIA를 얼마나 놓치지 않았는지를 보여준다. Specificity는 실제 NORMAL을 NORMAL로 맞춘 비율이다. F1-score는 Precision과 Recall의 균형을 보고, AUROC는 threshold 변화에 따른 전반적인 분리 능력을 평가한다. 의료영상에서는 PNEUMONIA recall과 FN 분석이 특히 중요하다.

## Q3. Accuracy 대신 AUC를 사용하는 이유는 무엇인가요?

Accuracy는 특정 threshold에서의 성능이다. threshold가 바뀌면 accuracy도 달라질 수 있다. AUROC는 threshold-independent metric으로, 모델이 NORMAL과 PNEUMONIA를 얼마나 잘 분리하는지 평가할 수 있다. 클래스 불균형 상황에서는 다수 클래스 위주 예측으로도 accuracy가 높게 보일 수 있으므로, accuracy만으로 모델 성능을 판단하기 어렵다.

## Q4. 데이터가 불균형한 경우 어떤 문제가 발생하나요?

데이터가 불균형하면 다수 클래스 위주로 예측해도 accuracy가 높게 나올 수 있다. 이 경우 minority class recall이 낮아질 수 있고, PNEUMONIA class를 놓치는 FN이 증가할 수 있다. 또한 loss가 majority class 중심으로 최적화될 수 있다. 따라서 class weight, sampler, focal loss, threshold tuning이 필요할 수 있다.

## Q5. 현재 데이터는 불균형 상태인가요? 그렇다면 어떻게 확인했나요?

현재 데이터는 불균형 상태이다. `data/splits/train_strict_duplicate_aware.csv` 기준 train 데이터는 NORMAL 1,073장(25.71%), PNEUMONIA 3,100장(74.29%)이다. `data/splits/val_strict_duplicate_aware.csv` 기준 validation 데이터는 NORMAL 268장(25.70%), PNEUMONIA 775장(74.30%)이다. 전체로는 NORMAL 1,341장(25.71%), PNEUMONIA 3,875장(74.29%)이므로 PNEUMONIA가 NORMAL보다 약 2.89배 많다.

## Q6. Baseline 모델의 가장 큰 한계는 무엇인가요?

Baseline 모델의 가장 큰 한계는 내부 validation 성능이 높더라도 일반화 성능과 모델 판단 근거를 보장하지 못한다는 점이다. Custom CNN은 pretrained backbone에 비해 feature representation이 제한적일 수 있고, patient-level split을 완전히 보장할 identifier가 없어 leakage 가능성이 남아 있다. 또한 Grad-CAM 분석이 없으므로 모델이 폐 영역을 보고 판단했는지 확인하기 어렵다. FN/FP 샘플은 CSV로 추출되었지만, 이미지 단위 검토가 아직 필요하다.

## Q7. 해당 문제를 해결하기 위한 개선 방향은 무엇인가요?

개선 방향은 DenseNet121, ResNet50, EfficientNet-B0/B1 기반 transfer learning을 적용하는 것이다. 먼저 classifier head-only training을 수행하고, 이후 backbone을 낮은 learning rate로 fine-tuning한다. class imbalance 대응을 위해 class weight, sampler, focal loss, threshold tuning을 검토한다. 평가는 AUROC, F1-score, Recall/Sensitivity, Specificity, Confusion Matrix를 함께 사용하고, FN/FP 오류를 별도로 분석한다. 마지막으로 Grad-CAM을 통해 모델이 폐 영역과 병변 의심 부위를 보는지, shortcut feature를 보는지 확인한다. validation split과 leaderboard 성능 차이도 함께 검토해야 한다.

---

# 최종 요약

## Mission 1 요약

* 데이터 구조: `file_name`, `label`, `duplicate_group_id`, `strict_duplicate_group_id` 등을 포함한 strict split CSV를 사용했다. train 4,173장, validation 1,043장, 전체 5,216장이다.
* 클래스 분포: 전체 기준 NORMAL 1,341장(25.71%), PNEUMONIA 3,875장(74.29%)으로 PNEUMONIA가 많은 불균형 데이터이다.
* Baseline 모델: 4개 convolution block을 가진 Custom CNN이며, 입력은 `[batch_size, 3, 224, 224]`, 출력은 `[batch_size, 2]`이다.
* 주요 성능: best checkpoint 기준 Accuracy 0.9769894535, Sensitivity 0.9883870968, Specificity 0.9440298507, F1-score 0.9845758355, AUROC 0.9960471834, Confusion Matrix는 TN 253, FP 15, FN 9, TP 766이다.
* 주요 문제점: Custom CNN 표현력 한계, internal validation과 leaderboard 차이 가능성, patient-level leakage 완전 배제 어려움, Grad-CAM 부재, FN/FP 이미지 리뷰 미완료가 있다.
* 개선 방향: DenseNet121, ResNet50, EfficientNet-B0/B1 transfer learning, head-only 학습 후 fine-tuning, threshold tuning, FN/FP 분석, Grad-CAM 해석을 우선 수행한다.
