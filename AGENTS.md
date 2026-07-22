# Chest X-ray Classification AI - AGENTS.md

# Project Role

This project is a medical image classification portfolio project.

The goal is to classify chest X-ray images into:

- NORMAL
- PNEUMONIA

This project focuses not only on model performance, but also on:

- reproducible PyTorch experiments
- medical-image-appropriate evaluation
- sensitivity and specificity
- FN/FP analysis
- confusion matrix analysis
- AUROC
- Grad-CAM interpretability
- clear portfolio documentation

This project is for research and education purposes only.
It must not be described as a clinical diagnostic system.

---

# Core Project Principles

- Do not evaluate the model using accuracy alone.
- Prioritize PNEUMONIA recall because FN is the most clinically risky error.
- Always report confusion matrix, precision, recall, F1-score, AUROC, sensitivity, and specificity.
- Separate model performance from clinical validity.
- Avoid overclaiming results from a small validation set.
- Use transfer learning models after building a Custom CNN baseline.
- Use Grad-CAM to check whether the model focuses on lung regions or irrelevant artifacts.
- Keep experiments reproducible and explainable.
- Improve one experiment at a time.

---

# Multi-Agent Workflow

## ChatGPT

Role:

- Experiment Lead
- Medical AI Modeling Mentor
- Result Reviewer
- Portfolio Writer

Responsibilities:

- design experiment strategy
- recommend model candidates
- review hyperparameter tuning direction
- interpret metrics
- analyze FN/FP cases
- interpret Grad-CAM outputs
- write report and portfolio sections
- generate implementation prompts for Codex
- review Codex reports

Do not:

- make clinical claims
- judge model quality by accuracy alone
- ignore data leakage risk
- overstate results from small validation data

---

## Codex

Role:

- PyTorch Implementation Agent
- Reproducible Experiment Engineer
- Rapid Modeling Implementation Agent

Responsibilities:

- implement isolated experiment features
- build Dataset and DataLoader code
- implement transform pipelines
- implement Custom CNN baseline
- implement pretrained transfer learning models
- implement training and validation loops
- implement metric calculation
- implement checkpoint saving
- implement Grad-CAM utilities
- generate reusable experiment scripts
- fix bugs in model training code
- update experiment documentation when needed

Codex should focus on implementation, not final interpretation.

---

## Antigravity

Role:

- Local Build Engineer
- Execution QA Agent
- Integration Reviewer

Responsibilities:

- verify whether training scripts run locally
- check path errors
- check output folder generation
- verify checkpoint saving
- verify metric logs
- verify confusion matrix outputs
- verify Grad-CAM image outputs
- review README run instructions
- check whether documentation matches actual files

Antigravity should not redesign the experiment unless explicitly asked.

---

# Agent Roles

## 1. Experiment Design Agent

Responsibilities:

- define experiment order
- decide baseline and transfer learning strategy
- define evaluation metrics
- decide tuning priorities
- prevent unnecessary model complexity

Must Evaluate:

- Is this experiment necessary?
- Does this compare models fairly?
- Is the validation set large enough to support the conclusion?
- Are FN and FP analyzed separately?
- Is overfitting being checked?

---

## 2. Data Pipeline Agent

Responsibilities:

- check dataset folder structure
- implement image loading
- implement train/validation/test split
- prevent data leakage
- ensure labels are mapped correctly

Must Evaluate:

- Are NORMAL and PNEUMONIA labels correct?
- Is the split reproducible?
- Is there patient-level leakage risk?
- Are image transforms appropriate for chest X-rays?

---

## 3. Modeling Agent

Responsibilities:

- implement Custom CNN baseline
- implement ResNet18/34
- implement DenseNet121
- implement EfficientNet-B0/B1
- support freeze and fine-tuning strategies
- expose model selection through config or argparse

Model Priority:

1. Custom CNN baseline
2. ResNet18
3. DenseNet121
4. EfficientNet-B0
5. ResNet50 or EfficientNet-B1 only if justified

Must Evaluate:

- Is the model too complex for the dataset size?
- Is transfer learning implemented correctly?
- Is the classifier head modified for binary classification?
- Are pretrained weights used properly?

---

## 4. Training Agent

Responsibilities:

- implement training loop
- implement validation loop
- implement loss function
- implement optimizer
- implement scheduler
- save best checkpoint
- log epoch-level metrics

Required Settings:

- random seed
- device
- batch size
- image size
- learning rate
- optimizer
- scheduler
- weight decay
- dropout if used
- number of epochs
- checkpoint path

Recommended Defaults:

- loss: CrossEntropyLoss or BCEWithLogitsLoss
- optimizer: AdamW
- scheduler: ReduceLROnPlateau or CosineAnnealingLR
- image size: 224x224
- batch size: 16 or 32
- initial learning rate:
  - classifier head: 1e-3
  - fine-tuning backbone: 1e-5 to 1e-4

---

## 5. Evaluation Agent

Responsibilities:

- calculate accuracy
- calculate precision
- calculate recall
- calculate F1-score
- calculate AUROC
- calculate confusion matrix
- calculate sensitivity
- calculate specificity
- save prediction results
- identify FN and FP samples

Must Report:

- overall accuracy
- PNEUMONIA precision
- PNEUMONIA recall
- NORMAL recall
- F1-score
- AUROC
- confusion matrix
- FN count
- FP count

Important:

- FN = PNEUMONIA predicted as NORMAL
- FP = NORMAL predicted as PNEUMONIA

FN should be analyzed separately because it is the most dangerous error type in this project.

---

## 6. Grad-CAM Agent

Responsibilities:

- implement Grad-CAM for CNN and pretrained models
- save Grad-CAM images
- generate Grad-CAM for correct samples
- generate Grad-CAM for FN samples
- generate Grad-CAM for FP samples
- organize outputs by model name and error type

Output Structure:

```text
outputs/gradcam/
  custom_cnn/
    correct/
    fn/
    fp/
  resnet18/
    correct/
    fn/
    fp/
  densenet121/
    correct/
    fn/
    fp/
```

Must Evaluate:

- Does the model focus on lung regions?
- Does the model focus on lesion-like areas?
- Does the model focus on text markers?
- Does the model focus on image borders?
- Does the model focus on shoulders, abdomen, or background artifacts?

Do not say "Grad-CAM looks good" without explaining what region the model focused on.

---

## 7. Documentation Agent

Responsibilities:

- update README.md
- update PROJECT_STATUS.md
- update experiment logs
- summarize model results
- document limitations
- prepare portfolio-ready explanations

Report Structure:

- problem definition
- dataset description
- model selection reason
- experiment design
- performance comparison
- FN/FP error analysis
- Grad-CAM interpretation
- limitations
- next improvements

Do not:

- overstate clinical usefulness
- claim diagnostic validity
- hide small validation set limitations
- compare paper results directly without context

---

# Coding Rules

- Use PyTorch.
- Keep code modular.
- Separate dataset, transforms, models, training, evaluation, metrics, and Grad-CAM.
- Avoid hard-coded absolute paths.
- Use argparse or config files for experiment settings.
- Keep functions focused on one responsibility.
- Do not change unrelated files.
- Do not rewrite the whole project unless explicitly requested.
- Prefer readable code over clever abstraction.
- Save outputs in organized folders.

Recommended structure:

```text
/src
  datasets.py
  transforms.py
  models.py
  train.py
  evaluate.py
  metrics.py
  gradcam.py
  utils.py

/configs
  baseline_cnn.yaml
  resnet18.yaml
  densenet121.yaml
  efficientnet_b0.yaml

/outputs
  checkpoints/
  logs/
  metrics/
  confusion_matrices/
  gradcam/

/reports
  experiment_summary.md
  error_analysis.md
  gradcam_analysis.md
```

---

# Validation Rules

After implementation, run if possible:

- import check
- one batch DataLoader check
- one batch forward pass
- one epoch smoke test
- metric calculation check
- checkpoint saving check
- Grad-CAM output check

If verification cannot run, explain the environment limitation clearly.

---

# Codex Report Format

After every task, Codex must return:

## Codex Report

### Files changed

- ...

### What changed

- ...

### Verification

- ...

### Results

- ...

### Issues

- ...

### Next recommended task

- ...

---

# Prompt Toolkit

Use the smallest prompt surface that fits the work:

- CODEX_DATASET_PROMPT.md
  - Dataset, DataLoader, split, transform implementation

- CODEX_BASELINE_PROMPT.md
  - Custom CNN baseline implementation

- CODEX_TRANSFER_PROMPT.md
  - ResNet, DenseNet, EfficientNet implementation

- CODEX_TRAINING_PROMPT.md
  - train/validation loop, optimizer, scheduler, checkpoint

- CODEX_EVALUATION_PROMPT.md
  - metrics, confusion matrix, FN/FP extraction, AUROC

- CODEX_GRADCAM_PROMPT.md
  - Grad-CAM implementation and output organization

- CODEX_REPORT_PROMPT.md
  - summarize implementation results

Deprecated prompt files should not be recreated unless the workflow becomes complex enough to justify separate templates.

---

# Technical Debt Policy

Allowed:

- simple baseline code during early experiments
- small duplication during initial model comparison
- lightweight local logging before introducing experiment tracking tools

Avoid:

- large rewrites during baseline stage
- premature MLflow/W&B integration
- unnecessary abstraction layers
- changing model and data pipeline at the same time
- optimizing only for leaderboard-style accuracy

Technical debt should be:

- documented
- isolated
- intentionally temporary

---

# Final Principle

Build the experiment slowly, reproducibly, and critically.

A medically responsible classification project is not proven by high accuracy alone.
The project is stronger when it clearly explains model behavior, error cases, limitations, and why the result should not be overclaimed.
