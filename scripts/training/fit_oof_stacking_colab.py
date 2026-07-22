"""Fit a Colab-based logistic-regression stacker from saved OOF probabilities.

Colab-ready script. This does not load images and does not run any neural
network forward pass. It only reads saved probability CSV files.

Expected input files:
- oof_densenet.csv: image_id, true_label, pred_prob
- oof_convnext.csv: image_id, true_label, pred_prob
- oof_efficientnet.csv: image_id, true_label, pred_prob
- test_densenet.csv: image_id, pred_prob
- test_convnext.csv: image_id, pred_prob
- test_efficientnet.csv: image_id, pred_prob
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from google.colab import drive
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


drive.mount("/content/drive")


# ==========================================
# 1. Colab paths
# ==========================================
PROJECT_ROOT = "/content/drive/MyDrive/xray_project"
DATA_DIR = f"{PROJECT_ROOT}/data"
PROB_DIR = f"{PROJECT_ROOT}/outputs"
OUTPUT_DIR = f"{PROJECT_ROOT}/stacking_outputs"

SAMPLE_SUBMISSION_PATH = f"{DATA_DIR}/sample_submission.csv"

OOF_FILES = {
    "dense": f"{PROB_DIR}/oof_densenet.csv",
    "conv": f"{PROB_DIR}/oof_convnext.csv",
    "eff": f"{PROB_DIR}/oof_efficientnet.csv",
}
TEST_FILES = {
    "dense": f"{PROB_DIR}/test_densenet.csv",
    "conv": f"{PROB_DIR}/test_convnext.csv",
    "eff": f"{PROB_DIR}/test_efficientnet.csv",
}

FEATURE_COLUMNS = ["prob_dense", "prob_conv", "prob_eff"]
META_MODEL_C = 0.5
THRESHOLDS = np.round(np.arange(0.01, 0.991, 0.01), 2)


# ==========================================
# 2. Loading and normalization
# ==========================================
def require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def normalize_probability_df(path: str, model_key: str, require_label: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {}

    if "image_id" not in df.columns:
        if "file_name" in df.columns:
            rename_map["file_name"] = "image_id"
        elif "id" in df.columns:
            rename_map["id"] = "image_id"

    if "pred_prob" not in df.columns:
        if "prob" in df.columns:
            rename_map["prob"] = "pred_prob"
        elif "prediction" in df.columns:
            rename_map["prediction"] = "pred_prob"

    if require_label and "true_label" not in df.columns:
        if "label" in df.columns:
            rename_map["label"] = "true_label"
        elif "target" in df.columns:
            rename_map["target"] = "true_label"

    df = df.rename(columns=rename_map)
    required = {"image_id", "pred_prob"}
    if require_label:
        required.add("true_label")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns after normalization: {sorted(missing)}")

    keep_cols = ["image_id", "pred_prob"]
    if require_label:
        keep_cols.append("true_label")
    out = df[keep_cols].copy()
    out["image_id"] = out["image_id"].astype(str)
    out = out.rename(columns={"pred_prob": f"prob_{model_key}"})
    if require_label:
        out["true_label"] = out["true_label"].astype(int)
    return out


def load_and_merge_oof() -> pd.DataFrame:
    merged = None
    for model_key, path in OOF_FILES.items():
        require_file(path, f"OOF probability CSV for {model_key}")
        df = normalize_probability_df(path, model_key, require_label=True)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["image_id", "true_label"], how="inner")
    if merged is None:
        raise RuntimeError("No OOF files loaded.")
    if merged[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Merged OOF features contain NaN values.")
    print(f"OOF merged rows: {len(merged)}")
    print(f"OOF class ratio PNEUMONIA: {merged['true_label'].mean():.4f}")
    return merged


def load_and_merge_test() -> pd.DataFrame:
    merged = None
    for model_key, path in TEST_FILES.items():
        require_file(path, f"test probability CSV for {model_key}")
        df = normalize_probability_df(path, model_key, require_label=False)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="image_id", how="inner")
    if merged is None:
        raise RuntimeError("No test files loaded.")
    if merged[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Merged test features contain NaN values.")
    print(f"Test merged rows: {len(merged)}")
    return merged


# ==========================================
# 3. Meta model and thresholding
# ==========================================
def train_meta_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    meta_model = LogisticRegression(
        penalty="l2",
        C=META_MODEL_C,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=42,
    )
    meta_model.fit(X_scaled, y_train)
    return scaler, meta_model


def find_best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in THRESHOLDS:
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate_oof(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    try:
        auroc = roc_auc_score(y_true, probs)
    except ValueError:
        auroc = float("nan")
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "auroc": float(auroc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pneumonia_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "normal_recall": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "predicted_positive_ratio": float(preds.mean()),
        "true_positive_ratio": float(y_true.mean()),
    }


# ==========================================
# 4. Submission
# ==========================================
def create_submission(test_df: pd.DataFrame, meta_test_probs: np.ndarray, threshold: float) -> str:
    require_file(SAMPLE_SUBMISSION_PATH, "sample submission")
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    id_col = "image_id" if "image_id" in sample.columns else "file_name"
    target_cols = [col for col in sample.columns if col != id_col]
    if len(target_cols) != 1:
        raise ValueError(f"Expected one target column in sample_submission, got {target_cols}")
    target_col = target_cols[0]

    pred_df = pd.DataFrame({"image_id": test_df["image_id"].astype(str), "meta_prob": meta_test_probs})
    sample_ids = sample[[id_col]].copy()
    sample_ids["image_id"] = sample_ids[id_col].astype(str)
    merged = sample_ids.merge(pred_df, on="image_id", how="left")
    if merged["meta_prob"].isna().any():
        missing = merged.loc[merged["meta_prob"].isna(), id_col].head(10).tolist()
        raise ValueError(f"Missing stacked predictions for sample rows: {missing}")

    submission = sample.copy()
    submission[target_col] = (merged["meta_prob"].to_numpy(dtype=float) >= threshold).astype(int)
    output_path = Path(OUTPUT_DIR) / "final_stacking_submission.csv"
    submission.to_csv(output_path, index=False)
    return str(output_path)


def save_outputs(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta_oof_probs: np.ndarray,
    meta_test_probs: np.ndarray,
    threshold: float,
    metrics: dict,
    model: LogisticRegression,
) -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_out = oof_df.copy()
    oof_out["meta_prob"] = meta_oof_probs
    oof_out["meta_pred"] = (meta_oof_probs >= threshold).astype(int)
    oof_out.to_csv(output_dir / "stacking_oof_predictions.csv", index=False)

    test_out = test_df.copy()
    test_out["meta_prob"] = meta_test_probs
    test_out.to_csv(output_dir / "stacking_test_probabilities.csv", index=False)

    config = {
        "meta_model": "LogisticRegression",
        "penalty": "l2",
        "C": META_MODEL_C,
        "features": FEATURE_COLUMNS,
        "coefficients": {
            "dense": float(model.coef_[0][0]),
            "conv": float(model.coef_[0][1]),
            "eff": float(model.coef_[0][2]),
        },
        "intercept": float(model.intercept_[0]),
        "best_threshold_by_oof_f1": float(threshold),
        "metrics": metrics,
        "input_files": {"oof": OOF_FILES, "test": TEST_FILES},
    }
    with open(output_dir / "stacking_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ==========================================
# 5. Main
# ==========================================
def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    require_file(SAMPLE_SUBMISSION_PATH, "sample submission")

    print("Loading saved OOF/Test probability CSV files only.")
    oof_df = load_and_merge_oof()
    test_df = load_and_merge_test()

    X_train = oof_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = oof_df["true_label"].to_numpy(dtype=int)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)

    scaler, meta_model = train_meta_model(X_train, y_train)
    print("\nMeta model coefficients")
    print(f"DenseNet coefficient: {meta_model.coef_[0][0]:.6f}")
    print(f"ConvNeXt coefficient: {meta_model.coef_[0][1]:.6f}")
    print(f"EfficientNet coefficient: {meta_model.coef_[0][2]:.6f}")
    print(f"Intercept: {meta_model.intercept_[0]:.6f}")

    meta_oof_probs = meta_model.predict_proba(scaler.transform(X_train))[:, 1]
    best_threshold, best_f1 = find_best_f1_threshold(y_train, meta_oof_probs)
    metrics = evaluate_oof(y_train, meta_oof_probs, best_threshold)
    print("\nBest OOF threshold search")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best OOF F1: {best_f1:.6f}")
    print(f"OOF accuracy: {metrics['accuracy']:.6f}")
    print(f"OOF AUROC: {metrics['auroc']:.6f}")
    print(f"Confusion matrix: TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")

    meta_test_probs = meta_model.predict_proba(scaler.transform(X_test))[:, 1]
    submission_path = create_submission(test_df, meta_test_probs, best_threshold)
    save_outputs(oof_df, test_df, meta_oof_probs, meta_test_probs, best_threshold, metrics, meta_model)

    test_preds = (meta_test_probs >= best_threshold).astype(int)
    print("\nFinal stacking submission saved")
    print(f"Path: {submission_path}")
    print(f"Predicted NORMAL count: {(test_preds == 0).sum()}")
    print(f"Predicted PNEUMONIA count: {(test_preds == 1).sum()}")
    print(f"Predicted PNEUMONIA ratio: {test_preds.mean():.4f}")
    print(
        "Meta test prob min/max/mean/std: "
        f"{meta_test_probs.min():.6f}/{meta_test_probs.max():.6f}/"
        f"{meta_test_probs.mean():.6f}/{meta_test_probs.std():.6f}"
    )


if __name__ == "__main__":
    main()
