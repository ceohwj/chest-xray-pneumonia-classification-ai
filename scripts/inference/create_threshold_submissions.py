"""Create thresholded submissions from saved probability CSV files.

Leaderboard metric for this hackathon: accuracy.

Usage examples:
  python -m scripts.inference.create_threshold_submissions \
    --test-probs /kaggle/working/test_probs.csv \
    --sample-submission /kaggle/input/datasets/hyunwoo11/submission/sample_submission.csv \
    --output-dir /kaggle/working

  python -m scripts.inference.create_threshold_submissions \
    --test-probs /kaggle/working/test_probs.csv \
    --oof-probs /kaggle/working/oof_ensemble.csv \
    --sample-submission /kaggle/input/datasets/hyunwoo11/submission/sample_submission.csv \
    --output-dir /kaggle/working
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


DEFAULT_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.48, 0.49, 0.50, 0.51, 0.52, 0.55, 0.60, 0.65, 0.70]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate threshold submissions from test probabilities. Optimizes threshold by OOF accuracy when labels are available."
    )
    parser.add_argument("--test-probs", required=True, help="CSV with file_name and prob columns for test predictions.")
    parser.add_argument("--sample-submission", required=True, help="sample_submission.csv used for file_name order.")
    parser.add_argument("--output-dir", default="outputs/submissions", help="Directory for generated submissions.")
    parser.add_argument("--oof-probs", default=None, help="Optional OOF CSV with file_name, label, and prob columns.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None, help="Thresholds to use when OOF labels are unavailable.")
    parser.add_argument("--threshold-step", type=float, default=0.001, help="OOF threshold search step.")
    return parser.parse_args()


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def threshold_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": safe_auroc(labels, probs),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "pneumonia_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "normal_recall": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "predicted_positive_ratio": float(preds.mean()),
        "true_positive_ratio": float(labels.mean()),
    }


def find_best_accuracy_threshold(oof_df: pd.DataFrame, threshold_step: float) -> tuple[float, dict]:
    required = {"label", "prob"}
    missing = required.difference(oof_df.columns)
    if missing:
        raise ValueError(f"OOF probability file is missing required columns: {sorted(missing)}")

    labels = oof_df["label"].to_numpy(dtype=int)
    probs = oof_df["prob"].to_numpy(dtype=float)
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in np.arange(0.05, 0.950001, threshold_step):
        preds = (probs >= threshold).astype(int)
        accuracy = accuracy_score(labels, preds)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_threshold = round(float(threshold), 3)

    metrics = threshold_metrics(labels, probs, best_threshold)
    metrics["best_accuracy"] = metrics.pop("accuracy")
    metrics["best_threshold"] = best_threshold
    return best_threshold, metrics


def build_threshold_list(best_threshold: float | None, user_thresholds: list[float] | None) -> list[float]:
    if best_threshold is not None:
        candidates = [
            best_threshold - 0.10,
            best_threshold - 0.06,
            best_threshold - 0.03,
            best_threshold - 0.02,
            best_threshold - 0.01,
            best_threshold,
            best_threshold + 0.01,
            best_threshold + 0.02,
            best_threshold + 0.03,
            best_threshold + 0.06,
            best_threshold + 0.10,
        ]
    else:
        candidates = user_thresholds if user_thresholds else DEFAULT_THRESHOLDS
    return sorted({round(float(np.clip(threshold, 0.05, 0.95)), 3) for threshold in candidates})


def validate_submission(sample_df: pd.DataFrame, submission_df: pd.DataFrame) -> None:
    if list(sample_df["file_name"]) != list(submission_df["file_name"]):
        raise ValueError("submission file_name order does not match sample_submission.csv")
    if submission_df["label"].isna().any():
        raise ValueError("submission label contains NaN")
    if not np.issubdtype(submission_df["label"].dtype, np.integer):
        raise TypeError(f"submission label dtype must be int, got {submission_df['label'].dtype}")


def save_threshold_submissions(test_probs_df: pd.DataFrame, sample_df: pd.DataFrame, output_dir: Path, thresholds: list[float]) -> list[str]:
    required = {"file_name", "prob"}
    missing = required.difference(test_probs_df.columns)
    if missing:
        raise ValueError(f"test probability file is missing required columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prob_map = dict(zip(test_probs_df["file_name"], test_probs_df["prob"]))
    aligned_probs = sample_df["file_name"].map(prob_map)
    if aligned_probs.isna().any():
        missing_names = sample_df.loc[aligned_probs.isna(), "file_name"].head(10).tolist()
        raise ValueError(f"sample_submission contains file names without probabilities: {missing_names}")

    generated_paths = []
    probs = aligned_probs.to_numpy(dtype=float)
    print("Leaderboard metric: accuracy. Thresholds are selected for class-ratio/accuracy behavior, not F1.")
    for threshold in thresholds:
        labels = (probs >= threshold).astype(int)
        submission_df = pd.DataFrame({"file_name": sample_df["file_name"].values, "label": labels.astype(int)})
        validate_submission(sample_df, submission_df)

        normal_count = int((submission_df["label"] == 0).sum())
        pneumonia_count = int((submission_df["label"] == 1).sum())
        pneumonia_ratio = float(submission_df["label"].mean())
        print(
            f"threshold={threshold:.3f} "
            f"predicted NORMAL count={normal_count} "
            f"predicted PNEUMONIA count={pneumonia_count} "
            f"predicted PNEUMONIA ratio={pneumonia_ratio:.4f} "
            f"prob min/max/mean/std={probs.min():.6f}/{probs.max():.6f}/{probs.mean():.6f}/{probs.std():.6f}"
        )

        path = output_dir / f"submission_th_{threshold:.3f}.csv"
        submission_df.to_csv(path, index=False)
        generated_paths.append(str(path))
    return generated_paths


def main() -> None:
    args = parse_args()
    test_probs_df = pd.read_csv(args.test_probs)
    sample_df = pd.read_csv(args.sample_submission)
    output_dir = Path(args.output_dir)

    best_threshold = None
    if args.oof_probs:
        oof_df = pd.read_csv(args.oof_probs)
        best_threshold, metrics = find_best_accuracy_threshold(oof_df, args.threshold_step)
        config_path = output_dir / "best_accuracy_threshold_config.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, allow_nan=True)
        print(f"Best OOF accuracy threshold: {best_threshold:.3f}")
        print(f"Saved accuracy threshold config: {config_path}")

    thresholds = build_threshold_list(best_threshold, args.thresholds)
    generated_paths = save_threshold_submissions(test_probs_df, sample_df, output_dir, thresholds)
    print("Generated submissions:")
    for path in generated_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
