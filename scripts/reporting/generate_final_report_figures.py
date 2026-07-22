"""Generate report figures for the final ensemble experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FIGURE_DIR = Path("reports/figures")
OUTPUT_DIR = Path("outputs/gradcam/ensemble_384")


def metrics_from_oof(path: Path, threshold: float = 0.4) -> dict[str, float | int]:
    df = pd.read_csv(path)
    labels = df["label"].astype(int)
    probs = df["prob"].astype(float)
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, probs),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "tp": int(tp),
    }


def load_json_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_threshold_tradeoff_plot() -> None:
    df = pd.read_csv(OUTPUT_DIR / "threshold_tradeoff_030_040_050.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(df))
    width = 0.34

    ax.bar([i - width / 2 for i in x], df["fn"], width=width, label="FN", color="#d95f02")
    ax.bar([i + width / 2 for i in x], df["fp"], width=width, label="FP", color="#1b9e77")
    for i, row in df.iterrows():
        ax.text(i - width / 2, row["fn"] + 0.8, str(int(row["fn"])), ha="center", va="bottom", fontsize=10)
        ax.text(i + width / 2, row["fp"] + 0.8, str(int(row["fp"])), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{v:.2f}" for v in df["threshold"]])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Error count")
    ax.set_title("Threshold FN/FP Trade-off")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "threshold_fn_fp_tradeoff.png", dpi=180)
    plt.close(fig)


def create_final_confusion_matrix_plot() -> None:
    cm = np.array([[1324, 17], [25, 3850]])
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred NORMAL", "Pred PNEUMONIA"])
    ax.set_yticklabels(["Actual NORMAL", "Actual PNEUMONIA"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "#111111"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    ax.set_title("Final 384 Ensemble OOF Confusion Matrix\nThreshold 0.40")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_384_ensemble_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_threshold_metrics_line_chart() -> pd.DataFrame:
    oof_df = pd.read_csv(Path("outputs/results(마지막)/oof_ensemble.csv"))
    rows = []
    thresholds = [0.30, 0.40, 0.50]
    labels = oof_df["label"].astype(int)
    probs = oof_df["prob"].astype(float)
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(labels, preds),
                "sensitivity": recall_score(labels, preds, zero_division=0),
                "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "threshold_metrics_line_chart.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(df["threshold"], df["accuracy"], marker="o", linewidth=2, label="Accuracy", color="#4c78a8")
    ax.plot(df["threshold"], df["sensitivity"], marker="o", linewidth=2, label="Sensitivity", color="#d95f02")
    ax.plot(df["threshold"], df["specificity"], marker="o", linewidth=2, label="Specificity", color="#1b9e77")
    ax.axvline(0.40, color="#4c78a8", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(0.401, 0.978, "OOF 0.40", rotation=90, va="bottom", ha="left", fontsize=9, color="#4c78a8")
    ax.set_ylim(0.975, 1.001)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("OOF Threshold vs Accuracy / Sensitivity / Specificity")
    ax.text(
        0.5,
        0.976,
        "Zoomed y-axis: 0.975-1.000",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "threshold_metrics_line_chart.png", dpi=180)
    plt.close(fig)
    return df


def create_oof_prediction_correlation_matrix() -> pd.DataFrame:
    paths = {
        "DenseNet121-CBAM": Path("outputs/results(마지막)/oof_densenet121.csv"),
        "EfficientNet-B3": Path("outputs/results(마지막)/oof_efficientnet_b3.csv"),
        "ConvNeXt-Tiny": Path("outputs/results(마지막)/oof_convnext_tiny.csv"),
    }
    merged = None
    for model_name, path in paths.items():
        df = pd.read_csv(path)[["file_name", "prob"]].rename(columns={"prob": model_name})
        merged = df if merged is None else merged.merge(df, on="file_name", how="inner")
    if merged is None:
        raise RuntimeError("No OOF probability files were loaded.")
    corr = merged.drop(columns=["file_name"]).corr(method="pearson")
    corr.to_csv(OUTPUT_DIR / "model_oof_prediction_correlation.csv")

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(corr.to_numpy(), vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=25, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.3f}", ha="center", va="center", color="#111111", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Model Prediction Correlation Matrix (OOF)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_prediction_correlation_matrix.png", dpi=180)
    plt.close(fig)
    return corr


def create_model_error_overlap_matrix() -> pd.DataFrame:
    paths = {
        "DenseNet121-CBAM": Path("outputs/results(마지막)/oof_densenet121.csv"),
        "EfficientNet-B3": Path("outputs/results(마지막)/oof_efficientnet_b3.csv"),
        "ConvNeXt-Tiny": Path("outputs/results(마지막)/oof_convnext_tiny.csv"),
    }
    error_sets = {}
    for model_name, path in paths.items():
        df = pd.read_csv(path)
        preds = (df["prob"] > 0.4).astype(int)
        error_sets[model_name] = set(df.loc[preds != df["label"], "file_name"])

    names = list(paths)
    values = []
    for row_name in names:
        row = []
        for col_name in names:
            intersection = len(error_sets[row_name] & error_sets[col_name])
            union = len(error_sets[row_name] | error_sets[col_name])
            row.append(intersection / union if union else 1.0)
        values.append(row)

    overlap = pd.DataFrame(values, index=names, columns=names)
    overlap.to_csv(OUTPUT_DIR / "model_error_overlap_jaccard.csv")

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(overlap.to_numpy(), vmin=0.0, vmax=1.0, cmap="OrRd")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yticklabels(names)
    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            ax.text(j, i, f"{overlap.iloc[i, j]:.3f}", ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard overlap")
    ax.set_title("Model Error Overlap Matrix (OOF, threshold 0.40)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_error_overlap_matrix.png", dpi=180)
    plt.close(fig)
    return overlap


def create_pseudo_label_probability_histogram() -> pd.DataFrame:
    before = pd.read_csv("outputs/Densenet+efficientnet_B3+convnext_384_pseudo/test_probs_before_pseudo.csv")
    after = pd.read_csv("outputs/Densenet+efficientnet_B3+convnext_384_pseudo/test_probs_384_pseudo.csv")
    before_probs = before["pred_prob"].astype(float)
    after_probs = after["pred_prob"].astype(float)
    stats = pd.DataFrame(
        [
            {
                "stage": "before_pseudo_labeling",
                "mean": before_probs.mean(),
                "std": before_probs.std(),
                "min": before_probs.min(),
                "max": before_probs.max(),
            },
            {
                "stage": "after_pseudo_labeling",
                "mean": after_probs.mean(),
                "std": after_probs.std(),
                "min": after_probs.min(),
                "max": after_probs.max(),
            },
        ]
    )
    stats.to_csv(OUTPUT_DIR / "pseudo_label_probability_distribution_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, 1, 26)
    ax.hist(before_probs, bins=bins, alpha=0.55, density=True, label="Before pseudo-labeling", color="#4c78a8")
    ax.hist(after_probs, bins=bins, alpha=0.50, density=True, label="After pseudo-labeling", color="#d95f02")
    ax.axvline(0.1, color="#555555", linestyle="--", linewidth=1)
    ax.axvline(0.9, color="#555555", linestyle="--", linewidth=1)
    ax.text(0.105, ax.get_ylim()[1] * 0.92, "Normal pseudo <= 0.10", fontsize=9, ha="left")
    ax.text(0.895, ax.get_ylim()[1] * 0.82, "Pneumonia pseudo >= 0.90", fontsize=9, ha="right")
    ax.set_xlabel("Predicted probability of PNEUMONIA")
    ax.set_ylabel("Density")
    ax.set_title("Test Probability Distribution Before vs After Pseudo-labeling")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pseudo_label_probability_histogram.png", dpi=180)
    plt.close(fig)
    return stats


def create_model_comparison_plot() -> pd.DataFrame:
    single_rows = [
        {"model": "DenseNet121 fine-tune", "protocol": "224px single validation", "threshold": 0.50, "fn": 15, "fp": 10},
        {"model": "EfficientNet-B0 fine-tune", "protocol": "224px single validation", "threshold": 0.50, "fn": 24, "fp": 7},
        {"model": "ConvNeXt-Tiny fine-tune", "protocol": "224px single validation", "threshold": 0.50, "fn": 30, "fp": 3},
        {"model": "ResNet50 fine-tune", "protocol": "224px single validation", "threshold": 0.50, "fn": 21, "fp": 19},
        {"model": "EfficientNet-B3 fine-tune", "protocol": "224px single validation", "threshold": 0.50, "fn": 44, "fp": 5},
    ]
    single_df = pd.DataFrame(single_rows)
    single_df.to_csv(OUTPUT_DIR / "model_fn_fp_single_validation_for_report.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(single_df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], single_df["fn"], width=width, label="FN", color="#bdbdbd", edgecolor="#333333")
    ax.bar([i + width / 2 for i in x], single_df["fp"], width=width, label="FP", color="#bdbdbd", edgecolor="#333333", hatch="//", alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(single_df["model"], rotation=25, ha="right")
    ax.set_ylabel("Error count")
    ax.set_title("224px Single Validation FN/FP Comparison (threshold 0.50)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_fn_fp_single_validation.png", dpi=180)
    plt.close(fig)

    final_metrics = metrics_from_oof(Path("outputs/results(마지막)/oof_ensemble.csv"), threshold=0.4)
    oof_rows = [
        {"model": "ConvNeXt-Tiny 384 OOF", "protocol": "384px 5-fold OOF", "threshold": 0.50, "fn": 35, "fp": 4},
        {"model": "EfficientNet-B3 384 OOF", "protocol": "384px 5-fold OOF", "threshold": 0.50, "fn": 44, "fp": 5},
        {
            "model": "Final 384 ensemble OOF",
            "protocol": "384px 5-fold OOF",
            "threshold": 0.40,
            "fn": final_metrics["fn"],
            "fp": final_metrics["fp"],
        },
    ]
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(OUTPUT_DIR / "model_fn_fp_384_oof_for_report.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = range(len(oof_df))
    ax.bar([i - width / 2 for i in x], oof_df["fn"], width=width, label="FN", color="#80b1d3", edgecolor="#333333")
    ax.bar([i + width / 2 for i in x], oof_df["fp"], width=width, label="FP", color="#80b1d3", edgecolor="#333333", hatch="//", alpha=0.9)
    tick_labels = [f"{row.model}\nth={row.threshold:.2f}" for row in oof_df.itertuples(index=False)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Error count")
    ax.set_title("384px 5-fold OOF FN/FP Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_fn_fp_384_oof.png", dpi=180)
    plt.close(fig)

    combined_df = pd.concat([single_df, oof_df], ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "model_comparison_for_report.csv", index=False)
    return combined_df


def create_flow_diagram() -> None:
    steps = [
        "Custom CNN\nBaseline",
        "Transfer Learning\nComparison",
        "Threshold\nSweep",
        "384px Standalone\nOOF",
        "384px Ensemble\nCLAHE + TTA",
        "Pseudo-labeling",
        "Grad-CAM\nReview",
        "Final Insight",
    ]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    y = 0.68
    box_w = 1.35
    box_h = 0.42
    gap = 0.22

    for i, step in enumerate(steps):
        x = i * (box_w + gap)
        box = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.04",
            linewidth=1.2,
            edgecolor="#34495e",
            facecolor="#eef4f7" if i < 4 else "#e9f5ec",
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + box_h / 2, step, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + box_w + 0.02, y + box_h / 2),
                (x + box_w + gap - 0.02, y + box_h / 2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#34495e",
            )
            ax.add_patch(arrow)

    branch_x = 3 * (box_w + gap)
    branch_y = 0.24
    branch_w = 3.9
    branch_h = 0.34
    branch = FancyBboxPatch(
        (branch_x, branch_y),
        branch_w,
        branch_h,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.1,
        edgecolor="#6b5b2a",
        facecolor="#fff7d6",
    )
    ax.add_patch(branch)
    ax.text(
        branch_x + branch_w / 2,
        branch_y + branch_h / 2,
        "Robustness Checks: Strict Split Revalidation / ConvNeXt Ablation Study",
        ha="center",
        va="center",
        fontsize=9.5,
    )

    for source_step in [3, 4]:
        source_x = source_step * (box_w + gap) + box_w / 2
        arrow = FancyArrowPatch(
            (source_x, y - 0.02),
            (branch_x + branch_w / 2, branch_y + branch_h + 0.02),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#6b5b2a",
            connectionstyle="arc3,rad=-0.12" if source_step == 3 else "arc3,rad=0.12",
        )
        ax.add_patch(arrow)

    ax.set_xlim(-0.15, len(steps) * (box_w + gap) - gap + 0.15)
    ax.set_ylim(0.1, 1.32)
    ax.set_title("Experiment Flow", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "experiment_flow_diagram.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_final_confusion_matrix_plot()
    create_threshold_tradeoff_plot()
    create_threshold_metrics_line_chart()
    create_oof_prediction_correlation_matrix()
    create_model_error_overlap_matrix()
    create_pseudo_label_probability_histogram()
    create_model_comparison_plot()
    create_flow_diagram()
    print(FIGURE_DIR / "final_384_ensemble_confusion_matrix.png")
    print(FIGURE_DIR / "threshold_fn_fp_tradeoff.png")
    print(FIGURE_DIR / "threshold_metrics_line_chart.png")
    print(FIGURE_DIR / "model_prediction_correlation_matrix.png")
    print(FIGURE_DIR / "model_error_overlap_matrix.png")
    print(FIGURE_DIR / "pseudo_label_probability_histogram.png")
    print(FIGURE_DIR / "model_fn_fp_single_validation.png")
    print(FIGURE_DIR / "model_fn_fp_384_oof.png")
    print(FIGURE_DIR / "experiment_flow_diagram.png")
    print(OUTPUT_DIR / "model_comparison_for_report.csv")


if __name__ == "__main__":
    main()
