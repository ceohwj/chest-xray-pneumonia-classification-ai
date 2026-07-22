"""Pre-baseline dataset audit for chest X-ray classification.

This script audits CSV/image mapping, class imbalance, patient-level split
feasibility, duplicate risk, and simple shortcut-learning signals before any
model training is started.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ExifTags, Image, ImageFile, UnidentifiedImageError

try:
    from .split import create_baseline_split, find_group_column, infer_patient_group
except ImportError:
    from split import create_baseline_split, find_group_column, infer_patient_group

ImageFile.LOAD_TRUNCATED_IMAGES = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-baseline chest X-ray dataset audit.")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--test_csv", default="data/test.csv")
    parser.add_argument("--sample_submission_csv", default="data/sample_submission.csv")
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_0_name", default=None, help="Confirmed class name for numeric label 0.")
    parser.add_argument("--label_1_name", default=None, help="Confirmed class name for numeric label 1.")
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "reports": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "outputs": Path("outputs"),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def build_file_index(image_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if not image_dir.exists():
        return index
    for path in image_dir.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            index[path.name].append(path)
            try:
                index[str(path.relative_to(image_dir))].append(path)
            except ValueError:
                pass
    return index


def resolve_image_path(file_name: str, image_dir: Path, split_name: str, index: dict[str, list[Path]]) -> Path | None:
    raw = Path(str(file_name))
    candidates = [
        image_dir / raw,
        image_dir / split_name / raw.name,
        image_dir / raw.name,
    ]
    if raw.parts and raw.parts[0] in {"train", "test"}:
        candidates.insert(0, image_dir / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for key in (str(raw), raw.name):
        if key in index and index[key]:
            return index[key][0]
    return None


def image_basic_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "readable": False}
    row: dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "suffix": path.suffix.lower(),
    }
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            row.update(
                {
                    "readable": True,
                    "format": img.format,
                    "width": width,
                    "height": height,
                    "mode": img.mode,
                    "channels": len(img.getbands()),
                    "aspect_ratio": width / height if height else np.nan,
                    "mean_intensity": float(arr.mean()),
                    "std_intensity": float(arr.std()),
                    "min_intensity": float(arr.min()),
                    "max_intensity": float(arr.max()),
                    "border_mean": border_mean(arr),
                    "corner_mean": corner_mean(arr),
                    "exif_keys": extract_exif_keys(img),
                }
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        row.update({"readable": False, "error": f"{type(exc).__name__}: {exc}"})
    return row


def extract_exif_keys(img: Image.Image) -> str:
    try:
        exif = img.getexif()
    except Exception:
        return ""
    if not exif:
        return ""
    names = []
    for key in exif.keys():
        names.append(ExifTags.TAGS.get(key, str(key)))
    return "|".join(sorted(map(str, names)))


def border_mean(arr: np.ndarray, border_frac: float = 0.05) -> float:
    h, w = arr.shape
    b_h = max(1, int(h * border_frac))
    b_w = max(1, int(w * border_frac))
    mask = np.zeros_like(arr, dtype=bool)
    mask[:b_h, :] = True
    mask[-b_h:, :] = True
    mask[:, :b_w] = True
    mask[:, -b_w:] = True
    return float(arr[mask].mean())


def corner_mean(arr: np.ndarray, corner_frac: float = 0.12) -> float:
    h, w = arr.shape
    c_h = max(1, int(h * corner_frac))
    c_w = max(1, int(w * corner_frac))
    patches = [arr[:c_h, :c_w], arr[:c_h, -c_w:], arr[-c_h:, :c_w], arr[-c_h:, -c_w:]]
    return float(np.concatenate([p.reshape(-1) for p in patches]).mean())


def audit_file_mapping(df: pd.DataFrame, split_name: str, image_dir: Path, index: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    check_rows = []
    metadata_rows = []
    duplicate_names = set(df.loc[df["file_name"].duplicated(keep=False), "file_name"].astype(str))
    for i, row in df.iterrows():
        file_name = str(row["file_name"])
        path = resolve_image_path(file_name, image_dir, split_name, index)
        meta = image_basic_metadata(path)
        base = {
            "split": split_name,
            "row_index": i,
            "file_name": file_name,
            "label": row["label"] if "label" in row else np.nan,
            "resolved_path": str(path) if path else "",
            "exists": bool(meta.get("exists", False)),
            "readable": bool(meta.get("readable", False)),
            "duplicated_file_name": file_name in duplicate_names,
        }
        check_rows.append(base | {"error": meta.get("error", "")})
        metadata_rows.append(base | meta)
        if (i + 1) % 1000 == 0:
            print(f"[audit] checked {split_name}: {i + 1}/{len(df)}")
    return pd.DataFrame(check_rows), pd.DataFrame(metadata_rows)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(path: str, hash_size: int = 8) -> str:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS), dtype=np.float32)
    bits = arr > arr.mean()
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def dhash(path: str, hash_size: int = 8) -> str:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS), dtype=np.float32)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def add_hashes(metadata_df: pd.DataFrame) -> pd.DataFrame:
    df = metadata_df.copy()
    df["sha256"] = ""
    df["ahash"] = ""
    df["dhash"] = ""
    readable = df["readable"].fillna(False).astype(bool)
    for idx, row in df.loc[readable].iterrows():
        path = row["resolved_path"]
        try:
            df.at[idx, "sha256"] = sha256_file(path)
            df.at[idx, "ahash"] = ahash(path)
            df.at[idx, "dhash"] = dhash(path)
        except Exception as exc:
            df.at[idx, "hash_error"] = f"{type(exc).__name__}: {exc}"
    return df


def duplicate_report(hashed_df: pd.DataFrame, train_split: pd.DataFrame, val_split: pd.DataFrame) -> pd.DataFrame:
    df = hashed_df.copy()
    train_files = set(train_split["file_name"].astype(str))
    val_files = set(val_split["file_name"].astype(str))
    split_bucket = []
    for _, row in df.iterrows():
        if row["split"] == "test":
            split_bucket.append("test")
        elif str(row["file_name"]) in train_files:
            split_bucket.append("generated_train")
        elif str(row["file_name"]) in val_files:
            split_bucket.append("generated_val")
        else:
            split_bucket.append("train_unassigned")
    df["generated_split"] = split_bucket

    rows = []
    for hash_col, duplicate_type in [("sha256", "exact_sha256"), ("ahash", "near_identical_ahash"), ("dhash", "near_identical_dhash")]:
        valid = df[df[hash_col].astype(str).str.len() > 0]
        for hash_value, group in valid.groupby(hash_col):
            if len(group) <= 1:
                continue
            splits = sorted(group["split"].unique())
            generated_splits = sorted(group["generated_split"].unique())
            rows.append(
                {
                    "duplicate_type": duplicate_type,
                    "hash": hash_value,
                    "count": len(group),
                    "splits": "|".join(splits),
                    "generated_splits": "|".join(generated_splits),
                    "file_names": "|".join(group["file_name"].astype(str).tolist()),
                    "paths": "|".join(group["resolved_path"].astype(str).tolist()),
                    "train_val_leakage_risk": {"generated_train", "generated_val"}.issubset(set(generated_splits)),
                    "train_test_leakage_risk": {"train", "test"}.issubset(set(splits)),
                }
            )
    return pd.DataFrame(rows)


def label_display(label: Any, label_names: dict[int, str] | None = None) -> str:
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        return f"label_{label}"
    if label_names and label_int in label_names:
        return f"label_{label_int} ({label_names[label_int]})"
    return f"label_{label_int}"


def confirmed_label_mapping(label_names: dict[int, str] | None) -> bool:
    return bool(label_names and 0 in label_names and 1 in label_names)


def save_label_distribution(train_df: pd.DataFrame, fig_path: Path, label_names: dict[int, str] | None = None) -> pd.DataFrame:
    counts = train_df["label"].value_counts().sort_index()
    ratios = counts / counts.sum()
    summary = pd.DataFrame(
        {
            "label": counts.index.astype(str),
            "class_name": [label_names.get(int(label), "") if label_names else "" for label in counts.index],
            "count": counts.values,
            "ratio": ratios.values,
        }
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([label_display(label, label_names) for label in counts.index], summary["count"], color=["#4c78a8", "#f58518"])
    ax.set_title("Train Label Distribution")
    ax.set_xlabel("Label" if confirmed_label_mapping(label_names) else "Label (mapping unconfirmed)")
    ax.set_ylabel("Image count")
    for i, row in summary.iterrows():
        ax.text(i, row["count"], f"{row['ratio']:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return summary


def readable_train_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    return metadata_df[(metadata_df["split"] == "train") & (metadata_df["readable"].fillna(False).astype(bool))].copy()


def sample_images_by_label(metadata_df: pd.DataFrame, fig_path: Path, seed: int, label_names: dict[int, str] | None = None) -> None:
    df = readable_train_metadata(metadata_df)
    if df.empty or "label" not in df:
        return
    labels = sorted(df["label"].dropna().unique())
    n = min(8, max(1, df.groupby("label").size().min()))
    fig, axes = plt.subplots(len(labels), n, figsize=(n * 1.8, len(labels) * 1.8))
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(seed)
    for r, label in enumerate(labels):
        subset = df[df["label"] == label]
        sample = subset.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        for c, (_, row) in enumerate(sample.iterrows()):
            with Image.open(row["resolved_path"]) as img:
                axes[r, c].imshow(img.convert("L"), cmap="gray")
            axes[r, c].set_title(label_display(label, label_names), fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def average_image_by_label(metadata_df: pd.DataFrame, fig_path: Path, seed: int, label_names: dict[int, str] | None = None, max_per_label: int = 500) -> None:
    df = readable_train_metadata(metadata_df)
    if df.empty:
        return
    labels = sorted(df["label"].dropna().unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        subset = df[df["label"] == label]
        if len(subset) > max_per_label:
            subset = subset.sample(n=max_per_label, random_state=seed)
        acc = np.zeros((224, 224), dtype=np.float64)
        used = 0
        for _, row in subset.iterrows():
            try:
                with Image.open(row["resolved_path"]) as img:
                    arr = np.asarray(img.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
                acc += arr
                used += 1
            except Exception:
                continue
        if used:
            ax.imshow(acc / used, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Average {label_display(label, label_names)} (n={used})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_numeric_by_label(metadata_df: pd.DataFrame, columns: list[str], fig_path: Path, title: str, label_names: dict[int, str] | None = None) -> None:
    df = readable_train_metadata(metadata_df)
    if df.empty:
        return
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    axes = np.atleast_1d(axes)
    labels = sorted(df["label"].dropna().unique())
    for ax, col in zip(axes, columns):
        data = [pd.to_numeric(df[df["label"] == label][col], errors="coerce").dropna() for label in labels]
        ax.boxplot(data, tick_labels=[label_display(label, label_names) for label in labels], showfliers=False)
        ax.set_title(col)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_size_distribution(metadata_df: pd.DataFrame, fig_path: Path, label_names: dict[int, str] | None = None) -> None:
    df = readable_train_metadata(metadata_df)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, group in df.groupby("label"):
        ax.scatter(group["width"], group["height"], s=10, alpha=0.35, label=label_display(label, label_names))
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_title("Image Size Distribution by Label")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_border_crops(metadata_df: pd.DataFrame, fig_path: Path, seed: int, label_names: dict[int, str] | None = None, max_per_label: int = 400) -> None:
    df = readable_train_metadata(metadata_df)
    if df.empty:
        return
    labels = sorted(df["label"].dropna().unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        subset = df[df["label"] == label]
        if len(subset) > max_per_label:
            subset = subset.sample(n=max_per_label, random_state=seed)
        acc = np.zeros((224, 224), dtype=np.float64)
        used = 0
        for _, row in subset.iterrows():
            try:
                with Image.open(row["resolved_path"]) as img:
                    arr = np.asarray(img.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
                mask = np.zeros_like(arr)
                b = 16
                mask[:b, :] = arr[:b, :]
                mask[-b:, :] = arr[-b:, :]
                mask[:, :b] = arr[:, :b]
                mask[:, -b:] = arr[:, -b:]
                acc += mask
                used += 1
            except Exception:
                continue
        if used:
            ax.imshow(acc / used, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Average border {label_display(label, label_names)}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def dataframe_counts_by_label(df: pd.DataFrame, label_names: dict[int, str] | None = None) -> str:
    if df.empty:
        return "none"
    counts = df["label"].value_counts().sort_index()
    return ", ".join([f"{label_display(idx, label_names)}: {count}" for idx, count in counts.items()])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n| --- |"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_report(
    paths: dict[str, Path],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    file_check: pd.DataFrame,
    metadata: pd.DataFrame,
    label_summary: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    split_info: dict[str, Any],
    image_dir: Path,
    label_names: dict[int, str] | None = None,
) -> None:
    train_check = file_check[file_check["split"] == "train"]
    test_check = file_check[file_check["split"] == "test"]
    missing = file_check[~file_check["exists"].fillna(False).astype(bool)]
    unreadable = file_check[file_check["exists"].fillna(False).astype(bool) & ~file_check["readable"].fillna(False).astype(bool)]
    image_verified = len(file_check) > 0 and missing.empty and unreadable.empty
    exact_dups = duplicate_df[duplicate_df["duplicate_type"] == "exact_sha256"] if not duplicate_df.empty else pd.DataFrame()
    phash_dups = duplicate_df[duplicate_df["duplicate_type"].str.contains("near", na=False)] if not duplicate_df.empty else pd.DataFrame()
    train_val_risk = bool((duplicate_df["train_val_leakage_risk"] == True).any()) if not duplicate_df.empty else False
    train_test_risk = bool((duplicate_df["train_test_leakage_risk"] == True).any()) if not duplicate_df.empty else False
    group_col = find_group_column(train_df.columns)
    inferred_groups = train_df["file_name"].map(infer_patient_group).dropna().nunique()
    mapping_confirmed = confirmed_label_mapping(label_names)
    if mapping_confirmed:
        label_mapping_status = f"CONFIRMED. label 0 = {label_names[0]}, label 1 = {label_names[1]}."
        label_mapping_action = "Use PNEUMONIA recall/sensitivity and NORMAL specificity in later evaluation."
        label_mapping_check_status = "PASS"
    else:
        label_mapping_status = "NOT CONFIRMED. Metrics and reports should use label_0 and label_1 until an official mapping is available."
        label_mapping_action = "Confirm class names from competition metadata before writing NORMAL/PNEUMONIA metrics."
        label_mapping_check_status = "WARN"

    ratios = {
        label_display(row.label, label_names): f"{row.ratio:.3f}"
        for row in label_summary.itertuples(index=False)
    }
    majority_ratio = label_summary["ratio"].max() if not label_summary.empty else np.nan
    imbalance_text = (
        "Class imbalance is meaningful; use class-aware metrics and consider class weights, WeightedRandomSampler, threshold tuning, and recall/sensitivity-focused evaluation."
        if pd.notna(majority_ratio) and majority_ratio >= 0.7
        else "Class imbalance is present but not extreme; still avoid accuracy-only evaluation."
    )

    risk_rows = [
        ("CSV files loaded", "PASS", f"train={train_df.shape}, test={test_df.shape}, sample_submission={sample_df.shape}", "Continue."),
        ("Image files mapped to CSV", "PASS" if image_verified else "WARN", f"missing={len(missing)}, unreadable={len(unreadable)}", "Fix missing/unreadable paths before training." if not image_verified else "Continue."),
        ("Patient-level split verified", "PASS" if split_info["patient_level_verified"] else "WARN", split_info.get("group_column") or "no patient identifier", split_info.get("limitation") or "Use group split."),
        ("Same patient across splits absent", "PASS" if split_info["patient_level_verified"] else "WARN", "Not independently verifiable without patient IDs.", "Treat split as temporary if patient IDs are unavailable."),
        ("Class imbalance checked", "PASS", f"counts={label_summary[['label','count']].to_dict('records')}", "Use non-accuracy metrics and imbalance-aware training options."),
        ("Duplicate images checked", "PASS" if not train_val_risk and not train_test_risk else "WARN", f"exact_groups={len(exact_dups)}", "Review duplicate_image_report.csv."),
        ("Near-duplicate images checked", "PASS" if len(phash_dups) == 0 else "WARN", f"perceptual_hash_groups={len(phash_dups)}", "Review perceptual hash groups before trusting validation metrics."),
        ("Border/text/device shortcut checked", "WARN", "Figures generated; manual radiographic artifact review is still required.", "Inspect random samples, average images, and border/corner plots."),
        ("Test set not used for validation", "PASS", "Only train.csv was split into train/val.", "Do not tune thresholds or early stopping on test.csv."),
        ("Label mapping confirmed", label_mapping_check_status, label_mapping_status, label_mapping_action),
    ]

    report = [
        "# Pre-baseline Dataset Audit Report",
        "",
        "This project is for research and education only. This audit does not establish clinical diagnostic validity.",
        "",
        "## 1. Dataset summary",
        "",
        f"- train rows: {len(train_df)}",
        f"- test rows: {len(test_df)}",
        f"- sample submission rows: {len(sample_df)}",
        f"- train columns: {', '.join(train_df.columns)}",
        f"- test columns: {', '.join(test_df.columns)}",
        f"- sample submission columns: {', '.join(sample_df.columns)}",
        f"- label mapping status: {label_mapping_status}",
        f"- image directory status: `{image_dir}` exists = {image_dir.exists()}",
        f"- file existence status: train missing={int((~train_check['exists']).sum())}, test missing={int((~test_check['exists']).sum())}",
        f"- unreadable/corrupted image status: {len(unreadable)} unreadable files",
        "",
        "## 2. Class distribution",
        "",
        markdown_table(label_summary),
        "",
        f"- ratios: {ratios}",
        f"- interpretation: {imbalance_text}",
        "- Later evaluation must include accuracy, sensitivity/recall, specificity, precision, F1-score, AUROC, PR-AUC, and confusion matrix.",
        "",
        "## 3. Patient-level leakage check",
        "",
        f"- patient_id availability: {'available in column ' + group_col if group_col else 'not available'}",
        f"- inferred grouping attempt: {inferred_groups} non-null inferred groups",
        f"- patient-level split possible: {split_info['patient_level_verified']}",
        f"- split decision: {split_info['method']}",
        f"- limitation: {split_info.get('limitation') or 'No patient-level limitation identified from CSV columns.'}",
        "",
        "## 4. Duplicate and near-duplicate check",
        "",
        f"- exact duplicate hash groups: {len(exact_dups)}",
        f"- perceptual hash duplicate groups: {len(phash_dups)}",
        f"- train/validation leakage risk: {train_val_risk}",
        f"- train/test leakage risk: {train_test_risk}",
        "- details saved to `reports/tables/duplicate_image_report.csv`.",
        "",
        "## 5. Shortcut-learning artifact check",
        "",
        "- Generated random sample grids, average images, average border/corner crops, image size plots, aspect ratio plots, brightness/contrast plots, and border intensity plots.",
        "- These figures provide evidence for possible border, cropping, brightness, and resolution shortcuts, but manual review is still required before training.",
        "- Shortcut probe classifier was not trained in this first audit deliverable, to avoid starting model training before review.",
        "",
        "## 6. Recommended preprocessing",
        "",
        "- Resize: start with 224x224.",
        "- Channel handling: if images are grayscale, keep one-channel for Custom CNN if desired; convert to 3-channel for ImageNet pretrained models.",
        "- Normalization: use ImageNet normalization for pretrained models first; consider dataset mean/std after reviewing split statistics.",
        "- Augmentation: avoid vertical flip. Consider mild rotation, small translation, small scaling, and brightness/contrast adjustment. Use horizontal flip only if accepted for this dataset.",
        "- If border/text/device artifacts appear label-associated, crop, mask, or use lung-region-focused preprocessing before model training.",
        "",
        "## 7. Baseline-ready split decision",
        "",
        f"- selected split method: {split_info['method']}",
        f"- train split count by label: {dataframe_counts_by_label(train_split, label_names)}",
        f"- validation split count by label: {dataframe_counts_by_label(val_split, label_names)}",
        f"- limitation: {split_info.get('limitation') or 'Patient-level grouping was available.'}",
        "- test.csv was not used for validation, early stopping, threshold tuning, or model selection.",
        "",
        "## 8. Risk checklist",
        "",
        "| Check item | Status | Evidence | Action |",
        "| --- | --- | --- | --- |",
    ]
    for item, status, evidence, action in risk_rows:
        report.append(f"| {item} | {status} | {str(evidence).replace('|', '/')} | {str(action).replace('|', '/')} |")
    report.append("")
    (paths["reports"] / "pre_baseline_audit.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    label_names = None
    if args.label_0_name and args.label_1_name:
        label_names = {0: args.label_0_name, 1: args.label_1_name}

    output_dir = Path(args.output_dir)
    paths = ensure_dirs(output_dir)
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    sample_csv = Path(args.sample_submission_csv)
    image_dir = Path(args.image_dir)

    print("[audit] loading CSV files")
    train_df = read_csv_safely(train_csv)
    test_df = read_csv_safely(test_csv)
    sample_df = read_csv_safely(sample_csv)

    print("[audit] indexing image files")
    index = build_file_index(image_dir)
    if not index:
        print("[audit] WARNING: no image files found; image-dependent checks will be partial")

    print("[audit] checking train image mapping and metadata")
    train_check, train_meta = audit_file_mapping(train_df, "train", image_dir, index)
    print("[audit] checking test image mapping and metadata")
    test_check, test_meta = audit_file_mapping(test_df, "test", image_dir, index)
    file_check = pd.concat([train_check, test_check], ignore_index=True)
    metadata = pd.concat([train_meta, test_meta], ignore_index=True)

    file_check.to_csv(paths["tables"] / "image_file_check.csv", index=False)
    metadata.to_csv(paths["tables"] / "image_metadata_by_file.csv", index=False)

    print("[audit] creating baseline split")
    train_split, val_split, split_info = create_baseline_split(train_df, seed=args.seed)
    train_split.to_csv(paths["outputs"] / "train_split.csv", index=False)
    val_split.to_csv(paths["outputs"] / "val_split.csv", index=False)

    print("[audit] saving label distribution")
    label_summary = save_label_distribution(train_df, paths["figures"] / "label_distribution.png", label_names)

    print("[audit] hashing readable images for duplicate checks")
    metadata_hashed = add_hashes(metadata)
    metadata_hashed.to_csv(paths["tables"] / "image_metadata_by_file.csv", index=False)
    dup_df = duplicate_report(metadata_hashed, train_split, val_split)
    if dup_df.empty:
        dup_df = pd.DataFrame(
            columns=[
                "duplicate_type",
                "hash",
                "count",
                "splits",
                "generated_splits",
                "file_names",
                "paths",
                "train_val_leakage_risk",
                "train_test_leakage_risk",
            ]
        )
    dup_df.to_csv(paths["tables"] / "duplicate_image_report.csv", index=False)

    print("[audit] creating shortcut-risk figures")
    sample_images_by_label(metadata_hashed, paths["figures"] / "random_samples_by_label.png", args.seed, label_names)
    average_image_by_label(metadata_hashed, paths["figures"] / "average_image_by_label.png", args.seed, label_names)
    plot_border_crops(metadata_hashed, paths["figures"] / "average_border_corner_crop_by_label.png", args.seed, label_names)
    plot_size_distribution(metadata_hashed, paths["figures"] / "image_size_distribution_by_label.png", label_names)
    plot_numeric_by_label(metadata_hashed, ["aspect_ratio"], paths["figures"] / "aspect_ratio_distribution_by_label.png", "Aspect Ratio by Label", label_names)
    plot_numeric_by_label(metadata_hashed, ["mean_intensity", "std_intensity"], paths["figures"] / "brightness_contrast_by_label.png", "Brightness and Contrast by Label", label_names)
    plot_numeric_by_label(metadata_hashed, ["border_mean", "corner_mean"], paths["figures"] / "border_intensity_by_label.png", "Border and Corner Intensity by Label", label_names)

    print("[audit] writing report")
    make_report(
        paths=paths,
        train_df=train_df,
        test_df=test_df,
        sample_df=sample_df,
        file_check=file_check,
        metadata=metadata_hashed,
        label_summary=label_summary,
        duplicate_df=dup_df,
        train_split=train_split,
        val_split=val_split,
        split_info=split_info,
        image_dir=image_dir,
        label_names=label_names,
    )
    print(f"[audit] done: {paths['reports'] / 'pre_baseline_audit.md'}")


if __name__ == "__main__":
    main()
