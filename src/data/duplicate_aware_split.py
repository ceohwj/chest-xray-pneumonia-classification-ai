"""Create a duplicate-aware train/validation split.

The duplicate report produced by ``audit.py`` is treated as graph evidence:
files listed together in each duplicate/near-duplicate row are connected, and
connected components are kept entirely in either train or validation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create duplicate-aware train/validation split.")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--duplicate_report", default="reports/tables/duplicate_image_report.csv")
    parser.add_argument("--output_train_csv", default="outputs/train_split.csv")
    parser.add_argument("--output_val_csv", default="outputs/val_split.csv")
    parser.add_argument("--report_path", default="reports/duplicate_aware_split_report.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    return parser.parse_args()


def split_file_names(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def build_duplicate_groups(train_df: pd.DataFrame, duplicate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_files = set(train_df["file_name"].astype(str))
    uf = UnionFind(train_files)
    edge_rows = []

    for _, row in duplicate_df.iterrows():
        names = [name for name in split_file_names(row["file_names"]) if name in train_files]
        if len(names) < 2:
            continue
        first = names[0]
        for other in names[1:]:
            uf.union(first, other)
        edge_rows.append(
            {
                "duplicate_type": row["duplicate_type"],
                "hash": row["hash"],
                "train_file_count": len(names),
                "train_files": "|".join(names),
            }
        )

    roots = {file_name: uf.find(file_name) for file_name in train_files}
    sorted_roots = {root: f"dup_group_{i:05d}" for i, root in enumerate(sorted(set(roots.values())), start=1)}
    grouped = train_df.copy()
    grouped["duplicate_group_id"] = grouped["file_name"].astype(str).map(lambda name: sorted_roots[roots[name]])
    grouped["class_name"] = grouped["label"].map(LABEL_NAMES)
    evidence = pd.DataFrame(edge_rows)
    return grouped, evidence


def choose_group_split(grouped_df: pd.DataFrame, seed: int, n_splits: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = grouped_df["label"].to_numpy()
    groups = grouped_df["duplicate_group_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total = len(grouped_df)
    global_ratio = grouped_df["label"].value_counts(normalize=True).sort_index()

    candidates = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(grouped_df, y, groups)):
        train_split = grouped_df.iloc[train_idx].copy()
        val_split = grouped_df.iloc[val_idx].copy()
        val_ratio = val_split["label"].value_counts(normalize=True).reindex(global_ratio.index, fill_value=0).sort_index()
        ratio_error = float((val_ratio - global_ratio).abs().sum())
        size_error = abs((len(val_split) / total) - (1 / n_splits))
        candidates.append(
            {
                "fold_idx": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "score": ratio_error + size_error,
                "ratio_error": ratio_error,
                "size_error": size_error,
            }
        )

    best = min(candidates, key=lambda item: (item["score"], item["ratio_error"], item["size_error"]))
    train_split = grouped_df.iloc[best["train_idx"]].copy().reset_index(drop=True)
    val_split = grouped_df.iloc[best["val_idx"]].copy().reset_index(drop=True)
    info = {key: value for key, value in best.items() if key not in {"train_idx", "val_idx"}}
    return train_split, val_split, info


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    counts = df["label"].value_counts().sort_index()
    for label, name in LABEL_NAMES.items():
        count = int(counts.get(label, 0))
        rows.append({"label": label, "class_name": name, "count": count, "percent": count / total * 100 if total else 0.0})
    return pd.DataFrame(rows)


def verify_split(train_df: pd.DataFrame, val_df: pd.DataFrame, original_df: pd.DataFrame) -> dict:
    train_files = set(train_df["file_name"])
    val_files = set(val_df["file_name"])
    train_groups = set(train_df["duplicate_group_id"])
    val_groups = set(val_df["duplicate_group_id"])
    assigned = Counter(list(train_df["file_name"]) + list(val_df["file_name"]))
    original_files = set(original_df["file_name"].astype(str))
    return {
        "duplicate_group_overlap_count": len(train_groups & val_groups),
        "file_overlap_count": len(train_files & val_files),
        "assigned_file_count": len(assigned),
        "original_train_file_count": len(original_files),
        "missing_from_split_count": len(original_files - set(assigned)),
        "extra_in_split_count": len(set(assigned) - original_files),
        "duplicate_assignment_count": sum(1 for count in assigned.values() if count != 1),
        "passes": len(train_groups & val_groups) == 0
        and len(train_files & val_files) == 0
        and len(original_files - set(assigned)) == 0
        and len(set(assigned) - original_files) == 0
        and all(count == 1 for count in assigned.values()),
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n| --- |"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in df.columns) + " |")
    return "\n".join(lines)


def train_test_risk_summary(duplicate_df: pd.DataFrame) -> pd.DataFrame:
    if "train_test_leakage_risk" not in duplicate_df:
        return pd.DataFrame()
    return duplicate_df[duplicate_df["train_test_leakage_risk"] == True][
        ["duplicate_type", "count", "splits", "generated_splits", "file_names"]
    ].copy()


def write_report(
    report_path: Path,
    duplicate_df: pd.DataFrame,
    duplicate_evidence: pd.DataFrame,
    grouped_df: pd.DataFrame,
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    split_info: dict,
    verification: dict,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    train_summary = class_summary(train_split)
    val_summary = class_summary(val_split)
    group_sizes = grouped_df.groupby("duplicate_group_id").size()
    duplicate_group_count = int((group_sizes > 1).sum())
    singleton_group_count = int((group_sizes == 1).sum())
    train_test_risk = train_test_risk_summary(duplicate_df)

    lines = [
        "# Duplicate-aware Split Report",
        "",
        "This split is for research and education only. It is not evidence of clinical diagnostic validity.",
        "",
        "## Duplicate evidence",
        "",
        "- Source: `reports/tables/duplicate_image_report.csv` generated by the pre-baseline audit.",
        "- Exact duplicates used: `exact_sha256` rows.",
        "- Near-duplicates used: `near_identical_ahash` and `near_identical_dhash` rows.",
        "- Each duplicate report row was treated as graph connectivity evidence among the listed training images.",
        "- No additional duplicate criteria were introduced.",
        "",
        "## Duplicate group assignment",
        "",
        f"- Original training images: {len(grouped_df)}",
        f"- Duplicate report rows: {len(duplicate_df)}",
        f"- Duplicate evidence rows involving at least two train images: {len(duplicate_evidence)}",
        f"- Total duplicate-aware groups: {grouped_df['duplicate_group_id'].nunique()}",
        f"- Non-singleton duplicate groups: {duplicate_group_count}",
        f"- Singleton groups: {singleton_group_count}",
        "- Connected components were built with union-find. A duplicate group is never intentionally split across train and validation.",
        "",
        "## Final split",
        "",
        f"- Split method: StratifiedGroupKFold candidate selection with seed 42 and no group overlap.",
        f"- Selected fold index: {split_info['fold_idx']}",
        f"- Train images: {len(train_split)} ({len(train_split) / len(grouped_df) * 100:.2f}%)",
        f"- Validation images: {len(val_split)} ({len(val_split) / len(grouped_df) * 100:.2f}%)",
        f"- Class-ratio score: {split_info['score']:.6f}",
        "",
        "### Train class counts",
        "",
        markdown_table(train_summary),
        "",
        "### Validation class counts",
        "",
        markdown_table(val_summary),
        "",
        "## Stratification comparison",
        "",
        "- Temporary split before duplicate grouping: train 4,172 / validation 1,044.",
        "- Duplicate-aware split preserves leakage prevention over exact 80/20 class matching.",
        f"- Duplicate-aware train NORMAL/PNEUMONIA: {int(train_summary.loc[train_summary['label'] == 0, 'count'].iloc[0])} / {int(train_summary.loc[train_summary['label'] == 1, 'count'].iloc[0])}",
        f"- Duplicate-aware validation NORMAL/PNEUMONIA: {int(val_summary.loc[val_summary['label'] == 0, 'count'].iloc[0])} / {int(val_summary.loc[val_summary['label'] == 1, 'count'].iloc[0])}",
        "",
        "## Verification",
        "",
        f"- No duplicate group crosses train/validation: {verification['duplicate_group_overlap_count'] == 0}",
        f"- Duplicate group overlap count: {verification['duplicate_group_overlap_count']}",
        f"- File overlap count: {verification['file_overlap_count']}",
        f"- All original train images assigned exactly once: {verification['passes']}",
        f"- Missing from split count: {verification['missing_from_split_count']}",
        f"- Extra in split count: {verification['extra_in_split_count']}",
        f"- Duplicate assignment count: {verification['duplicate_assignment_count']}",
        "",
        "## Remaining limitations",
        "",
        "- Patient-level identifiers are not available.",
        "- Patient-level leakage cannot be fully excluded.",
        "- Train/test near-duplicate risk is reported separately below and the test set was not modified.",
        "- Validation metrics still require cautious interpretation because near-duplicate detection can be conservative and patient identity is unknown.",
    ]

    if train_test_risk.empty:
        lines.extend(["", "## Train/test near-duplicate risk", "", "- No train/test near-duplicate risk rows were flagged in the duplicate report."])
    else:
        lines.extend(
            [
                "",
                "## Train/test near-duplicate risk",
                "",
                f"- Flagged rows: {len(train_test_risk)}",
                markdown_table(train_test_risk),
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    train_csv = Path(args.train_csv)
    duplicate_report = Path(args.duplicate_report)
    output_train_csv = Path(args.output_train_csv)
    output_val_csv = Path(args.output_val_csv)
    report_path = Path(args.report_path)
    output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    output_val_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[split] loading train CSV and duplicate report")
    train_df = pd.read_csv(train_csv)
    duplicate_df = pd.read_csv(duplicate_report)

    print("[split] building duplicate graph components")
    grouped_df, duplicate_evidence = build_duplicate_groups(train_df, duplicate_df)

    print("[split] selecting group-safe stratified fold")
    train_split, val_split, split_info = choose_group_split(grouped_df, seed=args.seed, n_splits=args.n_splits)

    print("[split] verifying split")
    verification = verify_split(train_split, val_split, train_df)
    if not verification["passes"]:
        raise RuntimeError(f"Duplicate-aware split verification failed: {verification}")

    train_split.to_csv(output_train_csv, index=False)
    val_split.to_csv(output_val_csv, index=False)

    print("[split] writing report")
    write_report(
        report_path=report_path,
        duplicate_df=duplicate_df,
        duplicate_evidence=duplicate_evidence,
        grouped_df=grouped_df,
        train_split=train_split,
        val_split=val_split,
        split_info=split_info,
        verification=verification,
    )

    print(f"[split] done: {output_train_csv}, {output_val_csv}, {report_path}")


if __name__ == "__main__":
    main()
