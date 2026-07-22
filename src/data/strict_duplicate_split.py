"""Create a stricter duplicate split by breaking oversized perceptual clusters.

The original duplicate_group_id is preserved. This script adds
strict_duplicate_group_id to reduce validation domination by very large
aHash/dHash connected components while still keeping exact duplicate evidence
together.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Create strict duplicate-group train/validation split.")
    parser.add_argument("--current_train_csv", default="outputs/train_split.csv")
    parser.add_argument("--current_val_csv", default="outputs/val_split.csv")
    parser.add_argument("--duplicate_report", default="reports/tables/duplicate_image_report.csv")
    parser.add_argument("--output_train_csv", default="outputs/train_split_strict.csv")
    parser.add_argument("--output_val_csv", default="outputs/val_split_strict.csv")
    parser.add_argument("--report_path", default="reports/strict_duplicate_split_report.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--large_group_threshold", type=int, default=50)
    parser.add_argument("--audit_group_min_size", type=int, default=21)
    parser.add_argument("--audit_group_max_size", type=int, default=50)
    parser.add_argument("--split_audit_groups", action="store_true")
    return parser.parse_args()


def split_file_names(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


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


def load_current_split(train_csv: Path, val_csv: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_df["previous_split"] = "train"
    val_df["previous_split"] = "val"
    combined = pd.concat([train_df, val_df], ignore_index=True)
    required = {"file_name", "label", "duplicate_group_id"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Current split files are missing required columns: {sorted(missing)}")
    if "class_name" not in combined.columns:
        combined["class_name"] = combined["label"].map(LABEL_NAMES)
    return combined


def build_exact_duplicate_ids(all_df: pd.DataFrame, duplicate_report: pd.DataFrame) -> dict[str, str | None]:
    file_names = set(all_df["file_name"].astype(str))
    uf = UnionFind(file_names)
    exact_rows = duplicate_report[duplicate_report["duplicate_type"] == "exact_sha256"].copy()

    for _, row in exact_rows.iterrows():
        names = [name for name in split_file_names(row["file_names"]) if name in file_names]
        if len(names) < 2:
            continue
        first = names[0]
        for other in names[1:]:
            uf.union(first, other)

    roots = {name: uf.find(name) for name in file_names}
    counts = Counter(roots.values())
    duplicate_roots = sorted(root for root, count in counts.items() if count > 1)
    root_to_id = {root: f"strict_exact_{idx:05d}" for idx, root in enumerate(duplicate_roots, start=1)}
    return {name: root_to_id.get(roots[name]) for name in file_names}


def add_strict_groups(
    all_df: pd.DataFrame,
    exact_ids: dict[str, str | None],
    large_group_threshold: int,
    audit_group_min_size: int,
    audit_group_max_size: int,
    split_audit_groups: bool,
) -> pd.DataFrame:
    out = all_df.copy()
    group_sizes = out.groupby("duplicate_group_id")["file_name"].transform("size")
    out["duplicate_group_size"] = group_sizes.astype(int)
    out["strict_group_policy"] = "keep"
    out.loc[
        out["duplicate_group_size"].between(audit_group_min_size, audit_group_max_size, inclusive="both"),
        "strict_group_policy",
    ] = "audit_keep"
    out.loc[out["duplicate_group_size"] > large_group_threshold, "strict_group_policy"] = "split_large"
    if split_audit_groups:
        out.loc[out["strict_group_policy"] == "audit_keep", "strict_group_policy"] = "split_audit"

    strict_ids = []
    for _, row in out.iterrows():
        file_name = str(row["file_name"])
        policy = str(row["strict_group_policy"])
        original_group = str(row["duplicate_group_id"])
        exact_id = exact_ids.get(file_name)
        if policy in {"split_large", "split_audit"}:
            strict_ids.append(exact_id if exact_id is not None else f"strict_file_{Path(file_name).stem}")
        else:
            strict_ids.append(f"strict_{original_group}")
    out["strict_duplicate_group_id"] = strict_ids
    out["strict_audit_flag"] = out["duplicate_group_size"].between(audit_group_min_size, audit_group_max_size, inclusive="both")
    return out


def choose_group_split(grouped_df: pd.DataFrame, seed: int, n_splits: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = grouped_df["label"].to_numpy(dtype=int)
    groups = grouped_df["strict_duplicate_group_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total = len(grouped_df)
    global_ratio = grouped_df["label"].value_counts(normalize=True).sort_index()

    candidates = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(grouped_df, y, groups)):
        val_split = grouped_df.iloc[val_idx].copy()
        val_ratio = val_split["label"].value_counts(normalize=True).reindex(global_ratio.index, fill_value=0).sort_index()
        ratio_error = float((val_ratio - global_ratio).abs().sum())
        size_error = abs((len(val_split) / total) - (1 / n_splits))
        largest_val_group = int(val_split.groupby("strict_duplicate_group_id").size().max())
        candidates.append(
            {
                "fold_idx": fold_idx,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "score": ratio_error + size_error + largest_val_group / total * 0.01,
                "ratio_error": ratio_error,
                "size_error": size_error,
                "largest_val_strict_group": largest_val_group,
            }
        )

    best = min(candidates, key=lambda item: (item["score"], item["ratio_error"], item["size_error"]))
    train_split = grouped_df.iloc[best["train_idx"]].copy().reset_index(drop=True)
    val_split = grouped_df.iloc[best["val_idx"]].copy().reset_index(drop=True)
    info = {key: value for key, value in best.items() if key not in {"train_idx", "val_idx"}}
    return train_split, val_split, info


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    counts = df["label"].value_counts().sort_index()
    rows = []
    for label, name in LABEL_NAMES.items():
        count = int(counts.get(label, 0))
        rows.append({"label": label, "class_name": name, "count": count, "percent": count / total * 100 if total else 0.0})
    return pd.DataFrame(rows)


def split_verification(train_df: pd.DataFrame, val_df: pd.DataFrame, all_df: pd.DataFrame) -> dict:
    train_files = set(train_df["file_name"].astype(str))
    val_files = set(val_df["file_name"].astype(str))
    train_groups = set(train_df["strict_duplicate_group_id"].astype(str))
    val_groups = set(val_df["strict_duplicate_group_id"].astype(str))
    assigned = Counter(list(train_files) + list(val_files))
    all_files = set(all_df["file_name"].astype(str))
    return {
        "file_overlap_count": len(train_files & val_files),
        "strict_duplicate_group_overlap_count": len(train_groups & val_groups),
        "assigned_file_count": len(assigned),
        "original_file_count": len(all_files),
        "missing_from_split_count": len(all_files - set(assigned)),
        "extra_in_split_count": len(set(assigned) - all_files),
        "duplicate_assignment_count": sum(1 for count in assigned.values() if count != 1),
        "passes": len(train_files & val_files) == 0
        and len(train_groups & val_groups) == 0
        and len(all_files - set(assigned)) == 0
        and len(set(assigned) - all_files) == 0
        and all(count == 1 for count in assigned.values()),
    }


def mixed_group_count(df: pd.DataFrame, group_col: str) -> int:
    label_counts = df.groupby(group_col)["label"].nunique()
    return int((label_counts > 1).sum())


def top_group_sizes(df: pd.DataFrame, group_col: str, top_n: int = 20) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_col)
    for group_id, part in grouped:
        counts = part["label"].value_counts().to_dict()
        rows.append(
            {
                group_col: group_id,
                "size": len(part),
                "normal": int(counts.get(0, 0)),
                "pneumonia": int(counts.get(1, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["size", group_col], ascending=[False, True]).head(top_n).reset_index(drop=True)


def write_report(
    report_path: Path,
    all_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    split_info: dict,
    verification: dict,
    args: argparse.Namespace,
) -> None:
    old_train = all_df[all_df["previous_split"] == "train"]
    old_val = all_df[all_df["previous_split"] == "val"]
    old_val_top = top_group_sizes(old_val, "duplicate_group_id", top_n=10)
    strict_top = top_group_sizes(strict_df, "strict_duplicate_group_id", top_n=20)

    old_summary = pd.DataFrame(
        [
            {"split": "old_train", **{f"label_{label}": int(count) for label, count in old_train["label"].value_counts().sort_index().items()}, "rows": len(old_train)},
            {"split": "old_val", **{f"label_{label}": int(count) for label, count in old_val["label"].value_counts().sort_index().items()}, "rows": len(old_val)},
            {"split": "strict_train", **{f"label_{label}": int(count) for label, count in train_split["label"].value_counts().sort_index().items()}, "rows": len(train_split)},
            {"split": "strict_val", **{f"label_{label}": int(count) for label, count in val_split["label"].value_counts().sort_index().items()}, "rows": len(val_split)},
        ]
    ).fillna(0)

    policy_counts = strict_df["strict_group_policy"].value_counts().rename_axis("policy").reset_index(name="image_count")
    strict_group_counts = strict_df.drop_duplicates("strict_duplicate_group_id")["strict_group_policy"].value_counts().rename_axis("policy").reset_index(name="strict_group_count")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Strict Duplicate Split Report",
        "",
        "This split is for research and education only. It is not evidence of clinical diagnostic validity.",
        "",
        "## Goal",
        "",
        "- Preserve leakage prevention for exact duplicates.",
        "- Reduce validation/OOF calibration distortion from oversized aHash/dHash connected components.",
        "- Keep the original `duplicate_group_id` column unchanged and add `strict_duplicate_group_id`.",
        "",
        "## Strict group policy",
        "",
        f"- `group_size <= {args.audit_group_min_size - 1}`: keep original group.",
        f"- `{args.audit_group_min_size} <= group_size <= {args.audit_group_max_size}`: audit flag, default keep.",
        f"- `group_size > {args.large_group_threshold}`: split to file-level groups unless exact duplicate evidence links files.",
        f"- Split audit groups option enabled: {args.split_audit_groups}",
        "",
        "### Policy image counts",
        "",
        markdown_table(policy_counts),
        "",
        "### Policy strict group counts",
        "",
        markdown_table(strict_group_counts),
        "",
        "## Split comparison",
        "",
        markdown_table(old_summary),
        "",
        f"- Original groups: {strict_df['duplicate_group_id'].nunique()}",
        f"- Strict groups: {strict_df['strict_duplicate_group_id'].nunique()}",
        f"- Original mixed-label groups: {mixed_group_count(strict_df, 'duplicate_group_id')}",
        f"- Strict mixed-label groups: {mixed_group_count(strict_df, 'strict_duplicate_group_id')}",
        f"- Selected strict fold index: {split_info['fold_idx']}",
        f"- Strict split score: {split_info['score']:.6f}",
        f"- Largest validation strict group: {split_info['largest_val_strict_group']}",
        "",
        "## Verification",
        "",
        f"- Train/val file overlap count: {verification['file_overlap_count']}",
        f"- Strict duplicate group train/val overlap count: {verification['strict_duplicate_group_overlap_count']}",
        f"- All files assigned exactly once: {verification['passes']}",
        f"- Missing from split count: {verification['missing_from_split_count']}",
        f"- Extra in split count: {verification['extra_in_split_count']}",
        f"- Duplicate assignment count: {verification['duplicate_assignment_count']}",
        "",
        "## Class summaries",
        "",
        "### Strict train",
        "",
        markdown_table(class_summary(train_split)),
        "",
        "### Strict validation",
        "",
        markdown_table(class_summary(val_split)),
        "",
        "## Original validation top duplicate groups",
        "",
        markdown_table(old_val_top),
        "",
        "## Strict group size top 20",
        "",
        markdown_table(strict_top),
        "",
        "## Notes",
        "",
        "- This is a split improvement experiment, not a final claim about model quality.",
        "- Patient-level identifiers are still unavailable, so patient-level leakage cannot be fully excluded.",
        "- Large perceptual clusters are split because connected-component chaining can merge visually similar but non-identical studies.",
        "- Exact duplicate evidence remains grouped under `strict_duplicate_group_id`.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    current_train_csv = Path(args.current_train_csv)
    current_val_csv = Path(args.current_val_csv)
    duplicate_report = Path(args.duplicate_report)
    output_train_csv = Path(args.output_train_csv)
    output_val_csv = Path(args.output_val_csv)
    report_path = Path(args.report_path)
    output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    output_val_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[strict split] loading current split and duplicate report")
    all_df = load_current_split(current_train_csv, current_val_csv)
    duplicate_df = pd.read_csv(duplicate_report)

    print("[strict split] building exact duplicate components")
    exact_ids = build_exact_duplicate_ids(all_df, duplicate_df)

    print("[strict split] creating strict duplicate groups")
    strict_df = add_strict_groups(
        all_df,
        exact_ids=exact_ids,
        large_group_threshold=args.large_group_threshold,
        audit_group_min_size=args.audit_group_min_size,
        audit_group_max_size=args.audit_group_max_size,
        split_audit_groups=args.split_audit_groups,
    )

    print("[strict split] selecting group-safe strict fold")
    train_split, val_split, split_info = choose_group_split(strict_df, seed=args.seed, n_splits=args.n_splits)

    print("[strict split] verifying strict split")
    verification = split_verification(train_split, val_split, strict_df)
    if not verification["passes"]:
        raise RuntimeError(f"Strict split verification failed: {verification}")

    print("[strict split] writing split CSVs")
    train_split.drop(columns=["previous_split"]).to_csv(output_train_csv, index=False)
    val_split.drop(columns=["previous_split"]).to_csv(output_val_csv, index=False)

    print("[strict split] writing report")
    write_report(
        report_path=report_path,
        all_df=all_df,
        strict_df=strict_df,
        train_split=train_split,
        val_split=val_split,
        split_info=split_info,
        verification=verification,
        args=args,
    )
    print(f"[strict split] done: {output_train_csv}, {output_val_csv}, {report_path}")


if __name__ == "__main__":
    main()
