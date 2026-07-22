"""Train/validation splitting utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


PATIENT_COLUMN_CANDIDATES = (
    "patient_id",
    "patient",
    "study_id",
    "subject_id",
    "case_id",
    "person_id",
    "record_id",
)


def find_group_column(columns: Iterable[str]) -> str | None:
    normalized = {col.lower(): col for col in columns}
    for candidate in PATIENT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    for col in columns:
        lower = col.lower()
        if any(token in lower for token in ("patient", "study", "subject", "case")):
            return col
    return None


def infer_patient_group(file_name: str) -> str | None:
    """Best-effort patient grouping from non-sequential file names."""
    stem = Path(str(file_name)).stem
    patterns = [
        r"(patient[_-]?\d+)",
        r"(subject[_-]?\d+)",
        r"(study[_-]?\d+)",
        r"(case[_-]?\d+)",
        r"(person[_-]?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()

    if re.fullmatch(r"(train|test)[_-]?\d+", stem, flags=re.IGNORECASE):
        return None
    return None


def create_baseline_split(
    train_df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create a patient-level split when possible, otherwise stratified split."""
    df = train_df.copy()
    group_col = find_group_column(df.columns)
    split_info = {
        "method": "stratified",
        "group_column": None,
        "patient_level_verified": False,
        "limitation": "Patient-level split cannot be verified from the provided CSV and image file names.",
    }

    if group_col is None:
        inferred = df["file_name"].map(infer_patient_group)
        if inferred.notna().any() and inferred.nunique() > 1:
            df["_patient_group"] = inferred.fillna(df["file_name"].astype(str))
            group_col = "_patient_group"
            split_info.update(
                {
                    "method": "group",
                    "group_column": "inferred_from_file_name",
                    "patient_level_verified": False,
                    "limitation": "Patient grouping was inferred from file names and is not independently verified.",
                }
            )

    if group_col is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, df["label"], groups=df[group_col]))
        train_split = df.iloc[train_idx].drop(columns=["_patient_group"], errors="ignore")
        val_split = df.iloc[val_idx].drop(columns=["_patient_group"], errors="ignore")
        split_info["method"] = "group"
        split_info["group_column"] = group_col
        if group_col != "_patient_group":
            split_info["patient_level_verified"] = True
            split_info["limitation"] = ""
        return train_split.reset_index(drop=True), val_split.reset_index(drop=True), split_info

    train_split, val_split = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True), split_info
