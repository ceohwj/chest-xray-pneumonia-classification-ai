"""Run the reusable transfer-ensemble trainer with Kaggle-friendly defaults.

This wrapper keeps the main implementation in scripts/training/train_transfer_ensemble.py
and supplies Kaggle-friendly defaults:
- input data from /kaggle/input/<dataset>/
- outputs/checkpoints/submissions under /kaggle/working/

Attach a Kaggle dataset containing:
- data/images/
- data/test.csv
- data/sample_submission.csv
- outputs/train_split.csv
- outputs/val_split.csv

Also attach or upload this repository/code so that scripts/training/train_transfer_ensemble.py
is available.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_DATA_FILES = (
    "outputs/train_split.csv",
    "outputs/val_split.csv",
    "data/test.csv",
    "data/sample_submission.csv",
    "data/images",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transfer ensemble training on Kaggle.")
    parser.add_argument("--data-root", default=None, help="Kaggle dataset root. Auto-detected when omitted.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Repo/code root containing scripts/training/train_transfer_ensemble.py.",
    )
    parser.add_argument("--models", nargs="+", default=["densenet121", "convnext_tiny", "efficientnet_b3"])
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def has_required_data(root: Path) -> bool:
    return all((root / relative_path).exists() for relative_path in REQUIRED_DATA_FILES)


def discover_data_root(explicit_root: str | None) -> Path:
    if explicit_root is not None:
        root = Path(explicit_root)
        if has_required_data(root):
            return root
        raise FileNotFoundError(f"--data-root does not contain the required dataset files: {root}")

    input_root = Path("/kaggle/input")
    candidates: list[Path] = []
    if input_root.exists():
        candidates.extend(sorted(path for path in input_root.iterdir() if path.is_dir()))
        for path in sorted(input_root.iterdir()):
            if path.is_dir():
                candidates.extend(sorted(child for child in path.iterdir() if child.is_dir()))

    for candidate in candidates:
        if has_required_data(candidate):
            return candidate

    searched = ", ".join(str(path) for path in candidates[:20])
    raise FileNotFoundError(f"Could not auto-detect Kaggle data root. Searched: {searched}")


def find_training_script(explicit_project_root: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_project_root is not None:
        candidates.append(Path(explicit_project_root))

    current = Path(__file__).resolve()
    candidates.extend(current.parents)
    candidates.extend([Path.cwd(), Path("/kaggle/working")])

    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(sorted(path for path in input_root.iterdir() if path.is_dir()))
        for path in sorted(input_root.iterdir()):
            if path.is_dir():
                candidates.extend(sorted(child for child in path.iterdir() if child.is_dir()))

    seen: set[Path] = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        script = root / "scripts" / "training" / "train_transfer_ensemble.py"
        if script.exists():
            return script

    raise FileNotFoundError(
        "Could not find scripts/training/train_transfer_ensemble.py in attached Kaggle code inputs."
    )


def main() -> None:
    args = parse_args()
    data_root = discover_data_root(args.data_root)
    train_script = find_training_script(args.project_root)
    working_root = Path("/kaggle/working")

    output_dir = working_root / "transfer_ensemble"
    checkpoint_dir = working_root / "checkpoints" / "transfer_ensemble"
    submission_dir = working_root / "submissions"
    metrics_dir = working_root / "metrics"

    command = [
        sys.executable,
        str(train_script),
        "--train-csv",
        str(data_root / "outputs" / "train_split.csv"),
        "--val-csv",
        str(data_root / "outputs" / "val_split.csv"),
        "--test-csv",
        str(data_root / "data" / "test.csv"),
        "--sample-submission",
        str(data_root / "data" / "sample_submission.csv"),
        "--image-dir",
        str(data_root / "data" / "images"),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--submission-dir",
        str(submission_dir),
        "--metrics-dir",
        str(metrics_dir),
        "--image-size",
        str(args.image_size),
        "--epochs",
        str(args.epochs),
        "--num-folds",
        str(args.num_folds),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--models",
        *args.models,
    ]
    if args.debug:
        command.append("--debug")
    if not args.pretrained:
        command.append("--no-pretrained")

    print(f"Data root: {data_root}")
    print(f"Training script: {train_script}")
    print("Running:")
    print(" ".join(command))

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
