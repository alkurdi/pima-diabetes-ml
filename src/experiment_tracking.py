"""Helpers for experiment run directory management and artifact tracking."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RUNS_ROOT = Path("results/runs")
SUMMARY_PATH = Path("results/summary.csv")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    model_path: Path
    metrics_json: Path
    metrics_csv: Path
    run_config_json: Path
    eval_config_json: Path
    predictions_csv: Path
    roc_curve_csv: Path
    roc_curve_png: Path
    confusion_matrix_csv: Path
    confusion_matrix_png: Path
    train_log: Path
    eval_log: Path


def generate_run_id() -> str:
    """Create a timestamp-based run id, e.g. 20260221_210530."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_paths(run_id: str, runs_root: Path = RUNS_ROOT) -> RunPaths:
    """Return canonical artifact paths for a run id."""
    run_dir = runs_root / run_id
    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        model_path=run_dir / "model.pkl",
        metrics_json=run_dir / "metrics.json",
        metrics_csv=run_dir / "metrics.csv",
        run_config_json=run_dir / "run_config.json",
        eval_config_json=run_dir / "eval_config.json",
        predictions_csv=run_dir / "predictions.csv",
        roc_curve_csv=run_dir / "roc_curve.csv",
        roc_curve_png=run_dir / "roc_curve.png",
        confusion_matrix_csv=run_dir / "confusion_matrix.csv",
        confusion_matrix_png=run_dir / "confusion_matrix.png",
        train_log=run_dir / "train_log.txt",
        eval_log=run_dir / "eval_log.txt",
    )


def ensure_run_dir(paths: RunPaths) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)


def resolve_train_run_paths(run_id: str | None) -> RunPaths:
    resolved_run_id = run_id or generate_run_id()
    paths = build_run_paths(resolved_run_id)
    ensure_run_dir(paths)
    return paths


def resolve_eval_run_paths(run_id: str | None) -> RunPaths:
    """Resolve run paths for evaluation.

    If run_id is omitted, evaluate the latest run directory for compatibility.
    """
    if run_id:
        paths = build_run_paths(run_id)
        ensure_run_dir(paths)
        return paths

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError("No runs found in results/runs. Run train.py first.")
    return build_run_paths(run_dirs[-1].name)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def save_single_row_csv(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def append_summary_row(row: dict, summary_path: Path = SUMMARY_PATH) -> None:
    """Append one evaluation row to results/summary.csv."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    fieldnames = [
        "run_id",
        "model_name",
        "seed",
        "threshold",
        "accuracy",
        "f1",
        "roc_auc",
        "predictions_csv",
        "roc_curve_png",
        "timestamp",
    ]
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
