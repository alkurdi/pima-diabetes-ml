"""Evaluate a saved pipeline within a tracked run directory."""

from __future__ import annotations

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

from experiment_tracking import (
    append_summary_row,
    now_iso,
    resolve_eval_run_paths,
    save_json,
    save_single_row_csv,
)
from preprocessing import make_train_test_split, split_features_target

DEFAULT_SEED = 42
DEFAULT_DATA_PATH = "data/diabetes.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def log_message(message: str, log_file) -> None:
    print(message)
    log_file.write(f"{message}\n")
    log_file.flush()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def get_positive_scores(model, X_test: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return positive-class scores usable for thresholding/ROC.

    Priority:
    1) predict_proba positive class
    2) decision_function transformed to [0, 1] by sigmoid
    3) fallback: predict labels (0/1) as scores
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
        return np.asarray(scores, dtype=float), "predict_proba"

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test)
        if raw_scores.ndim > 1:
            raw_scores = raw_scores[:, 0]
        scores = _sigmoid(np.asarray(raw_scores, dtype=float))
        return scores, "decision_function_sigmoid"

    labels = model.predict(X_test)
    return np.asarray(labels, dtype=float), "predict_fallback"


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, path) -> None:
    df = pd.DataFrame(
        {
            "row_index": np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_roc_artifacts(y_true: np.ndarray, y_proba: np.ndarray, csv_path, png_path) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_value = float(roc_auc_score(y_true, y_proba))

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds}).to_csv(
        csv_path, index=False
    )

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    return auc_value


def save_confusion_matrix_artifacts(cm: np.ndarray, csv_path, png_path) -> None:
    cm_df = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(csv_path)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def main() -> None:
    args = parse_args()
    run_paths = resolve_eval_run_paths(args.run_id)

    if not run_paths.run_config_json.exists():
        raise FileNotFoundError(f"Missing run config: {run_paths.run_config_json}")
    run_config = json.loads(run_paths.run_config_json.read_text())

    seed = int(args.seed if args.seed is not None else run_config.get("seed", DEFAULT_SEED))
    data_path = args.data_path or run_config.get("data_path", DEFAULT_DATA_PATH)
    test_size = args.test_size if args.test_size is not None else float(
        run_config.get("test_size", 0.2)
    )

    set_global_seed(seed)

    with run_paths.eval_log.open("w", encoding="utf-8") as eval_log:
        log_message(f"Starting evaluation for run_id={run_paths.run_id}", eval_log)
        log_message(f"Using model={run_paths.model_path}", eval_log)
        log_message(f"Using threshold={args.threshold}", eval_log)

        if not run_paths.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {run_paths.model_path}")

        model = pickle.loads(run_paths.model_path.read_bytes())
        data = pd.read_csv(data_path)
        X, y = split_features_target(data)

        _, X_test, _, y_test = make_train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
        )

        y_proba, score_source = get_positive_scores(model, X_test)
        y_pred = (y_proba >= args.threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        roc_auc = save_roc_artifacts(y_test.to_numpy(), y_proba, run_paths.roc_curve_csv, run_paths.roc_curve_png)
        save_predictions(y_test.to_numpy(), y_pred, y_proba, run_paths.predictions_csv)
        save_confusion_matrix_artifacts(cm, run_paths.confusion_matrix_csv, run_paths.confusion_matrix_png)

        metrics = {
            "stage": "evaluate_holdout",
            "run_id": run_paths.run_id,
            "model_name": str(run_config.get("model_name", "unknown")),
            "seed": seed,
            "threshold": float(args.threshold),
            "score_source": score_source,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc),
            "confusion_matrix_tn": int(cm[0, 0]),
            "confusion_matrix_fp": int(cm[0, 1]),
            "confusion_matrix_fn": int(cm[1, 0]),
            "confusion_matrix_tp": int(cm[1, 1]),
            "predictions_csv": str(run_paths.predictions_csv),
            "roc_curve_csv": str(run_paths.roc_curve_csv),
            "roc_curve_png": str(run_paths.roc_curve_png),
            "confusion_matrix_csv": str(run_paths.confusion_matrix_csv),
            "confusion_matrix_png": str(run_paths.confusion_matrix_png),
            "timestamp": now_iso(),
        }

        eval_config = {
            "run_id": run_paths.run_id,
            "seed": seed,
            "data_path": str(data_path),
            "test_size": float(test_size),
            "threshold": float(args.threshold),
            "score_source": score_source,
            "timestamp": now_iso(),
        }

        save_json(metrics, run_paths.metrics_json)
        save_single_row_csv(metrics, run_paths.metrics_csv)
        save_json(eval_config, run_paths.eval_config_json)

        append_summary_row(
            {
                "run_id": metrics["run_id"],
                "model_name": metrics["model_name"],
                "seed": metrics["seed"],
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "predictions_csv": metrics["predictions_csv"],
                "roc_curve_png": metrics["roc_curve_png"],
                "timestamp": metrics["timestamp"],
            }
        )

        log_message(f"Accuracy: {metrics['accuracy']:.4f}", eval_log)
        log_message(f"F1: {metrics['f1']:.4f}", eval_log)
        log_message(f"ROC-AUC: {metrics['roc_auc']:.4f}", eval_log)
        log_message(f"Score source: {score_source}", eval_log)
        log_message(f"Confusion Matrix:\n{cm}", eval_log)
        log_message(f"Saved predictions to: {run_paths.predictions_csv}", eval_log)
        log_message(f"Saved ROC curve to: {run_paths.roc_curve_csv} and {run_paths.roc_curve_png}", eval_log)
        log_message(
            f"Saved confusion matrix to: {run_paths.confusion_matrix_csv} and {run_paths.confusion_matrix_png}",
            eval_log,
        )
        log_message(f"Saved metrics JSON to: {run_paths.metrics_json}", eval_log)
        log_message(f"Saved metrics CSV to: {run_paths.metrics_csv}", eval_log)
        log_message(f"Saved eval config to: {run_paths.eval_config_json}", eval_log)
        log_message("Updated results/summary.csv", eval_log)
        print(f"RUN_DIR: {run_paths.run_dir}")


if __name__ == "__main__":
    main()
