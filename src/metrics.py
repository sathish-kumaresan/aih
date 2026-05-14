import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _seven_metrics(y_true, y_pred, score) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, score) if len(set(y_true)) > 1 else float("nan"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def image_metrics(y_true, margin, proba) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= 0.5).astype(int)
    return _seven_metrics(y_true, y_pred, np.asarray(margin))


def patient_metrics(df_pred: pd.DataFrame) -> dict:
    per_patient = df_pred.groupby("patient_id").agg(
        label=("label", "first"),
        proba=("proba", "mean"),
    ).reset_index()
    y_true = per_patient["label"].to_numpy()
    score = per_patient["proba"].to_numpy()
    y_pred = (score >= 0.5).astype(int)
    return _seven_metrics(y_true, y_pred, score)


def summarize(metric_dicts: list[dict], leak_count: int | None = None) -> pd.DataFrame:
    keys = [k for k in metric_dicts[0] if k != "confusion_matrix"]
    rows = []
    for k in keys:
        vals = [d[k] for d in metric_dicts]
        row = {
            "metric": k,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
        }
        for i, v in enumerate(vals, start=1):
            row[f"seed_{i}"] = v
        if leak_count is not None:
            row["leak_count"] = leak_count
        rows.append(row)
    return pd.DataFrame(rows)
