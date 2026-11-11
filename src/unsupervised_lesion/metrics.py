from __future__ import annotations

import numpy as np


def confusion(prediction: np.ndarray, target: np.ndarray,
              valid: np.ndarray | None = None) -> dict:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target dimensions differ")
    valid = np.ones_like(target) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != target.shape:
        raise ValueError("Valid mask dimensions differ")
    p, t = prediction[valid], target[valid]
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    tn = int(np.count_nonzero(~p & ~t))
    fn = int(np.count_nonzero(~p & t))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def from_counts(counts: dict) -> dict:
    tp, fp, tn, fn = (int(counts[key]) for key in ("tp", "fp", "tn", "fn"))
    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    dice = ratio(2 * tp, 2 * tp + fp + fn)
    iou = ratio(tp, tp + fp + fn)
    accuracy = ratio(tp + tn, tp + fp + tn + fn)
    balanced = ((recall + specificity) / 2
                if recall is not None and specificity is not None else None)
    return {**counts, "dice": dice, "f1": dice, "iou": iou,
            "precision": precision, "recall": recall, "sensitivity": recall,
            "specificity": specificity, "accuracy": accuracy,
            "balanced_accuracy": balanced}


def binary_metrics(prediction: np.ndarray, target: np.ndarray,
                   valid: np.ndarray | None = None) -> dict:
    return from_counts(confusion(prediction, target, valid))


def pooled(rows: list[dict]) -> dict:
    counts = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "tn", "fn")}
    return from_counts(counts)

