"""Dataset-weighted predictive metrics, OOD detection, and selective risk."""
import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from config import P
from losses import prediction, objective


def calibration(probabilities, labels):
    q = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    confidence = q.max(1)
    correct = (q.argmax(1) == y).astype(np.float64)
    # Clip only the integer bin index at confidence=1; probabilities unchanged.
    bins = np.minimum((confidence * P.ece_bins).astype(int), P.ece_bins - 1)
    count = np.bincount(bins, minlength=P.ece_bins)
    accuracy = np.full(P.ece_bins, np.nan)
    mean_confidence = np.full(P.ece_bins, np.nan)
    for b in range(P.ece_bins):
        if count[b]:
            accuracy[b] = correct[bins == b].mean()
            mean_confidence[b] = confidence[bins == b].mean()
    valid = count > 0
    ece = np.sum(count[valid] * np.abs(accuracy[valid] - mean_confidence[valid])) / len(y)
    return {"ece": float(ece), "count": count, "accuracy": accuracy,
            "confidence": mean_confidence, "edges": np.linspace(0, 1, P.ece_bins + 1)}


def classification_metrics(probabilities, labels, log_probabilities=None):
    q = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if q.ndim != 2 or q.shape != (len(y), P.classes) or len(y) == 0:
        raise ValueError("Expected nonempty N x 10 probabilities and N labels")
    if not (np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all()
            and np.allclose(q.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Invalid probabilities")
    if (y < 0).any() or (y >= P.classes).any():
        raise ValueError("Invalid classification labels")
    if log_probabilities is None:
        with np.errstate(divide="ignore"):
            log_probabilities = np.log(q)
    logq = np.asarray(log_probabilities, dtype=np.float64)
    if logq.shape != q.shape:
        raise ValueError("Log probability shape mismatch")
    true_logq = logq[np.arange(len(y)), y]
    if not np.isfinite(true_logq).all():
        raise FloatingPointError("Nonfinite predictive NLL; no clipping applied")
    correct = int((q.argmax(1) == y).sum())
    one_hot = np.eye(P.classes)[y]
    return {"correct": correct, "examples": len(y), "accuracy": correct / len(y),
            "nll": float(-true_logq.mean()),
            "brier": float(np.square(q - one_hot).sum(1).mean()),
            "ece": calibration(q, y)["ece"]}


def risk_coverage(probabilities, labels, uncertainty, indices=None):
    q, y = np.asarray(probabilities), np.asarray(labels)
    scores = np.asarray(uncertainty, dtype=np.float64)
    n = len(y)
    indices = np.arange(n) if indices is None else np.asarray(indices)
    if n == 0 or len(scores) != n or len(indices) != n or not np.isfinite(scores).all():
        raise ValueError("Invalid risk-coverage inputs")
    if len(np.unique(indices)) != n:
        raise ValueError("Test indices must be unique for deterministic tie breaks")
    order = np.lexsort((indices, scores))
    errors = (q.argmax(1) != y).astype(np.float64)[order]
    retained = np.arange(1, n + 1)
    risk = errors.cumsum() / retained
    metrics = {"aurc": float(risk.mean())}
    for c in P.coverage_levels:
        metrics[f"risk_{int(c * 100)}"] = float(risk[math.ceil(c * n) - 1])
    return {"coverage": retained / n, "risk": risk, "order": order, "metrics": metrics}


def ood_metrics(id_scores, ood_scores):
    id_scores, ood_scores = np.asarray(id_scores), np.asarray(ood_scores)
    if not len(id_scores) or not len(ood_scores):
        raise ValueError("Need both ID and OOD scores")
    scores = np.concatenate([id_scores, ood_scores])
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite OOD scores")
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    return {"auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}


@torch.no_grad()
def collect_predictions(model, method, batches, *, labelled=True, epoch=None):
    """For OOD inputs labelled=False discards garment targets entirely.

Validation may request an epoch-specific objective; aggregation weights each
batch by its size. Test and OOD evaluation never request training objectives.
"""
    model.eval()
    probabilities, log_probabilities, labels, uncertainty = [], [], [], []
    component_sums, total = {}, 0
    for images, targets in batches:
        logits = model(images)
        q, logq, u = prediction(logits, method)
        probabilities.append(q.numpy())
        log_probabilities.append(logq.numpy())
        if labelled:
            labels.append(targets.numpy())
        if u is not None:
            uncertainty.append(u.numpy())
        size = len(images)
        if epoch is not None:
            if not labelled:
                raise ValueError("Do not compute classification objectives on OOD inputs")
            value, components = objective(logits, targets, method, epoch)
            for name, tensor in {"objective": value, **components}.items():
                component_sums[name] = component_sums.get(name, 0.0) + tensor.item() * size
        total += size
    if total == 0:
        raise ValueError("Empty evaluation dataset")
    q = np.concatenate(probabilities)
    record = {"index": np.arange(total), "probabilities": q,
              "log_probabilities": np.concatenate(log_probabilities),
              "primary_uncertainty": 1.0 - q.max(1)}
    if labelled:
        record["labels"] = np.concatenate(labels)
    if uncertainty:
        record["evidential_uncertainty"] = np.concatenate(uncertainty)
    return record, {name: value / total for name, value in component_sums.items()}


def evaluate_id(record):
    metrics = classification_metrics(record["probabilities"], record["labels"],
                                     record["log_probabilities"])
    curve = risk_coverage(record["probabilities"], record["labels"],
                          record["primary_uncertainty"], record["index"])
    return metrics, curve
