"""Global pixel metrics and unchanged source-selected F1 thresholds."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


METRICS = ("pixel_auroc", "pixel_ap", "strict_f1", "strict_iou", "image_auroc", "image_ap")


def select_threshold(masks, scores):
    truth = (np.asarray(masks).reshape(-1) > 0).astype(np.uint8)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if np.unique(truth).size < 2:
        raise ValueError("Source validation must contain positive and negative pixels")
    precision, recall, thresholds = precision_recall_curve(truth, values)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = min(int(np.nanargmax(f1)), len(thresholds) - 1)
    return float(thresholds[best])


def pixel_metrics(masks, scores, threshold=None):
    truth = (np.asarray(masks).reshape(-1) > 0).astype(np.uint8)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    result = {"pixel_auroc": 100 * float(roc_auc_score(truth, values)),
              "pixel_ap": 100 * float(average_precision_score(truth, values))}
    if threshold is not None:
        prediction = values >= threshold
        tp = int(((prediction == 1) & (truth == 1)).sum())
        fp = int(((prediction == 1) & (truth == 0)).sum())
        fn = int(((prediction == 0) & (truth == 1)).sum())
        result.update(strict_f1=100 * 2 * tp / max(2 * tp + fp + fn, 1),
                      strict_iou=100 * tp / max(tp + fp + fn, 1), source_threshold=float(threshold))
    return result


def all_metrics(masks, scores, ratio, threshold):
    result = pixel_metrics(masks, scores, threshold)
    flat = scores.reshape(len(scores), -1)
    k = max(1, int(flat.shape[1] * ratio))
    image_scores = np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)
    labels = masks.reshape(len(masks), -1).any(axis=1).astype(np.uint8)
    if np.unique(labels).size == 2:
        result.update(image_auroc=100 * float(roc_auc_score(labels, image_scores)),
                      image_ap=100 * float(average_precision_score(labels, image_scores)))
    else:
        result.update(image_auroc=None, image_ap=None)
    result["image_topk_ratio"] = ratio
    return result
