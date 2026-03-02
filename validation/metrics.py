"""
Detection quality metrics for SMLM cluster analysis.

Cluster detection is a binary classification problem at the localization level:
each observed localization is either in a real cluster (positive) or not
(negative / background). Ground-truth labels come from simulations where we
know which molecules were clustered.

Metrics
-------
precision : TP / (TP + FP)
    What fraction of detected cluster members are genuine?

recall (sensitivity) : TP / (TP + FN)
    What fraction of genuine cluster members were found?

F1 : harmonic mean of precision and recall
    Balanced summary when precision and recall trade off.

Jaccard index : TP / (TP + FP + FN)
    Overlap between predicted and true cluster sets. Useful when
    cluster boundaries matter more than within-cluster accuracy.

For the multiscale test, "detection" is binary per dataset (p < alpha), so
we also track per-dataset sensitivity and specificity across a set of
synthetic experiments.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClusterMetrics:
    """
    Binary classification metrics for cluster detection.

    All attributes are scalars in [0, 1].
    """
    precision: float
    recall: float
    f1: float
    jaccard: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def specificity(self) -> float:
        """True negative rate."""
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """(sensitivity + specificity) / 2."""
        return (self.recall + self.specificity) / 2.0

    def __repr__(self) -> str:
        return (
            f"ClusterMetrics("
            f"precision={self.precision:.3f}, "
            f"recall={self.recall:.3f}, "
            f"F1={self.f1:.3f}, "
            f"Jaccard={self.jaccard:.3f}, "
            f"TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn})"
        )


def evaluate_detection(pred_labels: np.ndarray,
                       true_labels: np.ndarray,
                       positive_label: int = 1) -> ClusterMetrics:
    """
    Compute detection metrics from per-localization labels.

    Parameters
    ----------
    pred_labels : ndarray, shape (N,)
        Predicted labels. Values equal to positive_label = in cluster.
    true_labels : ndarray, shape (N,)
        Ground-truth labels.
    positive_label : int
        The label value that indicates "in cluster". Default 1.

    Returns
    -------
    ClusterMetrics
    """
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)

    if pred_labels.shape != true_labels.shape:
        raise ValueError("pred_labels and true_labels must have the same shape")

    pred_pos = (pred_labels == positive_label)
    true_pos = (true_labels == positive_label)

    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    tn = int((~pred_pos & ~true_pos).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return ClusterMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn
    )


def summarize_detection_experiments(results: list) -> dict:
    """
    Summarize detection results across multiple synthetic experiments.

    Parameters
    ----------
    results : list of dict
        Each dict must have keys 'detected' (bool) and 'has_clusters' (bool).

    Returns
    -------
    dict with sensitivity, specificity, fpr (false positive rate), accuracy,
    and raw counts.
    """
    n_positive = sum(r['has_clusters'] for r in results)
    n_negative = sum(not r['has_clusters'] for r in results)

    tp = sum(r['detected'] and r['has_clusters'] for r in results)
    fp = sum(r['detected'] and not r['has_clusters'] for r in results)
    fn = sum(not r['detected'] and r['has_clusters'] for r in results)
    tn = sum(not r['detected'] and not r['has_clusters'] for r in results)

    sensitivity = tp / n_positive if n_positive > 0 else 0.0
    specificity = tn / n_negative if n_negative > 0 else 0.0
    fpr = fp / n_negative if n_negative > 0 else 0.0
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0.0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'n_positive': n_positive,
        'n_negative': n_negative
    }
