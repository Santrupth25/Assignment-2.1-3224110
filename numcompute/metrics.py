"""Batch and streaming metrics for binary classification and regression."""

from collections import deque
import numpy as np


def _validate_pair(y_true, y_pred):
    """Convert and validate matching, non-empty one-dimensional arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape")
    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty")

    return y_true, y_pred


def _validate_binary_labels(y_true, y_pred=None):
    """Check that classification labels contain only 0 and 1."""
    arrays = [np.asarray(y_true)]
    if y_pred is not None:
        arrays.append(np.asarray(y_pred))

    for values in arrays:
        if not np.all(np.isin(values, [0, 1])):
            raise ValueError("Binary classification labels must be 0 or 1")


def accuracy(y_true, y_pred):
    """Return the fraction of predictions matching the true labels."""
    y_true, y_pred = _validate_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """Return the binary confusion matrix [[TN, FP], [FN, TP]]."""
    y_true, y_pred = _validate_pair(y_true, y_pred)
    _validate_binary_labels(y_true, y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def precision(y_true, y_pred):
    """Return positive-class precision, or 0 if undefined."""
    matrix = confusion_matrix(y_true, y_pred)
    tp = matrix[1, 1]
    fp = matrix[0, 1]
    return float(tp / (tp + fp)) if tp + fp else 0.0


def recall(y_true, y_pred):
    """Return positive-class recall, or 0 if undefined."""
    matrix = confusion_matrix(y_true, y_pred)
    tp = matrix[1, 1]
    fn = matrix[1, 0]
    return float(tp / (tp + fn)) if tp + fn else 0.0


def f1(y_true, y_pred):
    """Return the positive-class F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if p + r else 0.0


def mse(y_true, y_pred):
    """Return mean squared error for matching numeric arrays."""
    y_true, y_pred = _validate_pair(y_true, y_pred)

    try:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("MSE inputs must be numeric") from exc

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("MSE inputs must contain only finite values")

    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true, y_score):
    """Return false-positive rates, true-positive rates, and thresholds.

    y_score contains predicted probabilities or confidence scores for class 1.
    Both classes must be present to calculate an ROC curve.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if y_true.shape != y_score.shape:
        raise ValueError("Inputs must have the same shape")
    if y_true.size == 0:
        raise ValueError("Inputs cannot be empty")

    _validate_binary_labels(y_true)

    if not np.all(np.isfinite(y_score)):
        raise ValueError("Scores must contain only finite values")

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        raise ValueError("ROC curve requires both binary classes")

    order = np.argsort(-y_score, kind="stable")
    sorted_scores = y_score[order]
    sorted_true = y_true[order] == 1

    # Keep the last position for every distinct threshold.
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, y_true.size - 1]

    tp = np.cumsum(sorted_true)[threshold_indices]
    fp = 1 + threshold_indices - tp

    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Calculate the area under an ROC curve using trapezoidal integration."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    if fpr.ndim != 1 or tpr.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if fpr.shape != tpr.shape or fpr.size < 2:
        raise ValueError("Inputs must have the same shape with at least two points")
    if not np.all(np.isfinite(fpr)) or not np.all(np.isfinite(tpr)):
        raise ValueError("Inputs must contain only finite values")
    if np.any(np.diff(fpr) < 0):
        raise ValueError("FPR values must be non-decreasing")
    if np.any((fpr < 0) | (fpr > 1) | (tpr < 0) | (tpr > 1)):
        raise ValueError("FPR and TPR values must be between 0 and 1")

    return float(np.trapz(tpr, fpr))


class StreamingMetrics:
    """Accumulate binary classification metrics across data chunks.

    Set window_size to calculate metrics over the most recent samples too.
    Exact cumulative AUC is calculated from all supplied scores. If scores
    are omitted in update(), predicted labels are used as scores.
    """

    def __init__(self, window_size=None):
        if window_size is not None:
            if (
                not isinstance(window_size, (int, np.integer))
                or isinstance(window_size, (bool, np.bool_))
                or window_size <= 0
            ):
                raise ValueError("window_size must be a positive integer or None")

        self.window_size = window_size
        self.reset()

    def reset(self):
        """Clear accumulated metrics while retaining the window setting."""
        self.n_samples = 0
        self._confusion = np.zeros((2, 2), dtype=np.int64)
        self._true_chunks = []
        self._score_chunks = []

        self._window_true = deque(maxlen=self.window_size)
        self._window_pred = deque(maxlen=self.window_size)
        self._window_score = deque(maxlen=self.window_size)

        return self

    def update(self, y_true_chunk, y_pred_chunk, y_score=None):
        """Add one chunk of labels, predictions, and optional class-1 scores."""
        y_true, y_pred = _validate_pair(y_true_chunk, y_pred_chunk)
        _validate_binary_labels(y_true, y_pred)

        if y_score is None:
            scores = y_pred.astype(float)
        else:
            scores = np.asarray(y_score, dtype=float)

            if scores.ndim != 1 or scores.shape != y_true.shape:
                raise ValueError("y_score must be 1D and match y_true")
            if not np.all(np.isfinite(scores)):
                raise ValueError("y_score must contain only finite values")
            if np.any((scores < 0) | (scores > 1)):
                raise ValueError("y_score values must be between 0 and 1")

        self._confusion += confusion_matrix(y_true, y_pred)
        self.n_samples += y_true.size

        self._true_chunks.append(y_true.astype(np.int8, copy=True))
        self._score_chunks.append(scores.astype(float, copy=True))

        if self.window_size is not None:
            self._window_true.extend(y_true.tolist())
            self._window_pred.extend(y_pred.tolist())
            self._window_score.extend(scores.tolist())

        return self

    @staticmethod
    def _scores_from_confusion(matrix):
        """Calculate accuracy, precision, recall, and F1 from a matrix."""
        tn, fp = matrix[0]
        fn, tp = matrix[1]

        total = tn + fp + fn + tp
        acc = (tn + tp) / total if total else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1_score = (
            2 * prec * rec / (prec + rec) if prec + rec else 0.0
        )

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_score),
            "confusion_matrix": matrix.copy(),
        }

    @staticmethod
    def _calculate_auc(y_true, y_score):
        """Return NaN if AUC is undefined because a class is missing."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            return auc(fpr, tpr)
        except ValueError:
            return float("nan")

    def result(self):
        """Return cumulative metrics and optional rolling-window metrics."""
        if self.n_samples == 0:
            raise ValueError("No data has been received")

        result = self._scores_from_confusion(self._confusion)

        all_true = np.concatenate(self._true_chunks)
        all_scores = np.concatenate(self._score_chunks)

        result["auc"] = self._calculate_auc(all_true, all_scores)
        result["n_samples"] = self.n_samples

        if self.window_size is not None and self._window_true:
            window_true = np.asarray(self._window_true, dtype=np.int8)
            window_pred = np.asarray(self._window_pred, dtype=np.int8)
            window_score = np.asarray(self._window_score, dtype=float)

            rolling = self._scores_from_confusion(
                confusion_matrix(window_true, window_pred)
            )
            rolling["auc"] = self._calculate_auc(window_true, window_score)
            rolling["n_samples"] = len(window_true)

            result["rolling"] = rolling
        else:
            result["rolling"] = None

        return result


# Alias for singular naming compatibility.
StreamingMetric = StreamingMetrics
