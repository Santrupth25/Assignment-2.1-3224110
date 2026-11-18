import numpy as np
import pytest

from numcompute.metrics import auc, roc_curve, StreamingMetrics


def test_auc_perfect_curve():
    fpr = np.array([0.0, 0.0, 1.0])
    tpr = np.array([0.0, 1.0, 1.0])

    assert auc(fpr, tpr) == pytest.approx(1.0)


def test_auc_random_classifier():
    fpr = np.array([0.0, 1.0])
    tpr = np.array([0.0, 1.0])

    assert auc(fpr, tpr) == pytest.approx(0.5)


def test_auc_zero_area():
    fpr = np.array([0.0, 1.0])
    tpr = np.array([0.0, 0.0])

    assert auc(fpr, tpr) == pytest.approx(0.0)


def test_auc_invalid_shapes():
    with pytest.raises(ValueError):
        auc(np.array([0.0, 1.0]), np.array([0.0]))


def test_auc_decreasing_fpr():
    fpr = np.array([0.0, 0.8, 0.5])
    tpr = np.array([0.0, 0.7, 1.0])

    with pytest.raises(ValueError):
        auc(fpr, tpr)


def test_roc_curve_perfect_scores():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    assert fpr[0] == pytest.approx(0.0)
    assert tpr[0] == pytest.approx(0.0)
    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)
    assert len(thresholds) == len(fpr)


def test_roc_curve_requires_both_classes():
    y_true = np.array([1, 1, 1])
    scores = np.array([0.2, 0.5, 0.9])

    with pytest.raises(ValueError):
        roc_curve(y_true, scores)


def test_streaming_metrics_auc():
    metrics = StreamingMetrics()

    metrics.update(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.2, 0.8, 0.9]),
    )

    result = metrics.result()

    assert result["auc"] == pytest.approx(1.0)


def test_streaming_metrics_auc_across_chunks():
    metrics = StreamingMetrics()

    metrics.update(
        np.array([0, 0]),
        np.array([0, 0]),
        y_score=np.array([0.1, 0.2]),
    )

    metrics.update(
        np.array([1, 1]),
        np.array([1, 1]),
        y_score=np.array([0.8, 0.9]),
    )

    assert metrics.result()["auc"] == pytest.approx(1.0)


def test_streaming_metrics_auc_without_both_classes():
    metrics = StreamingMetrics()

    metrics.update(
        np.array([1, 1]),
        np.array([1, 1]),
        y_score=np.array([0.6, 0.9]),
    )

    assert np.isnan(metrics.result()["auc"])