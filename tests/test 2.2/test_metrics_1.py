
import numpy as np
import pytest

from numcompute_stream.metrics import StreamingMetrics


def test_metrics_initialisation():
    metrics = StreamingMetrics()
    assert metrics.n_samples == 0


def test_metrics_single_chunk():
    metrics = StreamingMetrics()

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics.update(y_true, y_pred)
    result = metrics.result()

    assert result["n_samples"] == 4
    assert result["accuracy"] == pytest.approx(0.75)


def test_metrics_multiple_chunks():
    metrics = StreamingMetrics()

    metrics.update(np.array([0, 1]), np.array([0, 1]))
    metrics.update(np.array([0, 1]), np.array([1, 1]))

    result = metrics.result()

    assert result["n_samples"] == 4
    assert result["accuracy"] == pytest.approx(0.75)


def test_metrics_precision_recall_f1():
    metrics = StreamingMetrics()

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics.update(y_true, y_pred)
    result = metrics.result()

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)


def test_metrics_confusion_matrix():
    metrics = StreamingMetrics()

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    metrics.update(y_true, y_pred)
    result = metrics.result()

    expected = np.array([[1, 1], [1, 1]])
    assert np.array_equal(result["confusion_matrix"], expected)


def test_metrics_auc():
    metrics = StreamingMetrics()

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    metrics.update(y_true, y_pred, y_score)
    result = metrics.result()

    assert result["auc"] == pytest.approx(1.0)


def test_metrics_reset():
    metrics = StreamingMetrics()
    metrics.update(np.array([0, 1]), np.array([0, 1]))

    metrics.reset()

    assert metrics.n_samples == 0

    with pytest.raises(ValueError):
        metrics.result()


def test_metrics_rejects_mismatched_shapes():
    metrics = StreamingMetrics()

    with pytest.raises(ValueError):
        metrics.update(
            np.array([0, 1, 1]),
            np.array([0, 1])
        )


def test_metrics_rolling_window():
    metrics = StreamingMetrics(window_size=2)

    metrics.update(np.array([0, 1]), np.array([0, 1]))
    metrics.update(np.array([0, 1]), np.array([1, 1]))

    result = metrics.result()

    assert result["n_samples"] == 4
    assert result["rolling"] is not None
    assert result["rolling"]["accuracy"] == pytest.approx(0.5)
