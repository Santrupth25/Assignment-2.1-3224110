import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numcompute_stream.visualize import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)


def test_plot_metric_over_time():
    values = [0.6, 0.75, 0.9]

    result = plot_metric_over_time(
        values,
        "Accuracy over time",
        "Accuracy",
    )

    assert result is not None
    plt.close("all")


def test_plot_metric_over_time_with_empty_values():
    with pytest.raises((ValueError, TypeError)):
        plot_metric_over_time([], "Accuracy", "Accuracy")


def test_compare_models():
    model1 = [0.7, 0.8, 0.9]
    model2 = [0.6, 0.75, 0.85]
    labels = ["Model A", "Model B"]

    result = compare_models(model1, model2, labels)

    assert result is not None
    plt.close("all")


def test_plot_predictions_vs_ground_truth():
    actual = np.array([0, 1, 0, 1])
    predicted = np.array([0, 1, 1, 1])

    result = plot_predictions_vs_ground_truth(actual, predicted)

    assert result is not None
    plt.close("all")


def test_plot_predictions_with_perfect_predictions():
    actual = np.array([0, 1, 0, 1])

    result = plot_predictions_vs_ground_truth(actual, actual)

    assert result is not None
    plt.close("all")


def test_plot_predictions_with_different_shapes():
    actual = np.array([0, 1, 0])
    predicted = np.array([0, 1])

    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(actual, predicted)


def test_plot_predictions_with_valid_binary_labels():
    actual = np.array([0, 1, 0, 1])
    predicted = np.array([0, 1, 1, 1])

    result = plot_predictions_vs_ground_truth(actual, predicted)

    assert result is not None