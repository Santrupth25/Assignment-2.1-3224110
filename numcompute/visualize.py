
"""Plot streaming metrics and model predictions using Matplotlib."""

import numpy as np
import matplotlib.pyplot as plt


def _as_plot_values(values, name):
    """Convert metric values to a 1D float array, allowing missing values."""
    try:
        if np.isscalar(values):
            values = [values]
        cleaned = [np.nan if value is None else float(value) for value in values]
        result = np.asarray(cleaned, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric values or None"
        ) from exc

    if result.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if result.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return result


def _finish_plot(fig, save_path=None, show=True):
    """Optionally save or display a figure."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metric_over_time(
    metric_values,
    title="Metric over time",
    ylabel="Metric",
    save_path=None,
    show=True,
    xlabel="Chunk",
    ax=None,
):
    """Plot one metric across streaming chunks.

    None values appear as gaps. Returns the Matplotlib figure.
    """
    values = _as_plot_values(metric_values, "metric_values")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    chunks = np.arange(1, values.size + 1)
    ax.plot(chunks, values, marker="o", label=ylabel)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _finish_plot(fig, save_path=save_path, show=show)
    return fig


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title="Model comparison",
    ylabel="Metric",
    save_path=None,
    show=True,
    xlabel="Chunk",
    ax=None,
):
    """Compare two models using metric values over streaming chunks.

    The metric sequences may have different lengths. Returns the figure.
    """
    first = _as_plot_values(metric1, "metric1")
    second = _as_plot_values(metric2, "metric2")

    if not isinstance(labels, (tuple, list)) or len(labels) != 2:
        raise ValueError("labels must contain exactly two labels")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(
        np.arange(1, first.size + 1),
        first,
        marker="o",
        label=str(labels[0]),
    )
    ax.plot(
        np.arange(1, second.size + 1),
        second,
        marker="s",
        label=str(labels[1]),
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _finish_plot(fig, save_path=save_path, show=show)
    return fig


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    save_path=None,
    show=True,
    title="Predictions vs ground truth",
    xlabel="Sample",
    ylabel="Class / value",
    ax=None,
):
    """Plot actual and predicted values for one data chunk."""
    true_values = np.asarray(y_true)
    pred_values = np.asarray(y_pred)

    if true_values.ndim != 1 or pred_values.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
    if true_values.size == 0:
        raise ValueError("Prediction arrays cannot be empty")
    if true_values.shape != pred_values.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    try:
        true_values = true_values.astype(float)
        pred_values = pred_values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Predictions and ground truth must be numeric"
        ) from exc

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    samples = np.arange(1, true_values.size + 1)
    ax.plot(
        samples, true_values, marker="o",
        linestyle="-", label="Ground truth"
    )
    ax.plot(
        samples, pred_values, marker="x",
        linestyle="--", label="Predictions"
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _finish_plot(fig, save_path=save_path, show=show)
    return fig
