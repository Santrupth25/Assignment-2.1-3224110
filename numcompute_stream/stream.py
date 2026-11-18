
"""Chunk-based training, evaluation, timing, and memory logging."""

import time
import tracemalloc
import numpy as np

from .metrics import StreamingMetrics, accuracy


class StreamTrainer:
    """Manage test-then-train evaluation for a streaming pipeline.

    Call score_chunk(X, y) before fit_chunk(X, y) for each incoming chunk.
    Results are stored as dictionaries in history.
    """

    def __init__(self, pipeline, window_size=None):
        """Create a trainer for a pipeline and optional rolling window."""
        if pipeline is None:
            raise ValueError("pipeline cannot be None")
        if not hasattr(pipeline, "predict"):
            raise TypeError("pipeline must provide predict()")

        self.pipeline = pipeline
        self.metrics = StreamingMetrics(window_size=window_size)
        self.history = []
        self._awaiting_fit_index = None
        self._trained_by_trainer = False

    @property
    def model(self):
        """Return the final estimator in the pipeline."""
        steps = getattr(self.pipeline, "steps", None)
        if not steps:
            return getattr(self.pipeline, "model", None)
        return steps[-1][1]

    def _is_trained(self):
        """Check whether the estimator is ready to make predictions."""
        model = self.model
        if model is None:
            return False

        if hasattr(model, "root"):
            return model.root is not None
        if hasattr(model, "tree_"):
            return model.tree_ is not None
        if hasattr(model, "is_fitted"):
            return bool(model.is_fitted)
        if hasattr(model, "classes_"):
            return model.classes_ is not None

        return self._trained_by_trainer

    @staticmethod
    def _validate_chunk(X, y):
        """Check that X and y have matching sample counts."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] == 0:
            raise ValueError("Chunk cannot be empty")
        if X.shape[0] != y.size:
            raise ValueError("X and y must contain the same number of samples")

        return X, y

    def _empty_record(self, n_samples):
        """Create a history record with unfilled evaluation fields."""
        return {
            "chunk": len(self.history) + 1,
            "n_samples": int(n_samples),
            "accuracy": None,
            "cumulative_accuracy": None,
            "cumulative_precision": None,
            "cumulative_recall": None,
            "cumulative_f1": None,
            "cumulative_auc": None,
            "cumulative_confusion_matrix": None,
            "rolling_accuracy": None,
            "rolling_precision": None,
            "rolling_recall": None,
            "rolling_f1": None,
            "rolling_auc": None,
            "training_time": None,
            "peak_memory_mb": None,
        }

    def score_chunk(self, X, y):
        """Evaluate a chunk before training on it and log its metrics."""
        X, y_true = self._validate_chunk(X, y)
        result = self._empty_record(len(y_true))

        # A buffered tree cannot make predictions before its first retrain.
        if not self._is_trained():
            self.history.append(result)
            self._awaiting_fit_index = len(self.history) - 1
            return result

        y_pred = np.asarray(self.pipeline.predict(X))
        if y_pred.ndim != 1 or y_pred.shape != y_true.shape:
            raise ValueError("Pipeline predictions must match the label shape")

        # Prefer predicted probability for the positive class when available.
        y_score = y_pred.astype(float)
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = np.asarray(
                self.pipeline.predict_proba(X), dtype=float
            )

            if probabilities.ndim == 2 and probabilities.shape[0] == y_true.size:
                classes = getattr(self.model, "classes_", None)
                if classes is None:
                    classes = getattr(self.model, "classes", None)

                if classes is not None:
                    classes = np.asarray(classes)
                    positive = np.flatnonzero(classes == 1)

                    if positive.size:
                        index = int(positive[0])
                        if index < probabilities.shape[1]:
                            y_score = probabilities[:, index]

        result["accuracy"] = accuracy(y_true, y_pred)
        self.metrics.update(y_true, y_pred, y_score)
        totals = self.metrics.result()

        result["cumulative_accuracy"] = totals["accuracy"]
        result["cumulative_precision"] = totals["precision"]
        result["cumulative_recall"] = totals["recall"]
        result["cumulative_f1"] = totals["f1"]

        total_auc = totals["auc"]
        result["cumulative_auc"] = (
            float(total_auc) if np.isfinite(total_auc) else None
        )
        result["cumulative_confusion_matrix"] = (
            totals["confusion_matrix"].copy()
        )

        rolling = totals.get("rolling")
        if rolling is not None:
            result["rolling_accuracy"] = rolling["accuracy"]
            result["rolling_precision"] = rolling["precision"]
            result["rolling_recall"] = rolling["recall"]
            result["rolling_f1"] = rolling["f1"]

            rolling_auc = rolling["auc"]
            result["rolling_auc"] = (
                float(rolling_auc) if np.isfinite(rolling_auc) else None
            )

        self.history.append(result)
        self._awaiting_fit_index = len(self.history) - 1
        return result

    def fit_chunk(self, X, y):
        """Train on a chunk and record elapsed time and peak memory."""
        X, y = self._validate_chunk(X, y)
        tracing_was_active = tracemalloc.is_tracing()

        if not tracing_was_active:
            tracemalloc.start()
        else:
            tracemalloc.reset_peak()

        start = time.perf_counter()

        try:
            if hasattr(self.pipeline, "partial_fit"):
                self.pipeline.partial_fit(X, y)
            elif hasattr(self.pipeline, "fit"):
                self.pipeline.fit(X, y)
            else:
                raise TypeError(
                    "pipeline must provide partial_fit() or fit()"
                )

            self._trained_by_trainer = True
            elapsed = time.perf_counter() - start
            _, peak_bytes = tracemalloc.get_traced_memory()

        finally:
            if not tracing_was_active and tracemalloc.is_tracing():
                tracemalloc.stop()

        peak_mb = peak_bytes / (1024.0 ** 2)

        # Attach training information to the chunk just scored, when possible.
        if self._awaiting_fit_index is not None:
            record = self.history[self._awaiting_fit_index]
            record["training_time"] = float(elapsed)
            record["peak_memory_mb"] = float(peak_mb)
            self._awaiting_fit_index = None
        else:
            record = self._empty_record(len(y))
            record["training_time"] = float(elapsed)
            record["peak_memory_mb"] = float(peak_mb)
            self.history.append(record)

        return self

    def reset(self):
        """Clear metrics and history without resetting the pipeline model."""
        self.metrics.reset()
        self.history.clear()
        self._awaiting_fit_index = None
        self._trained_by_trainer = False
        return self

    def get_history(self):
        """Return a copy of the recorded chunk results."""
        return [record.copy() for record in self.history]
