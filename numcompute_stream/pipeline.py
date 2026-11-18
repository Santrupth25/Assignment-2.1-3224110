
"""Batch and streaming pipeline composition for NumCompute."""

from copy import deepcopy
import numpy as np


class Pipeline:
    """Apply transformers in order and optionally finish with an estimator.

    Each step is a (name, object) pair. Transformers provide fit() and
    transform(); an optional final estimator provides fit() and predict().
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline must contain at least one step")

        cleaned = []
        names = []

        for step in steps:
            if not isinstance(step, (tuple, list)) or len(step) != 2:
                raise ValueError("Each step must be a (name, object) pair")

            name, obj = step
            if not isinstance(name, str) or not name:
                raise ValueError("Step names must be non-empty strings")
            if obj is None:
                raise ValueError("Pipeline steps cannot be None")
            if name in names:
                raise ValueError("Pipeline step names must be unique")

            names.append(name)
            cleaned.append((name, obj))

        for _, obj in cleaned[:-1]:
            if hasattr(obj, "predict"):
                raise ValueError("An estimator must be the final step")
            if not hasattr(obj, "fit") or not hasattr(obj, "transform"):
                raise TypeError("Intermediate steps need fit() and transform()")

        self.steps = cleaned
        self._fitted_transformer_indices = set()
        self._active_transformers = None
        self._pending_X = []
        self._pending_y = []
        self._pending_count = 0
        self._estimator_template = None

        if self._has_estimator():
            self._estimator_template = deepcopy(self.steps[-1][1])

    def _has_estimator(self):
        """Check whether the final step supports prediction."""
        return hasattr(self.steps[-1][1], "predict")

    def _transformer_objects(self, snapshot=False):
        """Return current transformers or the active model's saved snapshot."""
        if snapshot and self._active_transformers is not None:
            return self._active_transformers

        end = -1 if self._has_estimator() else len(self.steps)
        return [obj for _, obj in self.steps[:end]]

    def _snapshot_transformers(self):
        """Save preprocessing state matching the currently trained model."""
        self._active_transformers = deepcopy(self._transformer_objects())

    @staticmethod
    def _apply_transformers(data, transformers):
        """Pass data through a sequence of fitted transformers."""
        processed = data
        for transformer in transformers:
            processed = transformer.transform(processed)
        return processed

    def fit(self, data, y=None):
        """Fit all transformers and the optional final estimator."""
        processed = data
        self._fitted_transformer_indices.clear()
        self._active_transformers = None
        self._pending_X = []
        self._pending_y = []
        self._pending_count = 0

        end = len(self.steps) - 1 if self._has_estimator() else len(self.steps)

        for index, (_, transformer) in enumerate(self.steps[:end]):
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError("Transformer steps need fit() and transform()")

            transformer.fit(processed)
            self._fitted_transformer_indices.add(index)
            processed = transformer.transform(processed)

        if self._has_estimator():
            if y is None:
                raise ValueError("The final estimator requires y")

            self.steps[-1][1].fit(processed, y)
            self._snapshot_transformers()

        return self

    def transform(self, data):
        """Transform data through a transformer-only pipeline."""
        if self._has_estimator():
            raise ValueError("Cannot transform a pipeline ending with an estimator")

        return self._apply_transformers(data, self._transformer_objects())

    def fit_transform(self, data, y=None):
        """Fit a transformer-only pipeline and return transformed data."""
        if self._has_estimator():
            raise ValueError(
                "fit_transform is unavailable when the pipeline ends with an estimator"
            )

        self.fit(data, y=y)
        return self.transform(data)

    def predict(self, data):
        """Transform input using the active model's preprocessing and predict."""
        if not self._has_estimator():
            raise ValueError("The final step must support predict()")

        processed = self._apply_transformers(
            data, self._transformer_objects(snapshot=True)
        )
        return self.steps[-1][1].predict(processed)

    def predict_proba(self, data):
        """Return probabilities when the estimator supports predict_proba()."""
        if not self._has_estimator():
            raise ValueError("The final step must be an estimator")

        estimator = self.steps[-1][1]
        if not hasattr(estimator, "predict_proba"):
            raise ValueError("The estimator does not support predict_proba()")

        processed = self._apply_transformers(
            data, self._transformer_objects(snapshot=True)
        )
        return estimator.predict_proba(processed)

    def score(self, data, y):
        """Calculate classification accuracy on labelled data."""
        predictions = self.predict(data)
        y = np.asarray(y)

        if y.ndim != 1 or predictions.ndim != 1:
            raise ValueError("y and predictions must be 1D")
        if y.shape != predictions.shape:
            raise ValueError("y must match the number of predictions")
        if y.size == 0:
            raise ValueError("Cannot score an empty dataset")

        return float(np.mean(predictions == y))

    def partial_fit(self, data, y=None):
        """Update transformers and train the estimator on a data chunk.

        Incremental transformers use partial_fit(). Other transformers are
        fitted on their first chunk. Models with buffer_size and retrain_count
        accumulate raw chunks and transform the complete buffer consistently
        before retraining.
        """
        raw_chunk = np.asarray(data).copy()
        if raw_chunk.ndim != 2 or raw_chunk.shape[0] == 0:
            raise ValueError("Input chunk must be a non-empty 2D array")

        estimator_present = self._has_estimator()
        end = len(self.steps) - 1 if estimator_present else len(self.steps)
        processed = data

        for index, (_, transformer) in enumerate(self.steps[:end]):
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(processed)
                self._fitted_transformer_indices.add(index)
            elif index not in self._fitted_transformer_indices:
                transformer.fit(processed)
                self._fitted_transformer_indices.add(index)

            processed = transformer.transform(processed)

        if not estimator_present:
            return self

        if y is None:
            raise ValueError("The final estimator requires y")

        y_chunk = np.asarray(y)
        if y_chunk.ndim != 1 or y_chunk.size != raw_chunk.shape[0]:
            raise ValueError("y must have one label per input row")

        estimator = self.steps[-1][1]

        # Handle buffered models such as DecisionTreeClassifier.
        uses_buffer = (
            hasattr(estimator, "buffer_size")
            and hasattr(estimator, "retrain_count")
            and hasattr(estimator, "fit")
        )

        if uses_buffer:
            self._pending_X.append(raw_chunk)
            self._pending_y.append(y_chunk.copy())
            self._pending_count += y_chunk.size

            if self._pending_count >= int(estimator.buffer_size):
                buffered_X = np.vstack(self._pending_X)
                buffered_y = np.concatenate(self._pending_y)

                # Use the same current preprocessing state for the whole buffer.
                buffered_processed = self._apply_transformers(
                    buffered_X, self._transformer_objects()
                )

                old_count = int(estimator.retrain_count)
                estimator.fit(buffered_processed, buffered_y)
                estimator.retrain_count = old_count + 1

                self._pending_X.clear()
                self._pending_y.clear()
                self._pending_count = 0
                self._snapshot_transformers()

            return self

        # Reset estimators that cannot accept a changed feature count.
        input_features = processed.shape[1] if np.ndim(processed) == 2 else None
        fitted_features = getattr(estimator, "n_features_in_", None)
        if fitted_features is None:
            fitted_features = getattr(estimator, "n_features", None)

        if (
            input_features is not None
            and fitted_features is not None
            and int(fitted_features) != input_features
        ):
            estimator = deepcopy(self._estimator_template)
            self.steps[-1] = (self.steps[-1][0], estimator)
            self._active_transformers = None

        old_count = getattr(estimator, "retrain_count", None)

        if hasattr(estimator, "partial_fit"):
            estimator.partial_fit(processed, y_chunk)
        else:
            estimator.fit(processed, y_chunk)

        new_count = getattr(estimator, "retrain_count", None)

        if old_count is None or new_count is None or new_count > old_count:
            self._snapshot_transformers()

        return self

    @property
    def named_steps(self):
        """Return a dictionary of pipeline steps by name."""
        return dict(self.steps)

    def __getitem__(self, key):
        """Get a step by its name or numeric index."""
        if isinstance(key, str):
            return self.named_steps[key]
        return self.steps[key][1]


class FeatureUnion:
    """Fit multiple transformers on the same input and combine their outputs."""

    def __init__(self, transformers):
        if not transformers:
            raise ValueError("FeatureUnion requires at least one transformer")

        cleaned = []
        names = []

        for item in transformers:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each branch must be a (name, object) pair")

            name, transformer = item
            if not isinstance(name, str) or not name:
                raise ValueError("Branch names must be non-empty strings")
            if name in names:
                raise ValueError("Branch names must be unique")
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError("Each branch needs fit() and transform()")

            names.append(name)
            cleaned.append((name, transformer))

        self.transformers = cleaned
        self._fitted_indices = set()

    def fit(self, data, y=None):
        """Fit each branch independently using the same input data."""
        self._fitted_indices.clear()

        for index, (_, transformer) in enumerate(self.transformers):
            if (
                hasattr(transformer, "_has_estimator")
                and transformer._has_estimator()
            ):
                raise ValueError("FeatureUnion branches must be transformer-only")

            transformer.fit(data)
            self._fitted_indices.add(index)

        return self

    def partial_fit(self, data, y=None):
        """Incrementally update branches when partial_fit() is supported."""
        for index, (_, transformer) in enumerate(self.transformers):
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(data)
                self._fitted_indices.add(index)
            elif index not in self._fitted_indices:
                transformer.fit(data)
                self._fitted_indices.add(index)

        return self

    def transform(self, data):
        """Transform each branch and join the features horizontally."""
        results = []
        expected_rows = np.shape(data)[0] if np.ndim(data) > 0 else None

        for _, transformer in self.transformers:
            result = np.asarray(transformer.transform(data))

            if result.ndim != 2:
                raise ValueError("Transformer output must be 2D")
            if expected_rows is not None and result.shape[0] != expected_rows:
                raise ValueError("Transformer output row count does not match input")

            results.append(result)

        return np.hstack(results)

    def fit_transform(self, data, y=None):
        """Fit each branch and return the combined feature matrix."""
        self.fit(data, y=y)
        return self.transform(data)

    @property
    def named_transformers(self):
        """Return a dictionary of branches by name."""
        return dict(self.transformers)

    def __getitem__(self, key):
        """Get a branch by its name or numeric index."""
        if isinstance(key, str):
            return self.named_transformers[key]
        return self.transformers[key][1]
