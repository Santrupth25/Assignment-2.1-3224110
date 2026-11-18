"""A NumPy decision tree classifier with buffered incremental retraining."""

import numpy as np


def _class_counts(y):
    """Return counts for arbitrary class labels."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return counts


def gini(y):
    """Calculate Gini impurity for a one-dimensional label array."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("Labels must be 1D")
    if y.size == 0:
        return 0.0
    counts = _class_counts(y).astype(float)
    probabilities = counts / counts.sum()
    return float(1.0 - np.sum(probabilities**2))


def entropy(y):
    """Calculate Shannon entropy in bits for a one-dimensional label array."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("Labels must be 1D")
    if y.size == 0:
        return 0.0
    counts = _class_counts(y).astype(float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def information_gain(y, mask, criterion="gini"):
    """Calculate impurity reduction from a boolean split mask."""
    y = np.asarray(y)
    mask = np.asarray(mask)
    if y.ndim != 1 or mask.ndim != 1:
        raise ValueError("Inputs must be 1D")
    if y.shape != mask.shape:
        raise ValueError("Labels and mask must have the same shape")
    if mask.dtype != np.dtype(bool):
        raise TypeError("Mask must be boolean")
    if criterion == "gini":
        impurity = gini
    elif criterion == "entropy":
        impurity = entropy
    else:
        raise ValueError("criterion must be 'gini' or 'entropy'")
    if y.size == 0 or mask.all() or not mask.any():
        return 0.0
    before = impurity(y)
    left_count = int(np.sum(mask))
    right_count = y.size - left_count
    after = (
        left_count * impurity(y[mask])
        + right_count * impurity(y[~mask])
    ) / y.size
    return float(before - after)


class Node:
    """Tree node containing a split or a leaf prediction."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        label=None,
        class_counts=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label  # Encoded class index for leaves; None for split nodes.
        self.class_counts = class_counts


class DecisionTreeClassifier:
    """Train a depth-limited classification tree with buffered stream updates.

    ``partial_fit`` retrains from the buffered chunks whenever their combined
    sample count reaches ``buffer_size``. The latest fitted tree is used between
    retraining events.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_features=None,
        buffer_size=500,
        criterion="gini",
        random_state=42,
    ):
        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, (int, np.integer))
            or max_depth < 0
        ):
            raise ValueError("max_depth must be a non-negative integer or None")
        if (
            isinstance(min_samples_split, bool)
            or not isinstance(min_samples_split, (int, np.integer))
            or min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer of at least 2")
        if isinstance(buffer_size, bool) or not isinstance(
            buffer_size, (int, np.integer)
        ):
            raise TypeError("buffer_size must be an integer")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if isinstance(max_features, (int, np.integer)) and not isinstance(max_features, bool):
            if max_features <= 0:
                raise ValueError("max_features must be positive")
        elif isinstance(max_features, float):
            if not 0 < max_features <= 1:
                raise ValueError("float max_features must be in (0, 1]")
        elif max_features not in (None, "sqrt", "log2"):
            raise ValueError("Invalid max_features value")

        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.criterion = criterion
        self.random_state = random_state

        self.classes = None
        self.classes_ = None
        self.n_features = None
        self.n_features_in_ = None
        self.root = None
        self.tree_ = None
        self._n_classes = 0
        self._X_buffer = []
        self._y_buffer = []
        self._buffer_count = 0
        self.retrain_count = 0
        self._rng = np.random.default_rng(self.random_state)

    def _feature_count(self, n_features):
        """Resolve the number of split features to consider at one node."""
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, float):
            return max(1, int(np.ceil(self.max_features * n_features)))
        return min(int(self.max_features), n_features)

    def _selected_features(self, n_features):
        """Select candidate features for a split, using the configured seed."""
        count = self._feature_count(n_features)
        if count >= n_features:
            return np.arange(n_features)
        return np.sort(self._rng.choice(n_features, size=count, replace=False))

    def _best_split(self, X, y):
        """Return the best split feature, threshold, and impurity gain."""
        return self._best_split_vectorized(X, y)

    def _best_split_loop(self, X, y):
        """Find a best split by checking candidate thresholds one at a time."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or X.size == 0:
            raise ValueError("Invalid input dimensions or shapes")
        if np.isinf(X).any():
            raise ValueError("Features cannot contain infinity")

        best_feature, best_threshold, best_gain = None, None, 0.0
        for feature in self._selected_features(X.shape[1]):
            values = X[:, feature]
            unique_values = np.unique(values[np.isfinite(values)])
            if unique_values.size < 2:
                continue
            thresholds = unique_values[:-1] + (unique_values[1:] - unique_values[:-1]) / 2
            for threshold in thresholds:
                gain = information_gain(y, values <= threshold, self.criterion)
                if gain > best_gain + 1e-12:
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_gain = float(gain)
        return best_feature, best_threshold, best_gain

    def _best_split_vectorized(self, X, y):
        """Find a best split using cumulative class counts for each feature."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or X.size == 0:
            raise ValueError("Invalid input dimensions or shapes")
        if np.isinf(X).any():
            raise ValueError("Features cannot contain infinity")
        if y.size == 0:
            raise ValueError("Labels cannot be empty")

        best_feature, best_threshold, best_gain = None, None, 0.0
        n_samples, n_features = X.shape
        total_counts = np.bincount(y, minlength=self._n_classes)
        before = gini(y) if self.criterion == "gini" else entropy(y)

        for feature in self._selected_features(n_features):
            values = X[:, feature]
            valid = np.isfinite(values)
            order = np.argsort(values[valid], kind="stable")
            sorted_values = values[valid][order]
            sorted_y = y[valid][order]
            cuts = np.flatnonzero(np.diff(sorted_values) > 0)
            if cuts.size == 0:
                continue

            thresholds = sorted_values[cuts] + (
                sorted_values[cuts + 1] - sorted_values[cuts]
            ) / 2
            left_n = cuts + 1
            right_n = n_samples - left_n
            one_hot = np.eye(self._n_classes, dtype=np.int64)[sorted_y]
            left_counts = np.cumsum(one_hot, axis=0)[cuts]
            right_counts = total_counts[None, :] - left_counts
            left_p = left_counts / left_n[:, None]
            right_p = right_counts / right_n[:, None]

            if self.criterion == "gini":
                left_impurity = 1.0 - np.sum(left_p**2, axis=1)
                right_impurity = 1.0 - np.sum(right_p**2, axis=1)
            else:
                left_terms = np.where(left_p > 0, left_p * np.log2(np.maximum(left_p, 1e-300)), 0.0)
                right_terms = np.where(right_p > 0, right_p * np.log2(np.maximum(right_p, 1e-300)), 0.0)
                left_impurity = -np.sum(left_terms, axis=1)
                right_impurity = -np.sum(right_terms, axis=1)

            gains = before - (
                left_n * left_impurity + right_n * right_impurity
            ) / n_samples
            best_index = int(np.argmax(gains))
            if gains[best_index] > best_gain + 1e-12:
                best_gain = float(gains[best_index])
                best_feature = int(feature)
                best_threshold = float(thresholds[best_index])

        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        """Build a subtree recursively from encoded labels."""
        counts = np.bincount(y, minlength=self._n_classes)
        majority = int(np.argmax(counts))
        pure = np.count_nonzero(counts) <= 1
        depth_limit = self.max_depth is not None and depth >= self.max_depth
        if depth_limit or len(y) < self.min_samples_split or pure:
            return Node(label=majority, class_counts=counts)

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 1e-12:
            return Node(label=majority, class_counts=counts)

        left_mask = X[:, feature] <= threshold
        if not left_mask.any() or left_mask.all():
            return Node(label=majority, class_counts=counts)
        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[left_mask], y[left_mask], depth + 1),
            right=self._build(X[~left_mask], y[~left_mask], depth + 1),
        )

    @staticmethod
    def _validate_xy(X, y):
        """Validate training features and labels and return normalized arrays."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")
        if X.shape[0] != y.size:
            raise ValueError("X and y must have matching sample counts")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Training data cannot be empty")
        if np.isinf(X).any():
            raise ValueError("X cannot contain infinity")
        if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
            raise ValueError("Labels cannot contain NaN or infinity")
        return X, y

    def partial_fit(self, X_chunk, y_chunk):
        """Buffer a chunk and retrain when buffered samples reach buffer_size."""
        X_chunk, y_chunk = self._validate_xy(X_chunk, y_chunk)
        expected_features = self.n_features
        if expected_features is None and self._X_buffer:
            expected_features = self._X_buffer[0].shape[1]
        if expected_features is not None and X_chunk.shape[1] != expected_features:
            raise ValueError("Feature count does not match")

        self._X_buffer.append(X_chunk.copy())
        self._y_buffer.append(y_chunk.copy())
        self._buffer_count += y_chunk.size

        if self._buffer_count >= self.buffer_size:
            buffered_X = np.vstack(self._X_buffer)
            buffered_y = np.concatenate(self._y_buffer)
            self.fit(buffered_X, buffered_y)
            self.retrain_count += 1
        return self

    def fit(self, X, y):
        """Fit a new decision tree from scratch and clear any pending buffer."""
        X, y = self._validate_xy(X, y)
        self.classes, encoded_y = np.unique(y, return_inverse=True)
        self.classes_ = self.classes.copy()
        self._n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.n_features_in_ = self.n_features
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._build(X, encoded_y.astype(int), depth=0)
        self.tree_ = self.root
        self._X_buffer.clear()
        self._y_buffer.clear()
        self._buffer_count = 0
        return self

    def _walk(self, sample):
        """Return the leaf reached by one sample."""
        node = self.root
        while node.label is None:
            # A NaN comparison is false, so missing feature values follow the right branch.
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def predict(self, X):
        """Predict class labels for each row of a two-dimensional array."""
        if self.root is None:
            raise ValueError("Decision tree must be fitted first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("Input shape does not match fitted data")
        if np.isinf(X).any():
            raise ValueError("X cannot contain infinity")
        encoded = [self._walk(row).label for row in X]
        return self.classes[np.asarray(encoded, dtype=int)]

    def predict_proba(self, X):
        """Return leaf-based class probabilities in ``classes_`` order."""
        if self.root is None:
            raise ValueError("Decision tree must be fitted first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("Input shape does not match fitted data")
        if np.isinf(X).any():
            raise ValueError("X cannot contain infinity")

        probabilities = np.zeros((X.shape[0], self._n_classes), dtype=float)
        for row_index, row in enumerate(X):
            counts = self._walk(row).class_counts.astype(float)
            probabilities[row_index] = counts / counts.sum()
        return probabilities

    def score(self, X, y):
        """Return classification accuracy on the supplied labelled data."""
        y = np.asarray(y)
        predictions = self.predict(X)
        if y.ndim != 1 or y.size != predictions.size:
            raise ValueError("y must be 1D and match the number of samples")
        return float(np.mean(predictions == y))
