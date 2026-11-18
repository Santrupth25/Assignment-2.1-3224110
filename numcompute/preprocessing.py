"""NumPy preprocessing tools with batch and incremental fitting."""

import numpy as np


class ColumnSelector:
    """Select columns by their names and preserve their requested order."""

    def __init__(self, columns, selected):
        self.columns = list(columns)
        self.selected = list(selected)
        self.indices = None

    def fit(self, data):
        """Validate the input and store indices for the selected columns."""
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != len(self.columns):
            raise ValueError("Input shape does not match column names")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Column names must be unique")
        missing = [name for name in self.selected if name not in self.columns]
        if missing:
            raise ValueError(f"Selected column not found: {missing[0]}")
        self.indices = np.asarray(
            [self.columns.index(name) for name in self.selected], dtype=int
        )
        return self

    def transform(self, data):
        """Return the selected columns from a fitted selector."""
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != len(self.columns):
            raise ValueError("Input shape does not match column names")
        if self.indices is None:
            raise ValueError("ColumnSelector must be fitted first")
        return data[:, self.indices]

    def fit_transform(self, data):
        """Fit the selector and select columns in one call."""
        return self.fit(data).transform(data)


class SimpleImputer:
    """Fill missing values with a constant or an incrementally learned mean."""

    def __init__(self, fill_value=0, dtype=float, strategy="constant"):
        if dtype not in (float, str):
            raise ValueError("dtype must be float or str")
        if strategy not in ("constant", "mean"):
            raise ValueError("strategy must be 'constant' or 'mean'")
        if strategy == "mean" and dtype is not float:
            raise ValueError("Mean imputation requires dtype=float")

        self.fill_value = fill_value
        self.dtype = dtype
        self.strategy = strategy
        self.n_features_in_ = None
        self.count = None
        self.total = None
        self.fill_values = None

    @staticmethod
    def _missing_mask(data):
        """Identify common missing-value markers in mixed input arrays."""
        strings = np.char.strip(np.asarray(data, dtype=str))
        lower = np.char.lower(strings)
        return (
            (strings == "")
            | (lower == "nan")
            | (lower == "none")
            | (lower == "null")
        )

    def partial_fit(self, data):
        """Update learned imputation values from another chunk of rows."""
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("Input must be a non-empty 2D array")

        if self.n_features_in_ is None:
            self.n_features_in_ = data.shape[1]
        elif data.shape[1] != self.n_features_in_:
            raise ValueError("Number of columns changed")

        if self.strategy == "constant":
            return self

        strings = np.asarray(data, dtype=str)
        missing = self._missing_mask(data)
        numeric = np.full(data.shape, np.nan, dtype=float)
        valid = ~missing
        try:
            numeric[valid] = strings[valid].astype(float)
        except ValueError as exc:
            raise ValueError("Mean imputation requires numeric data") from exc
        valid &= np.isfinite(numeric)

        chunk_count = np.sum(valid, axis=0)
        chunk_total = np.sum(np.where(valid, numeric, 0.0), axis=0)
        if self.count is None:
            self.count = chunk_count.astype(np.int64)
            self.total = chunk_total.astype(float)
        else:
            self.count += chunk_count
            self.total += chunk_total

        fallback = float(self.fill_value)
        self.fill_values = np.divide(
            self.total,
            self.count,
            out=np.full(self.n_features_in_, fallback, dtype=float),
            where=self.count > 0,
        )
        return self

    def fit(self, data):
        """Reset and learn fill values from the supplied dataset."""
        self.n_features_in_ = None
        self.count = None
        self.total = None
        self.fill_values = None
        return self.partial_fit(data)

    def transform(self, data):
        """Replace missing values and return either a float or string array."""
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("Input must be 2D")
        if self.n_features_in_ is not None and data.shape[1] != self.n_features_in_:
            raise ValueError("Number of columns changed")

        strings = np.asarray(data, dtype=str)
        missing = self._missing_mask(data)

        if self.dtype is str:
            return np.where(missing, str(self.fill_value), strings)

        numeric = np.full(data.shape, np.nan, dtype=float)
        valid = ~missing
        try:
            numeric[valid] = strings[valid].astype(float)
        except ValueError as exc:
            raise ValueError("Input contains non-numeric values") from exc
        numeric[~np.isfinite(numeric)] = np.nan

        if self.strategy == "mean":
            if self.fill_values is None:
                raise ValueError("Fit the imputer before mean transformation")
            fills = self.fill_values
        else:
            try:
                fill = float(self.fill_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Numeric imputation needs a numeric fill_value") from exc
            fills = np.full(data.shape[1], fill, dtype=float)

        return np.where(np.isnan(numeric), fills, numeric)

    def fit_transform(self, data):
        """Fit the imputer and transform the same dataset."""
        return self.fit(data).transform(data)


class StandardScaler:
    """Standardise numeric columns using incrementally merged statistics."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.var = None
        self.n_features_in_ = None
        self._count = None
        self._mean = None
        self._M2 = None

    def partial_fit(self, data):
        """Update per-column means and variances from a new data chunk."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("Input must be a non-empty 2D array")
        if np.isinf(data).any():
            raise ValueError("Input cannot contain infinity")

        if self.n_features_in_ is None:
            self.n_features_in_ = data.shape[1]
            self._count = np.zeros(data.shape[1], dtype=np.int64)
            self._mean = np.zeros(data.shape[1], dtype=float)
            self._M2 = np.zeros(data.shape[1], dtype=float)
        elif data.shape[1] != self.n_features_in_:
            raise ValueError("Number of columns changed")

        valid = np.isfinite(data)
        chunk_count = np.sum(valid, axis=0).astype(np.int64)
        chunk_sum = np.sum(np.where(valid, data, 0.0), axis=0)
        chunk_mean = np.divide(
            chunk_sum,
            chunk_count,
            out=np.zeros(data.shape[1], dtype=float),
            where=chunk_count > 0,
        )
        centered = np.where(valid, data - chunk_mean, 0.0)
        chunk_M2 = np.sum(centered**2, axis=0)

        old_count = self._count.copy()
        total_count = old_count + chunk_count
        delta = chunk_mean - self._mean
        correction = np.divide(
            delta**2 * old_count * chunk_count,
            total_count,
            out=np.zeros(data.shape[1], dtype=float),
            where=total_count > 0,
        )
        self._M2 += chunk_M2 + correction
        self._mean += np.divide(
            delta * chunk_count,
            total_count,
            out=np.zeros(data.shape[1], dtype=float),
            where=total_count > 0,
        )
        self._count = total_count
        self._refresh_stats()
        return self

    def _refresh_stats(self):
        """Refresh the public mean, variance, and standard deviation arrays."""
        self.mean = np.where(self._count > 0, self._mean, np.nan)
        self.var = np.divide(
            self._M2,
            self._count,
            out=np.full(self.n_features_in_, np.nan, dtype=float),
            where=self._count > 0,
        )
        self.std = np.sqrt(self.var)

    def fit(self, data):
        """Reset the scaler and fit it to a complete dataset."""
        self.__init__()
        return self.partial_fit(data)

    def transform(self, data):
        """Standardise columns using the fitted means and standard deviations."""
        data = np.asarray(data, dtype=float)
        if self.mean is None:
            raise ValueError("StandardScaler must be fitted first")
        if data.ndim != 2 or data.shape[1] != self.n_features_in_:
            raise ValueError("Input shape does not match fitted data")
        if np.isinf(data).any():
            raise ValueError("Input cannot contain infinity")
        safe_std = np.where(self.std == 0, 1.0, self.std)
        return (data - self.mean) / safe_std

    def fit_transform(self, data):
        """Fit the scaler and standardise the same dataset."""
        return self.fit(data).transform(data)


class MinMaxScaler:
    """Scale numeric columns to a configured range, supporting chunk updates."""

    def __init__(self, lower=0, upper=1):
        if lower >= upper:
            raise ValueError("lower must be smaller than upper")
        self.lower = lower
        self.upper = upper
        self.min = None
        self.max = None
        self.n_features_in_ = None
        self._min = None
        self._max = None
        self._seen = None

    def partial_fit(self, data):
        """Update per-column minimum and maximum from a new chunk."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("Input must be a non-empty 2D array")
        if np.isinf(data).any():
            raise ValueError("Input cannot contain infinity")

        if self.n_features_in_ is None:
            self.n_features_in_ = data.shape[1]
            self._min = np.full(data.shape[1], np.inf)
            self._max = np.full(data.shape[1], -np.inf)
            self._seen = np.zeros(data.shape[1], dtype=bool)
        elif data.shape[1] != self.n_features_in_:
            raise ValueError("Number of columns changed")

        valid = np.isfinite(data)
        has_values = np.any(valid, axis=0)
        chunk_min = np.min(np.where(valid, data, np.inf), axis=0)
        chunk_max = np.max(np.where(valid, data, -np.inf), axis=0)
        update = has_values & self._seen
        first = has_values & ~self._seen
        self._min[update] = np.minimum(self._min[update], chunk_min[update])
        self._max[update] = np.maximum(self._max[update], chunk_max[update])
        self._min[first] = chunk_min[first]
        self._max[first] = chunk_max[first]
        self._seen |= has_values
        self.min = np.where(self._seen, self._min, np.nan)
        self.max = np.where(self._seen, self._max, np.nan)
        return self

    def fit(self, data):
        """Reset the scaler and calculate bounds from a complete dataset."""
        self.min = None
        self.max = None
        self.n_features_in_ = None
        self._min = None
        self._max = None
        self._seen = None
        return self.partial_fit(data)

    def transform(self, data):
        """Scale columns into the configured range using fitted bounds."""
        data = np.asarray(data, dtype=float)
        if self.min is None:
            raise ValueError("MinMaxScaler must be fitted first")
        if data.ndim != 2 or data.shape[1] != self.n_features_in_:
            raise ValueError("Input shape does not match fitted data")
        if np.isinf(data).any():
            raise ValueError("Input cannot contain infinity")
        span = self.max - self.min
        safe_span = np.where(span == 0, 1.0, span)
        scaled = (data - self.min) / safe_span
        return scaled * (self.upper - self.lower) + self.lower

    def fit_transform(self, data):
        """Fit the scaler and scale the same dataset."""
        return self.fit(data).transform(data)


class OneHotEncoder:
    """Encode categorical columns and allow categories to grow across chunks."""

    def __init__(self):
        self.categories = None
        self.n_features_in_ = None

    def partial_fit(self, data):
        """Add categories from a chunk while retaining previously learned ones."""
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("Input must be a non-empty 2D array")

        if self.n_features_in_ is None:
            self.n_features_in_ = data.shape[1]
            self.categories = [np.unique(data[:, i]) for i in range(data.shape[1])]
        else:
            if data.shape[1] != self.n_features_in_:
                raise ValueError("Number of columns changed")
            self.categories = [
                np.union1d(old, np.unique(data[:, i]))
                for i, old in enumerate(self.categories)
            ]
        return self

    def fit(self, data):
        """Reset the encoder and learn categories from a complete dataset."""
        self.categories = None
        self.n_features_in_ = None
        return self.partial_fit(data)

    def transform(self, data):
        """Return binary indicator columns using the learned category order."""
        data = np.asarray(data)
        if self.categories is None:
            raise ValueError("OneHotEncoder must be fitted first")
        if data.ndim != 2 or data.shape[1] != self.n_features_in_:
            raise ValueError("Input shape does not match fitted data")

        parts = [
            (data[:, i, None] == categories[None, :]).astype(int)
            for i, categories in enumerate(self.categories)
        ]
        if not parts:
            return np.empty((data.shape[0], 0), dtype=int)
        return np.hstack(parts)

    def fit_transform(self, data):
        """Fit the encoder and encode the same dataset."""
        return self.fit(data).transform(data)
