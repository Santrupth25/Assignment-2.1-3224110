"""Batch and streaming statistical functions using NumPy."""

import numpy as np


def mean(data, axis=0):
    """Calculate the mean while ignoring NaN values."""
    return np.nanmean(data, axis=axis)


def median(data, axis=0):
    """Calculate the median while ignoring NaN values."""
    return np.nanmedian(data, axis=axis)


def variance(data, axis=0):
    """Calculate population variance while ignoring NaN values."""
    return np.nanvar(data, axis=axis)


def std(data, axis=0):
    """Calculate population standard deviation while ignoring NaN values."""
    return np.nanstd(data, axis=axis)


def minimum(data, axis=0):
    """Return the minimum while ignoring NaN values."""
    return np.nanmin(data, axis=axis)


def maximum(data, axis=0):
    """Return the maximum while ignoring NaN values."""
    return np.nanmax(data, axis=axis)


def histogram(data, bins=10):
    """Return histogram counts and edges, excluding non-finite values."""
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    return np.histogram(values, bins=bins)


def quantile(data, q, axis=0):
    """Calculate quantiles while ignoring NaN values."""
    q_values = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(q_values)) or np.any((q_values < 0) | (q_values > 1)):
        raise ValueError("q must be between 0 and 1")
    return np.nanquantile(data, q_values, axis=axis)


class StreamingStats:
    """Maintain running means, variances, histograms, and optional exact quantiles."""

    def __init__(self, hist_edges=None, track_quantiles=True):
        self.count = None
        self.mean = None
        self.M2 = None
        self.hist_edges = None
        self.hist_counts = None
        self.track_quantiles = bool(track_quantiles)
        self._quantile_chunks = []

        if hist_edges is not None:
            edges = np.asarray(hist_edges, dtype=float)
            if (
                edges.ndim != 1
                or edges.size < 2
                or not np.all(np.isfinite(edges))
                or not np.all(np.diff(edges) > 0)
            ):
                raise ValueError("Histogram edges must be finite and increasing")
            self.hist_edges = edges.copy()

    def update_stats(self, X_chunk):
        """Update running per-column statistics from a new two-dimensional chunk."""
        X_chunk = np.asarray(X_chunk, dtype=float)
        if X_chunk.ndim != 2 or X_chunk.shape[0] == 0:
            raise ValueError("Input must be a non-empty 2D array")

        n_features = X_chunk.shape[1]
        if self.count is None:
            self.count = np.zeros(n_features, dtype=np.int64)
            self.mean = np.zeros(n_features, dtype=float)
            self.M2 = np.zeros(n_features, dtype=float)
        elif n_features != len(self.count):
            raise ValueError("Feature count does not match previous chunks")

        # NaN and infinity are excluded from the numerical summaries.
        valid = np.isfinite(X_chunk)
        clean = np.where(valid, X_chunk, 0.0)
        chunk_count = np.sum(valid, axis=0).astype(np.int64)
        chunk_sum = np.sum(clean, axis=0)
        chunk_mean = np.divide(
            chunk_sum,
            chunk_count,
            out=np.zeros(n_features, dtype=float),
            where=chunk_count > 0,
        )
        centered = np.where(valid, X_chunk - chunk_mean, 0.0)
        chunk_M2 = np.sum(centered**2, axis=0)

        old_count = self.count.copy()
        total_count = old_count + chunk_count
        delta = chunk_mean - self.mean
        correction = np.divide(
            delta**2 * old_count * chunk_count,
            total_count,
            out=np.zeros(n_features, dtype=float),
            where=total_count > 0,
        )
        self.M2 += chunk_M2 + correction
        self.mean += np.divide(
            delta * chunk_count,
            total_count,
            out=np.zeros(n_features, dtype=float),
            where=total_count > 0,
        )
        self.count = total_count

        if self.hist_edges is not None:
            self._update_histogram(X_chunk)
        if self.track_quantiles:
            self._quantile_chunks.append(
                np.where(np.isfinite(X_chunk), X_chunk, np.nan)
            )
        return self

    def _update_histogram(self, X_chunk):
        """Accumulate per-feature histogram counts for configured bin edges."""
        n_features = X_chunk.shape[1]
        n_bins = len(self.hist_edges) - 1
        if self.hist_counts is None:
            self.hist_counts = np.zeros((n_features, n_bins), dtype=np.int64)

        valid = (
            np.isfinite(X_chunk)
            & (X_chunk >= self.hist_edges[0])
            & (X_chunk <= self.hist_edges[-1])
        )
        bin_indices = np.searchsorted(self.hist_edges, X_chunk, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        feature_indices = np.broadcast_to(
            np.arange(n_features)[None, :], X_chunk.shape
        )
        np.add.at(
            self.hist_counts,
            (feature_indices[valid], bin_indices[valid]),
            1,
        )

    def mean_result(self):
        """Return each feature's running mean, or NaN if it has no valid values."""
        if self.count is None:
            raise ValueError("No data has been received")
        return np.where(self.count > 0, self.mean, np.nan)

    def variance(self):
        """Return population variance for each feature."""
        if self.count is None:
            raise ValueError("No data has been received")
        return np.divide(
            self.M2,
            self.count,
            out=np.full_like(self.M2, np.nan),
            where=self.count > 0,
        )

    def std(self):
        """Return population standard deviation for each feature."""
        return np.sqrt(self.variance())

    def histogram_result(self):
        """Return accumulated per-feature counts and a copy of the bin edges."""
        if self.hist_edges is None:
            raise ValueError("Histogram edges are not configured")
        if self.hist_counts is None:
            raise ValueError("No data has been received")
        return self.hist_counts.copy(), self.hist_edges.copy()

    def quantile_result(self, q):
        """Calculate exact quantiles from retained chunks when enabled."""
        if not self.track_quantiles or not self._quantile_chunks:
            raise ValueError("Quantile data is not available")
        q_values = np.asarray(q, dtype=float)
        if not np.all(np.isfinite(q_values)) or np.any((q_values < 0) | (q_values > 1)):
            raise ValueError("q must be between 0 and 1")
        data = np.vstack(self._quantile_chunks)
        return np.nanquantile(data, q_values, axis=0)

    def reset(self):
        """Clear accumulated data while retaining histogram configuration."""
        self.count = None
        self.mean = None
        self.M2 = None
        self.hist_counts = None
        self._quantile_chunks = []
        return self
