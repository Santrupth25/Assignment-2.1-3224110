import numpy as np
import pytest

from numcompute.stats import (
    mean,
    median,
    variance,
    std,
    minimum,
    maximum,
    histogram,
    quantile,
    StreamingStats,
)


def test_mean():
    data = np.array([1, 2, 3, 4, 5])
    assert mean(data) == pytest.approx(3.0)


def test_median():
    data = np.array([1, 3, 5, 7])
    assert median(data) == pytest.approx(4.0)


def test_variance():
    data = np.array([1, 2, 3, 4, 5])
    assert variance(data) == pytest.approx(2.0)


def test_standard_deviation():
    data = np.array([1, 2, 3, 4, 5])
    assert std(data) == pytest.approx(np.sqrt(2.0))


def test_minimum_and_maximum():
    data = np.array([4, 2, 8, 1, 6])
    assert minimum(data) == 1
    assert maximum(data) == 8


def test_statistics_ignore_nan():
    data = np.array([1.0, 2.0, np.nan, 3.0])
    assert mean(data) == pytest.approx(2.0)
    assert median(data) == pytest.approx(2.0)


def test_axis_statistics():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    result = mean(data, axis=0)
    np.testing.assert_allclose(result, [3.0, 4.0])


def test_quantile():
    data = np.array([1, 2, 3, 4, 5])
    result = quantile(data, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0])


def test_quantile_invalid_value():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        quantile(data, 1.5)


def test_histogram():
    data = np.array([0.5, 1.5, 2.5, 3.5])
    counts, edges = histogram(data, bins=[0, 1, 2, 3, 4])

    np.testing.assert_array_equal(counts, [1, 1, 1, 1])
    np.testing.assert_allclose(edges, [0, 1, 2, 3, 4])


def test_streaming_stats_mean_and_variance():
    stats = StreamingStats()

    stats.update_stats(np.array([[1.0], [2.0]]))
    stats.update_stats(np.array([[3.0], [4.0]]))

    np.testing.assert_allclose(stats.mean_result(), [2.5])
    np.testing.assert_allclose(stats.variance(), [1.25])
    np.testing.assert_allclose(stats.std(), [np.sqrt(1.25)])


def test_streaming_stats_ignore_invalid_values():
    stats = StreamingStats(track_quantiles=False)
    stats.update_stats(np.array([[1.0], [np.nan], [3.0]]))

    np.testing.assert_allclose(stats.mean_result(), [2.0])
    np.testing.assert_allclose(stats.variance(), [1.0])


def test_streaming_stats_histogram():
    stats = StreamingStats(hist_edges=[0, 2, 4])
    stats.update_stats(np.array([[1.0], [2.0], [3.0], [4.0]]))

    counts, edges = stats.histogram_result()

    np.testing.assert_array_equal(counts, [[1, 3]])
    np.testing.assert_allclose(edges, [0, 2, 4])


def test_streaming_stats_quantile():
    stats = StreamingStats()
    stats.update_stats(np.array([[1.0], [3.0]]))
    stats.update_stats(np.array([[5.0], [7.0]]))

    result = stats.quantile_result(0.5)
    np.testing.assert_allclose(result, [4.0])


def test_streaming_stats_reset():
    stats = StreamingStats()
    stats.update_stats(np.array([[1.0], [2.0]]))
    stats.reset()

    with pytest.raises(ValueError, match="No data"):
        stats.mean_result()


def test_streaming_stats_feature_mismatch():
    stats = StreamingStats()
    stats.update_stats(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="Feature count"):
        stats.update_stats(np.array([[3.0]]))


def test_streaming_stats_invalid_input():
    stats = StreamingStats()

    with pytest.raises(ValueError):
        stats.update_stats(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        stats.update_stats(np.empty((0, 2)))