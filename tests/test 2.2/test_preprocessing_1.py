
import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    SimpleImputer,
    OneHotEncoder,
)

def test_standard_scaler_partial_fit():
    X1 = np.array([[1.0], [3.0]])
    X2 = np.array([[5.0], [7.0]])

    scaler = StandardScaler()
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    transformed = scaler.transform(np.array([[4.0]]))

    assert transformed[0, 0] == pytest.approx(0.0)


def test_standard_scaler_transform_shape():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])

    scaler = StandardScaler()
    scaler.fit(X)
    transformed = scaler.transform(X)

    assert transformed.shape == X.shape
    assert np.allclose(np.mean(transformed, axis=0), 0.0)


def test_minmax_scaler_partial_fit():
    scaler = MinMaxScaler()

    scaler.partial_fit(np.array([[2.0], [4.0]]))
    scaler.partial_fit(np.array([[6.0], [8.0]]))

    transformed = scaler.transform(np.array([[2.0], [8.0]]))

    assert transformed[0, 0] == pytest.approx(0.0)
    assert transformed[1, 0] == pytest.approx(1.0)


def test_minmax_scaler_transform_shape():
    X = np.array([[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]])

    scaler = MinMaxScaler()
    scaler.fit(X)
    transformed = scaler.transform(X)

    assert transformed.shape == X.shape
    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 1.0)


def test_simple_imputer_mean():
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X)
    transformed = imputer.transform(X)

    assert transformed[0, 1] == pytest.approx(5.0)
    assert not np.isnan(transformed).any()


def test_simple_imputer_partial_fit():
    imputer = SimpleImputer(strategy="mean")

    imputer.partial_fit(np.array([[2.0], [4.0]]))
    imputer.partial_fit(np.array([[6.0], [8.0]]))

    transformed = imputer.transform(np.array([[np.nan]]))

    assert transformed[0, 0] == pytest.approx(5.0)


def test_onehot_encoder_partial_fit():
    encoder = OneHotEncoder()

    encoder.partial_fit(np.array([["red"], ["blue"]]))
    encoder.partial_fit(np.array([["green"], ["red"]]))

    transformed = encoder.transform(
        np.array([["red"], ["blue"], ["green"]])
    )

    assert transformed.shape == (3, 3)
    assert np.all(np.sum(transformed, axis=1) == 1)

def test_onehot_encoder_preserves_known_categories():
    encoder = OneHotEncoder()

    encoder.partial_fit(np.array([["cat"], ["dog"]]))
    encoder.partial_fit(np.array([["bird"]]))

    transformed = encoder.transform(
        np.array([["cat"], ["dog"], ["bird"]])
    )

    assert transformed.shape == (3, 3)
    assert np.all(np.sum(transformed, axis=1) == 1)
    assert not np.array_equal(transformed[0], transformed[1])
    assert not np.array_equal(transformed[0], transformed[2])
    assert not np.array_equal(transformed[1], transformed[2])



def test_standard_scaler_rejects_feature_mismatch():
    scaler = StandardScaler()
    scaler.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError):
        scaler.partial_fit(np.array([[5.0]]))
