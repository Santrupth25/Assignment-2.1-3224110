import numpy as np
import pytest

from numcompute.pipeline import Pipeline, FeatureUnion
from numcompute.preprocessing import StandardScaler, MinMaxScaler


class SimpleModel:
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.label = int(y[0])
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.label)


class DoubleTransformer:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * 2


class TripleTransformer:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * 3


class OneDimensionalTransformer:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X)[:, 0]


def make_data():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    return X, y


def test_pipeline_initialisation():
    pipeline = Pipeline([("scale", StandardScaler())])
    assert "scale" in pipeline.named_steps
    assert pipeline["scale"] is pipeline.named_steps["scale"]


def test_pipeline_rejects_empty_steps():
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_fit_transform():
    X = np.array([[1.0], [2.0], [3.0]])
    pipeline = Pipeline([("scale", StandardScaler())])

    transformed = pipeline.fit_transform(X)

    assert transformed.shape == X.shape
    np.testing.assert_allclose(np.mean(transformed, axis=0), [0.0])


def test_pipeline_transform_before_fit():
    X = np.array([[1.0], [2.0]])
    pipeline = Pipeline([("scale", StandardScaler())])

    with pytest.raises(ValueError):
        pipeline.transform(X)


def test_pipeline_predict():
    X, y = make_data()
    pipeline = Pipeline([
        ("double", DoubleTransformer()),
        ("model", SimpleModel()),
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    np.testing.assert_array_equal(predictions, y)


def test_pipeline_score():
    X, y = make_data()
    pipeline = Pipeline([("model", SimpleModel())])
    pipeline.fit(X, y)

    assert pipeline.score(X, y) == pytest.approx(1.0)


def test_pipeline_score_invalid_target_shape():
    X, y = make_data()
    pipeline = Pipeline([("model", SimpleModel())])
    pipeline.fit(X, y)

    with pytest.raises(ValueError):
        pipeline.score(X, np.array([[1], [1], [1]]))


def test_pipeline_requires_targets_for_estimator():
    X, _ = make_data()
    pipeline = Pipeline([("model", SimpleModel())])

    with pytest.raises(ValueError):
        pipeline.fit(X)


def test_pipeline_transform_not_allowed_with_estimator():
    X, y = make_data()
    pipeline = Pipeline([("model", SimpleModel())])
    pipeline.fit(X, y)

    with pytest.raises(ValueError):
        pipeline.transform(X)


def test_pipeline_named_steps():
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("minmax", MinMaxScaler()),
    ])

    assert "scale" in pipeline.named_steps
    assert "minmax" in pipeline.named_steps
    assert pipeline[0] is pipeline.named_steps["scale"]


def test_feature_union_fit_transform():
    X = np.array([[1.0], [2.0], [3.0]])

    union = FeatureUnion([
        ("double", DoubleTransformer()),
        ("triple", TripleTransformer()),
    ])

    transformed = union.fit_transform(X)

    expected = np.array([
        [2.0, 3.0],
        [4.0, 6.0],
        [6.0, 9.0],
    ])

    np.testing.assert_allclose(transformed, expected)


def test_feature_union_named_transformers():
    union = FeatureUnion([
        ("double", DoubleTransformer()),
        ("triple", TripleTransformer()),
    ])

    assert union["double"] is union.named_transformers["double"]
    assert union[1] is union.named_transformers["triple"]


def test_feature_union_rejects_empty_transformers():
    with pytest.raises(ValueError):
        FeatureUnion([])


def test_feature_union_rejects_duplicate_names():
    with pytest.raises(ValueError):
        FeatureUnion([
            ("same", DoubleTransformer()),
            ("same", TripleTransformer()),
        ])


def test_feature_union_rejects_non_2d_output():
    union = FeatureUnion([
        ("invalid", OneDimensionalTransformer()),
    ])

    X = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="2D"):
        union.fit_transform(X)


def test_feature_union_partial_fit():
    union = FeatureUnion([
        ("double", DoubleTransformer()),
        ("triple", TripleTransformer()),
    ])

    X = np.array([[1.0], [2.0]])
    union.partial_fit(X)

    transformed = union.transform(X)
    assert transformed.shape == (2, 2)


def test_pipeline_rejects_duplicate_step_names():
    with pytest.raises(ValueError):
        Pipeline([
            ("scale", StandardScaler()),
            ("scale", MinMaxScaler()),
        ])