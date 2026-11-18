
import numpy as np

from numcompute.tree import DecisionTreeClassifier


def test_tree_learns_simple_classes():
    X = np.array([[1], [2], [8], [9]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    assert np.array_equal(tree.predict(X), y)
    assert tree.score(X, y) == 1.0


def test_tree_predicts_each_sample():
    X = np.array([[1], [2], [8], [9]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    predictions = tree.predict(np.array([[1.5], [8.5]]))

    assert predictions.shape == (2,)
    assert predictions[0] == 0
    assert predictions[1] == 1


def test_tree_respects_depth_limit():
    X = np.array([[1], [2], [8], [9]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)

    assert tree.predict(X).shape == y.shape


def test_tree_incremental_training():
    X = np.array([[1], [2], [8], [9]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, buffer_size=2)

    tree.partial_fit(X[:2], y[:2])

    # The first chunk triggers training.
    assert tree.root is not None
    assert tree.retrain_count == 1
    assert np.array_equal(tree.predict(X[:2]), y[:2])

    tree.partial_fit(X[2:], y[2:])

    # The second chunk triggers another training cycle.
    assert tree.retrain_count == 2
    assert np.array_equal(tree.predict(X[2:]), y[2:])
