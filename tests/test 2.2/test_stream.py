
import numpy as np
import pytest

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


class TreePipeline:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def make_data():
    X = np.array([[1], [2], [8], [9]], dtype=float)
    y = np.array([0, 0, 1, 1])
    return X, y


def make_trainer():
    model = DecisionTreeClassifier(max_depth=3)
    pipeline = TreePipeline(model)
    trainer = StreamTrainer(pipeline)
    return model, pipeline, trainer


def test_stream_trainer_initialisation():
    model, pipeline, trainer = make_trainer()

    assert trainer.pipeline is pipeline
    assert trainer.model is model
    assert trainer.history == []


def test_stream_trainer_processes_chunk():
    X, y = make_data()
    model, pipeline, trainer = make_trainer()

    trainer.fit_chunk(X, y)

    assert model.root is not None
    assert np.array_equal(pipeline.predict(X), y)


def test_stream_trainer_scores_chunk():
    X, y = make_data()
    _, pipeline, trainer = make_trainer()

    pipeline.fit(X, y)
    result = trainer.score_chunk(X, y)

    assert result["accuracy"] == 1.0
    assert result["cumulative_accuracy"] == 1.0
    assert result["n_samples"] == len(y)


def test_stream_trainer_rejects_invalid_chunk():
    _, _, trainer = make_trainer()

    X = np.array([[1], [2]], dtype=float)
    y = np.array([0])

    with pytest.raises(ValueError):
        trainer.fit_chunk(X, y)


def test_stream_trainer_records_training_details():
    X, y = make_data()
    _, _, trainer = make_trainer()

    trainer.fit_chunk(X, y)
    history = trainer.get_history()

    assert len(history) == 1
    assert history[0]["training_time"] >= 0
    assert history[0]["peak_memory_mb"] >= 0



def test_stream_trainer_reset():
    X, y = make_data()
    _, _, trainer = make_trainer()

    trainer.fit_chunk(X, y)
    trainer.reset()

    assert trainer.history == []

    with pytest.raises(ValueError, match="No data has been received"):
        trainer.metrics.result()
