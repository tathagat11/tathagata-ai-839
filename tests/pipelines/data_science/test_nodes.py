import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from tathagata_ai_839.pipelines.data_science.nodes import (
    evaluate_model,
    split_data,
    train_model,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    features = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "feature3": np.random.rand(100),
        }
    )
    target = pd.DataFrame({"target": np.random.randint(0, 2, 100)})
    return features, target


@pytest.fixture
def model_parameters():
    return {
        "test_size": 0.2,
        "random_state": 42,
        "model_params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    }


def test_split_data(sample_data, model_parameters):
    features, target = sample_data
    result = split_data(features, target, model_parameters)

    assert "X_train" in result and "X_test" in result
    assert "y_train" in result and "y_test" in result
    assert len(result["X_train"]) == 80
    assert len(result["X_test"]) == 20


def test_train_model(sample_data, model_parameters):
    features, target = sample_data
    split = split_data(features, target, model_parameters)
    model = train_model(split["X_train"], split["y_train"], model_parameters)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.max_depth == 10


def test_evaluate_model(sample_data, model_parameters):
    features, target = sample_data
    split = split_data(features, target, model_parameters)
    model = train_model(split["X_train"], split["y_train"], model_parameters)
    metrics = evaluate_model(model, split["X_test"], split["y_test"])

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert all(0 <= value <= 1 for value in metrics.values())
