import pandas as pd
import pytest
from tathagata_ai_839.pipelines.data_science.nodes import (
    split_data,
    detect_target_drift
)


@pytest.fixture
def sample_features():
    return pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature2': [0.2, 0.3, 0.4, 0.5, 0.6],
        'feature3': [0.3, 0.4, 0.5, 0.6, 0.7]
    })


@pytest.fixture
def sample_target():
    return pd.DataFrame({'target': [0, 1, 0, 1, 0]})


@pytest.fixture
def model_parameters():
    return {
        "test_size": 0.4,
        "random_state": 42,
        "model_params": {"n_estimators": 10, "max_depth": 2, "random_state": 42}
    }


def test_split_data(sample_features, sample_target, model_parameters):
    result = split_data(sample_features, sample_target, model_parameters)

    assert isinstance(result, dict)
    assert all(key in result for key in ["X_train", "X_test", "y_train", "y_test"])
    assert len(result["X_train"]) + len(result["X_test"]) == len(sample_features)


def test_detect_target_drift(mocker):
    # Patch the entire evidently report module
    mocker.patch('evidently.report.Report')
    mocker.patch('evidently.ColumnMapping')
    mocker.patch('os.makedirs')
    mocker.patch('json.dump')
    mocker.patch('builtins.open', mocker.mock_open())

    y_train = pd.DataFrame({'target': [0, 1, 0, 1, 0]})
    y_test = pd.DataFrame({'target': [1, 0, 1, 0, 1]})

    # Mock the report instance after import
    mock_report = mocker.patch('tathagata_ai_839.pipelines.data_science.nodes.Report')()
    mock_report.as_dict.return_value = {
        "metrics": [{
            "result": {
                "drift_detected": True,
                "drift_score": 0.8
            }
        }]
    }

    with pytest.raises(ValueError, match="Significant target drift detected"):
        detect_target_drift(y_train, y_test)