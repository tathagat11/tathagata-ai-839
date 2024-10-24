import pandas as pd
import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from tathagata_ai_839.pipelines.data_processing.nodes import (
    load_and_erase_data,
    preprocess_data,
    split_data as split_data_processing,
)
from tathagata_ai_839.pipelines.data_science.nodes import (
    split_data as split_data_science,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "checking_status": ["no checking", "0<=X<200", "no checking"],
            "duration": [9, 18, 12],
            "credit_history": ["existing paid", "existing paid", "no credits/all paid"],
            "purpose": ["business", "business", "furniture/equipment"],
            "credit_amount": [1449.0, 1941.0, 2759.0],
            "savings_status": ["<100", ">=1000", "<100"],
            "employment": ["4<=X<7", "1<=X<4", ">=7"],
            "installment_commitment": [3, 4, 2],
            "personal_status": ["female div/dep/mar", "male single", "male single"],
            "other_parties": ["none", "none", "none"],
            "residence_since": [2, 2, 4],
            "property_magnitude": ["car", "life insurance", "life insurance"],
            "age": [27, 35, 34],
            "other_payment_plans": ["none", "none", "none"],
            "housing": ["own", "own", "own"],
            "existing_credits": [2, 1, 2],
            "job": ["skilled", "unskilled resident", "skilled"],
            "num_dependents": [1, 1, 1],
            "own_telephone": ["none", "yes", "none"],
            "foreign_worker": ["yes", "yes", "yes"],
            "health_status": ["good", "good", "good"],
            "y": [True, False, False],
        }
    )


@pytest.fixture
def sample_erasure_list():
    return pd.DataFrame({
        "index": [1]  # This will erase the second row
    })


@pytest.fixture
def model_parameters():
    return {
        "test_size": 0.2,
        "random_state": 42,
        "model_params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    }


def test_data_processing_pipeline(sample_data, sample_erasure_list):
    # Test only the data processing steps that don't create artifacts
    loaded_data = load_and_erase_data(sample_data, sample_erasure_list)
    preprocessed_data = preprocess_data(loaded_data)
    split_data = split_data_processing(preprocessed_data)

    # Assertions
    assert loaded_data is not None
    assert len(loaded_data) == len(sample_data) - len(sample_erasure_list)
    assert preprocessed_data is not None
    assert "features" in split_data and "target" in split_data


def test_data_science_splits(sample_data, model_parameters):
    # Only test the data splitting functionality
    features = pd.DataFrame(sample_data.drop('y', axis=1))
    target = pd.DataFrame(sample_data['y'])
    
    split_result = split_data_science(features, target, model_parameters)
    
    assert "X_train" in split_result
    assert "X_test" in split_result
    assert "y_train" in split_result
    assert "y_test" in split_result
    assert len(split_result["X_train"]) > 0
    assert len(split_result["X_test"]) > 0