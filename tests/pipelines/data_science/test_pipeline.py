# tests/pipelines/data_science/test_pipeline.py

import pytest
import pandas as pd
import numpy as np
from tathagata_ai_839.pipelines.data_science.pipeline import create_pipeline
from tathagata_ai_839.pipelines.data_science.nodes import split_data, train_model, evaluate_model

@pytest.fixture
def sample_data():
    np.random.seed(42)
    features = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    target = pd.DataFrame({'target': np.random.randint(0, 2, 100)})
    return features, target

@pytest.fixture
def model_parameters():
    return {
        "test_size": 0.2,
        "random_state": 42,
        "model_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    }

def test_data_science_pipeline(sample_data, model_parameters):
    features, target = sample_data
    
    # Create the pipeline
    pipeline = create_pipeline()

    # Step 1: Split data
    split_result = split_data(features, target, model_parameters)
    assert len(split_result["X_train"]) == 80
    assert len(split_result["X_test"]) == 20
    
    # Step 2: Train model
    model = train_model(split_result["X_train"], split_result["y_train"], model_parameters)
    assert model is not None
    
    # Step 3: Evaluate model
    metrics = evaluate_model(model, split_result["X_test"], split_result["y_test"])
    assert all(0 <= value <= 1 for value in metrics.values())
    
    print("Data science pipeline test passed successfully!")