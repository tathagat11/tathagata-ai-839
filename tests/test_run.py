# tests/test_run.py

import pytest
import pandas as pd
import numpy as np
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

from tathagata_ai_839.pipelines.data_processing.nodes import load_data, preprocess_data, split_data as split_data_processing
from tathagata_ai_839.pipelines.data_science.nodes import split_data as split_data_science, train_model, evaluate_model

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'checking_status': ['no checking', '0<=X<200', 'no checking'],
        'duration': [9, 18, 12],
        'credit_history': ['existing paid', 'existing paid', 'no credits/all paid'],
        'purpose': ['business', 'business', 'furniture/equipment'],
        'credit_amount': [1449.0, 1941.0, 2759.0],
        'savings_status': ['<100', '>=1000', '<100'],
        'employment': ['4<=X<7', '1<=X<4', '>=7'],
        'installment_commitment': [3, 4, 2],
        'personal_status': ['female div/dep/mar', 'male single', 'male single'],
        'other_parties': ['none', 'none', 'none'],
        'residence_since': [2, 2, 4],
        'property_magnitude': ['car', 'life insurance', 'life insurance'],
        'age': [27, 35, 34],
        'other_payment_plans': ['none', 'none', 'none'],
        'housing': ['own', 'own', 'own'],
        'existing_credits': [2, 1, 2],
        'job': ['skilled', 'unskilled resident', 'skilled'],
        'num_dependents': [1, 1, 1],
        'own_telephone': ['none', 'yes', 'none'],
        'foreign_worker': ['yes', 'yes', 'yes'],
        'health_status': ['good', 'good', 'good'],
        'y': [True, False, False]
    })

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

def test_full_run(sample_data, model_parameters):
    # Data Processing Pipeline
    loaded_data = load_data(sample_data)
    preprocessed_data = preprocess_data(loaded_data)
    split_data = split_data_processing(preprocessed_data)
    
    features = split_data['features']
    target = split_data['target']
    
    # Data Science Pipeline
    split_result = split_data_science(features, target, model_parameters)
    model = train_model(split_result["X_train"], split_result["y_train"], model_parameters)
    metrics = evaluate_model(model, split_result["X_test"], split_result["y_test"])
    
    # Assertions
    assert loaded_data is not None
    assert preprocessed_data is not None
    assert 'features' in split_data and 'target' in split_data
    assert model is not None
    assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1'])
    assert all(0 <= value <= 1 for value in metrics.values())
    
    print("Full project run test passed successfully!")
    print(f"Model metrics: {metrics}")

def test_kedro_run():
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        session.run()
    
    print("Kedro project run completed successfully!")

if __name__ == "__main__":
    pytest.main([__file__])