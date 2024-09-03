import pytest
import pandas as pd
from tathagata_ai_839.pipelines.data_processing.nodes import load_data, preprocess_data, split_data

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

def test_data_processing_pipeline(sample_data):
    # Test load_data
    loaded_data = load_data(sample_data)
    assert loaded_data is not None
    assert len(loaded_data) == len(sample_data)

    # Test preprocess_data
    preprocessed_data = preprocess_data(loaded_data)
    assert preprocessed_data is not None
    assert len(preprocessed_data) == len(sample_data)
    assert 'checking_status_no checking' in preprocessed_data.columns
    assert 'purpose_business' in preprocessed_data.columns
    assert 'duration' in preprocessed_data.columns
    assert 'credit_amount' in preprocessed_data.columns

    # Test split_data
    split_result = split_data(preprocessed_data)
    features = split_result['features']
    target = split_result['target']
    
    assert features is not None
    assert target is not None
    assert len(features) == len(sample_data)
    assert len(target) == len(sample_data)
    assert 'y' not in features.columns
    assert 'y' in target.columns

    print("All tests passed successfully!")

if __name__ == "__main__":
    pytest.main([__file__])