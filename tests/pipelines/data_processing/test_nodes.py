import pandas as pd
import pytest

from src.tathagata_ai_839.pipelines.data_processing.nodes import (
    identify_categorical_columns,
    identify_numerical_columns,
    load_data,
    preprocess_data,
    split_data,
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


def test_load_data(sample_data):
    result = load_data(sample_data)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_data)


def test_identify_categorical_columns(sample_data):
    cat_columns = identify_categorical_columns(sample_data)
    expected_cat_columns = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker",
        "health_status",
    ]
    assert set(cat_columns) == set(expected_cat_columns)


def test_identify_numerical_columns(sample_data):
    num_columns = identify_numerical_columns(sample_data)
    expected_num_columns = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]
    assert set(num_columns) == set(expected_num_columns)


def test_preprocess_data(sample_data):
    processed_df = preprocess_data(sample_data)

    # Check if the result is a DataFrame
    assert isinstance(processed_df, pd.DataFrame)

    # Check if all columns except 'y' are numerical (after one-hot encoding and scaling)
    assert all(processed_df.drop("y", axis=1).dtypes != "object")

    # Check if 'y' column is preserved
    assert "y" in processed_df.columns

    # Check if the number of rows is preserved
    assert len(processed_df) == len(sample_data)

    # Check if categorical columns are one-hot encoded
    assert "checking_status_no checking" in processed_df.columns
    assert "purpose_business" in processed_df.columns

    # Check if numerical columns are present (they should be scaled)
    assert "duration" in processed_df.columns
    assert "credit_amount" in processed_df.columns


def test_split_data(sample_data):
    preprocessed_data = preprocess_data(sample_data)
    split = split_data(preprocessed_data)

    assert isinstance(split, dict)
    assert "features" in split
    assert "target" in split
    assert isinstance(split["features"], pd.DataFrame)
    assert isinstance(split["target"], pd.DataFrame)
    assert "y" not in split["features"].columns
    assert "y" in split["target"].columns
    assert len(split["features"]) == len(split["target"])
