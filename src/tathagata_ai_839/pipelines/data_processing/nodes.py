import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from .data_quality import generate_data_quality_report, get_data_quality_metrics
import logging

logger = logging.getLogger(__name__)

def load_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Load the data from the CSV file.

    Args:
        data: Raw DataFrame loaded by Kedro

    Returns:
        Loaded DataFrame
    """
    return data


def run_data_quality_checks(df: pd.DataFrame) -> Dict[str, dict]:
    """
    Run data quality checks on the input data.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary containing key metrics
    """
    try:
        generate_data_quality_report(df)
        metrics = get_data_quality_metrics(df)
    except Exception as e:
        logger.error(f"Error in generating data quality report: {str(e)}")
        metrics = {"error": str(e)}
    
    return {"data_quality_metrics": metrics}


def identify_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify categorical columns in the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        List of categorical column names
    """
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def identify_numerical_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify numerical columns in the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        List of numerical column names
    """
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling categorical variables, scaling numerical variables,
    and separating the target variable.

    Args:
        df: Raw DataFrame

    Returns:
        Preprocessed DataFrame
    """
    # Identify categorical and numerical columns
    categorical_columns = identify_categorical_columns(df)
    numerical_columns = identify_numerical_columns(df)

    # Remove the target variable from feature columns
    target_column = "y"
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    # Create preprocessing steps
    categorical_transformer = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore"
    )
    numerical_transformer = StandardScaler()

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    # Fit and transform the data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    onehot_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_columns
    )
    feature_names = numerical_columns + list(onehot_cols)

    # Create a new DataFrame with processed features
    processed_df = pd.DataFrame(X_processed, columns=feature_names, index=df.index)

    # Add the target variable back to the DataFrame
    processed_df[target_column] = y

    return processed_df


def split_data(df: pd.DataFrame, target_column: str = "y") -> Dict[str, pd.DataFrame]:
    """
    Split the data into features and target.

    Args:
        df: Preprocessed DataFrame
        target_column: Name of the target column

    Returns:
        Dictionary containing 'features' and 'target' as DataFrames
    """
    X = df.drop(columns=[target_column])
    y = df[[target_column]]  # Keep as DataFrame instead of Series
    return {"features": X, "target": y}
