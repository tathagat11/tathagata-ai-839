import requests
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import Dict

# Load the dataset
df = pd.read_csv('data/01_raw/dataset_id_96.csv')

def identify_categorical_columns(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

def identify_numerical_columns(df):
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()

def preprocess_data(df):
    # Identify categorical and numerical columns
    categorical_columns = identify_categorical_columns(df)
    numerical_columns = identify_numerical_columns(df)

    # Remove the target variable from feature columns
    target_column = "y"
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    # Create preprocessing steps
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
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
    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    onehot_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
    feature_names = numerical_columns + list(onehot_cols)

    # Create a new DataFrame with processed features
    processed_df = pd.DataFrame(X_processed, columns=feature_names, index=df.index)

    return processed_df

def split_data(
    features: pd.DataFrame, target: pd.Series, parameters: Dict
) -> Dict[str, pd.DataFrame]:
    """Splits data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

# Preprocess the full dataset
X = preprocess_data(df)
y = df['y']

# Split the data
parameters = {"test_size": 0.2, "random_state": 20}
data_split = split_data(X, y, parameters)

# Select a random sample of 5 rows from the test set
sample_data = data_split['X_test'].sample(n=5, random_state=42)

# Print the sample data
print("Sample Data (Preprocessed):")
print(sample_data)

# MLFlow Data input
input_data = {
    "dataframe_split": {
        "columns": sample_data.columns.tolist(),
        "data": sample_data.values.tolist()
    }
}

# Make a prediction
predictions = None
try:
    response = requests.post(
        'http://127.0.0.1:5001/invocations',
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
   
    print(f"\nResponse status code: {response.status_code}")
   
    if response.status_code == 200:
        predictions = response.json()
        
        if isinstance(predictions, dict) and 'predictions' in predictions:
            predictions = predictions['predictions']
        
        if not isinstance(predictions, list):
            print("Unexpected prediction format:", predictions)
            
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response content:")
        print(response.text)
except requests.exceptions.RequestException as e:
    print("Error connecting to the server:", e)

# Compare predictions with actual labels
print("\nComparison with actual labels:")
actual_labels = y[sample_data.index].tolist()
if predictions and isinstance(predictions, list):
    for i, (pred, actual) in enumerate(zip(predictions, actual_labels)):
        print(f"Sample {i+1}: Predicted: {pred}, Actual: {actual}")
else:
    print("Unable to compare predictions with actual labels due to unexpected prediction format.")