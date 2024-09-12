import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import json

# Load the dataset
df = pd.read_csv('data/01_raw/dataset_id_96.csv')

def preprocess_data(df):
    
    expected_features = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 
                         'existing_credits', 'num_dependents', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 
                         'X_7', 'X_8', 'X_9', 'X_10', 'checking_status_0<=X<200', 'checking_status_<0', 
                         'checking_status_>=200', 'checking_status_no checking', 'credit_history_all paid', 
                         'credit_history_critical/other existing credit', 'credit_history_delayed previously', 
                         'credit_history_existing paid', 'credit_history_no credits/all paid', 
                         'purpose_business', 'purpose_domestic appliance', 'purpose_education', 
                         'purpose_furniture/equipment', 'purpose_new car', 'purpose_radio/tv', 
                         'purpose_repairs', 'purpose_retraining', 'purpose_used car', 
                         'savings_status_100<=X<500', 'savings_status_500<=X<1000', 'savings_status_<100', 
                         'savings_status_>=1000', 'savings_status_no known savings', 'employment_1<=X<4', 
                         'employment_4<=X<7', 'employment_<1', 'employment_>=7', 'employment_unemployed', 
                         'personal_status_female div/dep/mar', 'personal_status_male div/sep', 
                         'personal_status_male mar/wid', 'personal_status_male single', 
                         'other_parties_co applicant', 'other_parties_guarantor', 'other_parties_none', 
                         'property_magnitude_car', 'property_magnitude_life insurance', 
                         'property_magnitude_no known property', 'property_magnitude_real estate', 
                         'other_payment_plans_bank', 'other_payment_plans_none', 'other_payment_plans_stores', 
                         'housing_for free', 'housing_own', 'housing_rent', 
                         'job_high qualif/self emp/mgmt', 'job_skilled', 'job_unemp/unskilled non res', 
                         'job_unskilled resident', 'own_telephone_none', 'own_telephone_yes', 
                         'foreign_worker_no', 'foreign_worker_yes', 'health_status_bad', 'health_status_good']

    numerical_columns = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 
                         'existing_credits', 'num_dependents', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 
                         'X_7', 'X_8', 'X_9', 'X_10']
    categorical_columns = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment',
                           'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans',
                           'housing', 'job', 'own_telephone', 'foreign_worker', 'health_status']

    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Fit and transform the data
    X = df.drop(columns=['y'])
    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    onehot_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
    feature_names = numerical_columns + list(onehot_columns)

    # Create DataFrame with processed features
    processed_df = pd.DataFrame(X_processed, columns=feature_names, index=df.index)

    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in processed_df.columns:
            processed_df[feature] = 0.0

    # Reorder columns to match expected features
    processed_df = processed_df[expected_features]

    return processed_df

# Select a random sample of 5 rows
sample_data = df.sample(n=5, random_state=42)

# Preprocess the sample data
sample_data_preprocessed = preprocess_data(sample_data)

# Print the sample data
print("Sample Data (Preprocessed):")
print(sample_data_preprocessed)

# Prepare the data for MLflow model server (MLflow 2.0+ format)
input_data = {
    "dataframe_split": {
        "columns": sample_data_preprocessed.columns.tolist(),
        "data": sample_data_preprocessed.values.tolist()
    }
}

# Print the input data for debugging
print("\nInput data sent to the model:")
print(json.dumps(input_data, indent=2))

# Make a prediction
predictions = None
try:
    response = requests.post(
        'http://127.0.0.1:5001/invocations',
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
   
    print(f"\nResponse status code: {response.status_code}")
    print("Response headers:")
    print(json.dumps(dict(response.headers), indent=2))
   
    if response.status_code == 200:
        predictions = response.json()
        print("\nPredictions:")
        print(json.dumps(predictions, indent=2))
        
        if isinstance(predictions, dict) and 'predictions' in predictions:
            predictions = predictions['predictions']
        
        if isinstance(predictions, list):
            for i, pred in enumerate(predictions):
                print(f"Sample {i+1}: {pred}")
        else:
            print("Unexpected prediction format:", predictions)
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response content:")
        print(response.text)
except requests.exceptions.RequestException as e:
    print("Error connecting to the server:", e)

# Compare predictions with actual labels
print("\nComparison with actual labels:")
actual_labels = sample_data['y'].tolist()
if predictions and isinstance(predictions, list):
    for i, (pred, actual) in enumerate(zip(predictions, actual_labels)):
        print(f"Sample {i+1}: Predicted: {pred}, Actual: {actual}")
else:
    print("Unable to compare predictions with actual labels due to unexpected prediction format.")

print("\nActual labels from sample:")
for i, actual in enumerate(actual_labels):
    print(f"Sample {i+1}: Actual: {actual}")