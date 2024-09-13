import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Dict
import mlflow

app = FastAPI()

class InputData(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: float
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str
    health_status: str
    X_1: float
    X_2: float
    X_3: float
    X_4: float
    X_5: float
    X_6: float
    X_7: float
    X_8: float
    X_9: float
    X_10: float

class BatchInputData(BaseModel):
    data: List[InputData]

# Define categorical and numerical columns
categorical_columns = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment',
                       'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans',
                       'housing', 'job', 'own_telephone', 'foreign_worker', 'health_status']
numerical_columns = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age',
                     'existing_credits', 'num_dependents', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6',
                     'X_7', 'X_8', 'X_9', 'X_10']

def create_preprocessor():
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    
    return preprocessor

# Load the model and create the preprocessor
model = mlflow.pyfunc.load_model("/models")
preprocessor = create_preprocessor()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Preprocess the data
    X_processed = preprocessor.fit_transform(df)
    
    # Get feature names
    onehot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
    feature_names = numerical_columns + list(onehot_cols)
    
    # Create DataFrame with correct column names
    processed_df = pd.DataFrame(X_processed, columns=feature_names, index=df.index)
    
    # Ensure all expected columns are present in the processed DataFrame
    expected_columns = model.metadata.get_input_schema().input_names()
    for col in expected_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0.0  # Add missing columns with default value 0.0
    
    # Reorder columns to match the expected order
    processed_df = processed_df.reindex(columns=expected_columns)
    
    # Convert all columns to float64
    processed_df = processed_df.astype(float)
    
    return processed_df

@app.post("/predict")
async def predict(input_data: Union[InputData, BatchInputData]):
    try:
        if isinstance(input_data, InputData):
            df = pd.DataFrame([input_data.dict()])
        else:
            df = pd.DataFrame([item.dict() for item in input_data.data])
        
        processed_df = preprocess_data(df)
        
        print("Processed DataFrame:")
        print(processed_df.dtypes)
        print(processed_df)
        
        # Make predictions
        predictions = model.predict(processed_df)
        
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)