import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Union, Optional, Dict
import mlflow
import json

import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('model_usage.log')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

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
        start_time = datetime.now()

        if isinstance(input_data, InputData):
            df = pd.DataFrame([input_data.dict()])
            batch_size = 1
        else:
            df = pd.DataFrame([item.dict() for item in input_data.data])
            batch_size = len(input_data.data)
        
        processed_df = preprocess_data(df)
        
        # Make predictions
        predictions = model.predict(processed_df)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        log_data = {
            "timestamp": start_time.isoformat(),
            "batch_size": batch_size,
            "processing_time": processing_time,
            "input_shape": df.shape,
            "output_shape": predictions.shape if hasattr(predictions, 'shape') else (len(predictions),),
        }
        logger.info(f"Model Usage: {json.dumps(log_data)}")
        
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_log_line(line: str) -> Dict:
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        parts = line.strip().split(' - ', 3)
        if len(parts) == 4:
            return {
                "timestamp": parts[0],
                "name": parts[1],
                "levelname": parts[2],
                "message": parts[3]
            }
        else:
            return {"raw": line.strip()}

@app.get("/logs")
async def get_logs(
    limit: int = Query(100, description="Number of log entries to return"),
    level: Optional[str] = Query(None, description="Log level filter (INFO, ERROR, etc.)"),
    start_date: Optional[str] = Query(None, description="Start date for log filtering (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for log filtering (YYYY-MM-DD)")
) -> List[Dict]:
    try:
        log_entries = []
        parsing_errors = []
        with open('model_usage.log', 'r') as log_file:
            for line_number, line in enumerate(log_file, 1):
                try:
                    log_entry = parse_log_line(line)
                    
                    # Apply filters
                    if level and log_entry.get('levelname', '').upper() != level.upper():
                        continue
                    
                    log_date = None
                    if 'timestamp' in log_entry:
                        try:
                            log_date = datetime.fromisoformat(log_entry['timestamp'].split()[0]).date()
                        except ValueError:
                            pass
                    
                    if start_date and log_date:
                        if log_date < datetime.strptime(start_date, "%Y-%m-%D").date():
                            continue
                    
                    if end_date and log_date:
                        if log_date > datetime.strptime(end_date, "%Y-%m-%D").date():
                            continue
                    
                    log_entries.append(log_entry)
                    
                    if len(log_entries) >= limit:
                        break
                except Exception as e:
                    parsing_errors.append(f"Error parsing line {line_number}: {str(e)}")
        
        if parsing_errors:
            return {
                "log_entries": log_entries,
                "parsing_errors": parsing_errors
            }
        return log_entries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)