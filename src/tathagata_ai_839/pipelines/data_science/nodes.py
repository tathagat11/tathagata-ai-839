import logging
from typing import Dict, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
from evidently.metrics import DataDriftTable
import json

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logger = logging.getLogger(__name__)

def detect_target_drift(y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Detects target drift between training and test sets.
    
    Args:
        y_train: Target variable from the training set (DataFrame)
        y_test: Target variable from the test set (DataFrame)
    
    Raises:
        ValueError: If significant target drift is detected
    """
    # y_train has only one column but for dimensions
    target_column = y_train.columns[0]  

    # Create DataFrames for drift detection
    train_data = y_train.rename(columns={target_column: 'target'})
    test_data = y_test.rename(columns={target_column: 'target'})

    # Set up column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'

    # Generate drift report
    drift_report = Report(metrics=[
        TargetDriftPreset(),
        DataDriftTable()
    ])
    drift_report.run(reference_data=train_data, current_data=test_data, column_mapping=column_mapping)
    
    # Save the HTML report
    report_path = os.path.join('data', '08_reporting', 'target_drift_report.html')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    drift_report.save_html(report_path)

    # Save the JSON report
    json_report_path = os.path.join('data', '08_reporting', 'target_drift_report.json')
    with open(json_report_path, 'w') as f:
        json.dump(drift_report.json(), f, indent=2)

    # Extract drift information
    report_dict = drift_report.as_dict()
    target_drift_detected = report_dict['metrics'][0]['result']['drift_detected']
    target_drift_score = report_dict['metrics'][0]['result']['drift_score']
    
    logger.info(f"Target drift detection completed. Drift detected: {target_drift_detected}")
    logger.info(f"Target drift score: {target_drift_score}")
    logger.info(f"Drift report saved as HTML: {report_path}")
    logger.info(f"Drift report saved as JSON: {json_report_path}")

    if target_drift_detected:
        raise ValueError(f"Significant target drift detected (score: {target_drift_score}). Pipeline stopped. Check the reports at {report_path} and {json_report_path}")

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


def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    parameters: Dict,
    model_name: str
) -> RandomForestClassifier:
    """Trains the random forest model."""
    model = RandomForestClassifier(**parameters["model_params"])
    model.fit(X_train, y_train.values.ravel())

    with mlflow.start_run(run_name=f"Train_{model_name}", nested=True):
        mlflow.log_params(parameters["model_params"])
        
        signature = infer_signature(X_train, y_train)

        # Model logging
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"model_{model_name.lower()}",
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name=f"Model_{model_name}"
        )
    
    return model

def evaluate_model(
    model: RandomForestClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame,
    model_name: str
) -> Dict[str, float]:
    """Calculates and logs the accuracy of the model."""
    y_pred = model.predict(X_test)
    y_true = y_test.values.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    logger.info(f"{model_name} has accuracy of {accuracy:.3f} on test data.")
    
    with mlflow.start_run(run_name=f"Evaluate_{model_name}", nested=True):
        mlflow.log_metric(f"{model_name.lower()}_test_accuracy", accuracy)
        mlflow.log_metric(f"{model_name.lower()}_test_precision", precision)
        mlflow.log_metric(f"{model_name.lower()}_test_recall", recall)
        mlflow.log_metric(f"{model_name.lower()}_test_f1", f1)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def compare_models(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    model_a: RandomForestClassifier,
    model_b: RandomForestClassifier,
    X_test_new: pd.DataFrame,
    y_test_new: pd.DataFrame
) -> Tuple[RandomForestClassifier, str]:
    """
    Compares two models based on their performance metrics and selects the better one.
    
    Args:
        metrics_a: Performance metrics of Model A
        metrics_b: Performance metrics of Model B
        model_a: Model A (trained on original data)
        model_b: Model B (trained on new data)
        X_test_new: Features of the new test data
        y_test_new: Target of the new test data
    
    Returns:
        A tuple containing the better performing model and its name
    """
    y_pred_a = model_a.predict(X_test_new)
    y_pred_b = model_b.predict(X_test_new)
    y_true = y_test_new.values.ravel()

    # Create contingency table
    table = np.zeros((2, 2))
    table[0, 0] = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
    table[0, 1] = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    table[1, 0] = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    table[1, 1] = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))

    # Perform McNemar's test
    result = mcnemar(table, exact=False, correction=True)

    logger.info(f"McNemar's test statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")

    with mlflow.start_run(run_name="Compare_Models", nested=True):
        mlflow.log_metric("mcnemar_statistic", result.statistic)
        mlflow.log_metric("mcnemar_pvalue", result.pvalue)

        # Select the better model based on accuracy and statistical significance
        if metrics_b['accuracy'] > metrics_a['accuracy'] and result.pvalue < 0.05:
            logger.info("Model B performs significantly better. Selecting Model B.")
            selected_model = model_b
            selected_model_name = "Model_B"
        else:
            logger.info("Model A performs better or there's no significant difference. Selecting Model A.")
            selected_model = model_a
            selected_model_name = "Model_A"

        mlflow.log_param("selected_model", selected_model_name)

    return selected_model, selected_model_name