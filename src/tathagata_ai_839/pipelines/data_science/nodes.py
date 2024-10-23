import json
import logging
import os
from typing import Dict

import mlflow
import mlflow.sklearn
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import TargetDriftPreset
from evidently.metrics import DataDriftTable
from evidently.report import Report
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

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
    train_data = y_train.rename(columns={target_column: "target"})
    test_data = y_test.rename(columns={target_column: "target"})

    # Set up column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = "target"

    # Generate drift report
    drift_report = Report(metrics=[TargetDriftPreset(), DataDriftTable()])
    drift_report.run(
        reference_data=train_data, current_data=test_data, column_mapping=column_mapping
    )

    # Save the HTML report
    report_path = os.path.join("data", "08_reporting", "target_drift_report.html")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    drift_report.save_html(report_path)

    # Save the JSON report
    json_report_path = os.path.join("data", "08_reporting", "target_drift_report.json")
    with open(json_report_path, "w") as f:
        json.dump(drift_report.json(), f, indent=2)

    # Extract drift information
    report_dict = drift_report.as_dict()
    target_drift_detected = report_dict["metrics"][0]["result"]["drift_detected"]
    target_drift_score = report_dict["metrics"][0]["result"]["drift_score"]

    logger.info(
        f"Target drift detection completed. Drift detected: {target_drift_detected}"
    )
    logger.info(f"Target drift score: {target_drift_score}")
    logger.info(f"Drift report saved as HTML: {report_path}")
    logger.info(f"Drift report saved as JSON: {json_report_path}")

    if target_drift_detected:
        raise ValueError(
            f"Significant target drift detected (score: {target_drift_score}). Pipeline stopped. Check the reports at {report_path} and {json_report_path}"
        )


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
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict
) -> RandomForestClassifier:
    """Trains the random forest model."""
    model = RandomForestClassifier(**parameters["model_params"])
    model.fit(X_train, y_train.values.ravel())

    mlflow.log_params(parameters["model_params"])

    signature = infer_signature(X_train, y_train)

    # Model logging
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="Model",
    )
    return model


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Dict[str, float]:
    """Calculates and logs the accuracy of the model."""
    y_pred = model.predict(X_test)
    y_true = y_test.values.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    logger.info("Model has accuracy of %.3f on test data.", accuracy)

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def create_model_card(model, metrics: Dict) -> None:
    """
    Creates a model card based on the trained model and evaluation metrics.
    """
    model_card = {
        "model_type": type(model).__name__,
        "model_parameters": model.get_params(),
        "evaluation_metrics": metrics,
    }
    return json.dumps(model_card)
