import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

    logger = logging.getLogger(__name__)
    logger.info("Model has accuracy of %.3f on test data.", accuracy)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
