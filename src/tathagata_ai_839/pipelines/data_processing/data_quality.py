import os

import pandas as pd
from evidently.metric_preset import DataQualityPreset
from evidently.metrics import DatasetCorrelationsMetric
from evidently.report import Report


def generate_data_quality_report(data: pd.DataFrame) -> None:
    """
    Generate a data quality report using Evidently and save it as HTML at a constant location.

    Args:
        data: Input DataFrame
    """
    data_quality_report = Report(
        metrics=[DataQualityPreset(), DatasetCorrelationsMetric()]
    )

    data_quality_report.run(current_data=data, reference_data=None)

    # Save the report at a constant location
    report_path = os.path.join("data", "08_reporting", "data_quality_report.html")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    data_quality_report.save_html(report_path)


def get_data_quality_metrics(data: pd.DataFrame) -> dict:
    """
    Extract key metrics from the data.

    Args:
        data: Input DataFrame

    Returns:
        Dictionary containing key data quality metrics
    """
    metrics = {}

    metrics["num_features"] = len(data.columns)
    metrics["num_rows"] = len(data)
    metrics["missing_values"] = data.isnull().sum().sum() / (
        data.shape[0] * data.shape[1]
    )

    return metrics
