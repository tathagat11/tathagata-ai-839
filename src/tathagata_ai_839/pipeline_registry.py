from typing import Dict

from kedro.pipeline import Pipeline

from tathagata_ai_839.pipelines import data_processing as dp
from tathagata_ai_839.pipelines import data_science as ds

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "__default__": data_processing_pipeline + data_science_pipeline,
    }
