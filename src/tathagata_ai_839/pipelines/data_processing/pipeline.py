from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load_and_erase_data, preprocess_data, split_data, run_data_quality_checks, create_data_card

def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data processing pipeline.

    Args:
        **kwargs: Additional keyword arguments passed to the pipeline constructor

    Returns:
        Pipeline: The data processing pipeline
    """
    return pipeline(
        [
            node(
                func=load_and_erase_data,
                inputs=["dataset_id_96", "erasure_list"],
                outputs="loaded_data",
                name="load_data",
            ),
            node(
                func=run_data_quality_checks,
                inputs="loaded_data",
                outputs="data_quality_metrics",
                name="run_data_quality_checks",
            ),
            node(
                func=preprocess_data,
                inputs="loaded_data",
                outputs="preprocessed_data",
                name="preprocess_data",
            ),
            node(
                func=split_data,
                inputs="preprocessed_data",
                outputs={"features": "features", "target": "target"},
                name="split_data",
            ),
            node(
                func=create_data_card,
                inputs=["loaded_data", "data_quality_metrics"],
                outputs="data_card",
                name="create_data_card",
            )
        ]
    )