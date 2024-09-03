from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load_data, preprocess_data, split_data, run_data_quality_checks

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs="dataset_id_96",
                outputs="loaded_data",
                name="load_data",
            ),
            node(
                func=run_data_quality_checks,
                inputs="loaded_data",
                outputs={"data_quality_metrics": "data_quality_metrics"},
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
        ]
    )