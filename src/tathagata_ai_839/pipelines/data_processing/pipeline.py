from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load_data, preprocess_data, split_data, run_data_quality_checks

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            #Old Data:
            node(
                func=load_data,
                inputs="dataset_id_96",
                outputs="loaded_data_original",
                name="load_data_original",
            ),
            node(
                func=run_data_quality_checks,
                inputs="loaded_data_original",
                outputs="data_quality_metrics_original",
                name="run_data_quality_checks_original",
            ),
            node(
                func=preprocess_data,
                inputs="loaded_data_original",
                outputs="preprocessed_data_original",
                name="preprocess_data_original",
            ),
            node(
                func=split_data,
                inputs="preprocessed_data_original",
                outputs={"features": "features_original", "target": "target_original"},
                name="split_data_original",
            ),

            #New Data:
            node(
                func=load_data,
                inputs="dataset_id_T01_V3_96",
                outputs="loaded_data_new",
                name="load_data_new",
            ),
            node(
                func=run_data_quality_checks,
                inputs="loaded_data_new",
                outputs="data_quality_metrics_new",
                name="run_data_quality_checks_new",
            ),
            node(
                func=preprocess_data,
                inputs="loaded_data_new",
                outputs="preprocessed_data_new",
                name="preprocess_data_new",
            ),
            node(
                func=split_data,
                inputs="preprocessed_data_new",
                outputs={"features": "features_new", "target": "target_new"},
                name="split_data_new",
            ),
        ]
    )