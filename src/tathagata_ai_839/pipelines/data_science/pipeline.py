from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_model, evaluate_model, detect_target_drift, compare_models


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["features_original", "target_original", "params:model_options"],
                outputs={
                    "X_train": "X_train_original",
                    "X_test": "X_test_original",
                    "y_train": "y_train_original",
                    "y_test": "y_test_original",
                },
                name="split_data_node_original",
            ),
            node(
                func=split_data,
                inputs=["features_new", "target_new", "params:model_options"],
                outputs={
                    "X_train": "X_train_new",
                    "X_test": "X_test_new",
                    "y_train": "y_train_new",
                    "y_test": "y_test_new",
                },
                name="split_data_node_new",
            ),

            node(
                func=train_model,
                inputs=["X_train_original", "y_train_original", "params:model_options", "params:model_a_name"],
                outputs="model_a",
                name="train_model_a_node",
            ),

            node(
                func=train_model,
                inputs=["X_train_new", "y_train_new", "params:model_options", "params:model_b_name"],
                outputs="model_b",
                name="train_model_b_node",
            ),

            node(
                func=evaluate_model,
                inputs=["model_a", "X_test_new", "y_test_new", "params:model_a_name"],
                outputs="metrics_a_new",
                name="evaluate_model_a_new_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model_b", "X_test_new", "y_test_new", "params:model_b_name"],
                outputs="metrics_b_new",
                name="evaluate_model_b_new_node",
            ),

            node(
                func=compare_models,
                inputs=["metrics_a_new", "metrics_b_new", "model_a", "model_b", "X_test_new", "y_test_new"],
                outputs=["selected_model", "selected_model_name"],
                name="compare_models_node",
            ),
        ]
    )
