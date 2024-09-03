from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_model, evaluate_model, detect_target_drift


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["features", "target", "params:model_options"],
                outputs={
                    "X_train": "X_train",
                    "X_test": "X_test",
                    "y_train": "y_train",
                    "y_test": "y_test",
                },
                name="split_data_node",
            ),
            node(
                func=detect_target_drift,
                inputs=["y_train", "y_test"],
                outputs=None,
                name="target_drift_detection_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )
