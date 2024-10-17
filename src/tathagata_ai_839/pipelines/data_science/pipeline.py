from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_model, evaluate_model, detect_target_drift, create_model_card


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data science pipeline.

    Args:
        **kwargs: Additional keyword arguments passed to the pipeline constructor

    Returns:
        Pipeline: The data science pipeline
    """
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
                name="test_train_split_data",
            ),
            node(
                func=detect_target_drift,
                inputs=["y_train", "y_test"],
                outputs=None,
                name="target_drift_detection",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="model",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model",
            ),
            node(
                func=create_model_card,
                inputs=["model", "metrics"],
                outputs="model_card",
                name="create_model_card",
            )
        ]
    )
