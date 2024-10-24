import pytest
from kedro.pipeline import Pipeline
from tathagata_ai_839.pipelines.data_science.pipeline import create_pipeline


def test_create_pipeline():
    pipeline = create_pipeline()
    
    # Test that a valid pipeline is created
    assert isinstance(pipeline, Pipeline)
    
    # Test pipeline structure
    nodes = pipeline.nodes
    
    # Check number of nodes
    assert len(nodes) == 5
    
    # List of expected nodes with their exact configuration
    expected_nodes = [
        {
            'name': 'test_train_split_data',
            'inputs': ['features', 'target', 'params:model_options'],
            'outputs': ['X_train', 'X_test', 'y_train', 'y_test']
        },
        {
            'name': 'target_drift_detection',
            'inputs': ['y_train', 'y_test'],
            'outputs': []  # Kedro converts None to empty list
        },
        {
            'name': 'train_model',
            'inputs': ['X_train', 'y_train', 'params:model_options'],
            'outputs': ['model']  # Kedro converts string to single-item list
        },
        {
            'name': 'evaluate_model',
            'inputs': ['model', 'X_test', 'y_test'],
            'outputs': ['metrics']
        },
        {
            'name': 'create_model_card',
            'inputs': ['model', 'metrics'],
            'outputs': ['model_card']
        }
    ]
    
    # Verify each node matches expected configuration
    for idx, (node, expected) in enumerate(zip(nodes, expected_nodes)):
        assert node.name == expected['name'], f"Node {idx} name mismatch"
        assert set(node.inputs) == set(expected['inputs']), f"Node {idx} inputs mismatch"
        assert set(node.outputs) == set(expected['outputs']), f"Node {idx} outputs mismatch"