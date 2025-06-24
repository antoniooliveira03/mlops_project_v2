from kedro.pipeline import Pipeline, node, pipeline
from .nodes import unit_test, unit_test_y



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=unit_test,
            inputs={"df": "X_train_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="X_train_validated",
            name="unit_test_node_train"
        ),
        node(
            func=unit_test,
            inputs={"df": "X_val_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="X_val_validated",
            name="unit_test_node_val"
        ),
        node(
            func=unit_test,
            inputs={"df": "X_test_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="X_test_validated",
            name="unit_test_node_test"
        ),
        node(
            func=unit_test_y,
            inputs={"y": "y_train_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="y_train_validated",
            name="unit_test_node_train_y"
        ),
        node(
            func=unit_test_y,
            inputs={"y": "y_val_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="y_val_validated",
            name="unit_test_node_val_y"
        ),
        node(
            func=unit_test_y,
            inputs={"y": "y_test_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="y_test_validated",
            name="unit_test_node_test_y"
        )
    ])

