from kedro.pipeline import Pipeline, node, pipeline
from .nodes import unit_test, unit_test_y



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=unit_test,
            inputs={"df": "X_train_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="unit_test_result_train",
            name="unit_test_node_train"
        ),
        node(
            func=unit_test,
            inputs={"df": "X_val_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="unit_test_result_val",
            name="unit_test_node_val"
        ),
        node(
            func=unit_test_y,
            inputs={"y": "y_train_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="unit_test_result_train_y",
            name="unit_test_node_train_y"
        ),
        node(
            func=unit_test_y,
            inputs={"y": "y_val_data",
                    "mlruns_path": "params:mlruns_path"},
            outputs="unit_test_result_val_y",
            name="unit_test_node_val_y"
        ),
    ])

