from kedro.pipeline import Pipeline, node, pipeline
from .nodes import unit_test_final

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=unit_test_final,
            inputs={"df": "X_train_final",
                    "mlruns_path": "params:mlruns_path"},
            outputs="X_train_validated_final",
            name="unit_test_node_train_final"
        ),
        node(
            func=unit_test_final,
            inputs={"df": "X_val_final",  
                    "mlruns_path": "params:mlruns_path"},
            outputs="X_val_validated_final",
            name="unit_test_node_val_final"
        )
    ])
