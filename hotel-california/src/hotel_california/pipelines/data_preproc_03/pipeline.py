from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_features,
    prepare_target
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_features,
            inputs="X_train_validated",
            outputs="X_train_prepared",
            name="prepare_features_train"
        ),
        node(
            func=prepare_features,
            inputs="X_val_validated",
            outputs="X_val_prepared",
            name="prepare_features_val"
        ),
        node(
            func=prepare_target,
            inputs=["y_train_validated", "X_train_prepared"],
            outputs="y_train_prepared",
            name="prepare_target_train"
        ),
        node(
            func=prepare_target,
            inputs=["y_val_validated", "X_val_prepared"],
            outputs="y_val_prepared",
            name="prepare_target_val"
        ),
    ])