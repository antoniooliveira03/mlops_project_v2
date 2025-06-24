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
            outputs="X_train_final",
            name="prepare_features_train"
        ),
        node(
            func=prepare_features,
            inputs="X_val_validated",
            outputs="X_val_final",
            name="prepare_features_val"
        ),
        node(
            func=prepare_features,
            inputs="X_test_validated",
            outputs="X_test_final",
            name="prepare_features_test"
        ),
        node(
            func=prepare_target,
            inputs=["y_train_validated", "X_train_final"],
            outputs="y_train_final",
            name="prepare_target_train"
        ),
        node(
            func=prepare_target,
            inputs=["y_val_validated", "X_val_final"],
            outputs="y_val_final",
            name="prepare_target_val"
        ),
        node(
            func=prepare_target,
            inputs=["y_test_validated", "X_test_final"],
            outputs="y_test_final",
            name="prepare_target_test"
        ),
    ])