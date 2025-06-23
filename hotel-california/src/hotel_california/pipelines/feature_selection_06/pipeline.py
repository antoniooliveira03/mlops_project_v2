from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_features_node, feature_selection_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_features_node,
            inputs=None,
            outputs=["categorical_features", "numerical_features"],
            name="load_features_node"
        ),
        node(
            func=feature_selection_node,
            inputs=[
                "X_train_final",
                "params:mlruns_path",
                "X_train_final",
                "y_train_final",
                "categorical_features",
                "numerical_features",
                "params:feature_selection"
            ],
            outputs="selected_features",
            name="feature_selection_node"
        )
    ])