"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["X_train_final","X_val_final","y_train_final","y_val_final",
                        "params:target",
                        "production_model_metrics",
                        "production_model",
                        "params:hyperparameters",
                        "selected_features"],
                outputs=["champion_model", "champion_columns" ,"champion_model_metrics"],
                name="model_selection",
            ),
        ]
    )
