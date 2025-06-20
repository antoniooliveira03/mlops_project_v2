from kedro.pipeline import Pipeline, node, pipeline
from .nodes import drop

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=drop,
            inputs="X_train_validated",
            outputs="X_train_clean",
            name="drop_company_reservation_train"
        ),
        node(
            func=drop,
            inputs="X_val_validated",
            outputs="X_val_clean",
            name="drop_company_reservation_val"
        ),
    ])

