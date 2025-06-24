from kedro.pipeline import Pipeline, node, pipeline
from .nodes import ingestion

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=dict(
                    df="X_train_final",
                    parameters="parameters",
                    target_df="y_train_final",
                    prefix="params:train_prefix",
                ),
                outputs=["train_ingested", "train_target_ingested"],
                name="ingest_train_data",
            ),
            node(
                func=ingestion,
                inputs=dict(
                    df="X_val_final",
                    parameters="parameters",
                    target_df="y_val_final",
                    prefix="params:val_prefix",
                ),
                outputs=["validation_ingested", "val_target_ingested"],
                name="ingest_validation_data",
            ),
        ]
    )
