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
                    target_df="y_train_final"
                ),
                outputs="train_ingested",
                name="ingest_train_data",
            ),
            node(
                func=ingestion,
                inputs=dict(
                    df="X_val_final",
                    parameters="parameters",
                    target_df="y_val_final"
                ),
                outputs="validation_ingested",
                name="ingest_validation_data",
            ),
        ]
    )