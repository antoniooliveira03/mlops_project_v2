from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    drop,
    normalize_column_names,
    rename_columns,
    create_arrivaltime_feature,
    merge_target_with_datetime
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        # Normalize column names
        node(
            func=normalize_column_names,
            inputs=["X_train_validated", "y_train_validated"],
            outputs=["X_train_normalized", "y_train_normalized"],
            name="normalize_columns_train"
        ),
        node(
            func=normalize_column_names,
            inputs=["X_val_validated", "y_val_validated"],
            outputs=["X_val_normalized", "y_val_normalized"],
            name="normalize_columns_val"
        ),

        # Rename columns
        node(
            func=rename_columns,
            inputs="X_train_validated",
            outputs="X_train_renamed",
            name="rename_columns_train"
        ),
        node(
            func=rename_columns,
            inputs="X_val_validated",
            outputs="X_val_renamed",
            name="rename_columns_val"
        ),

        # Drop unnecessary columns
        node(
            func=drop,
            inputs="X_train_validated",
            outputs="X_train_clean",
            name="drop_train"
        ),
        node(
            func=drop,
            inputs="X_val_validated",
            outputs="X_val_clean",
            name="drop_val"
        ),

        # Create datetime features
        node(
            func=create_arrivaltime_feature,
            inputs="X_train_validated",
            outputs="X_train_with_datetime",
            name="create_datetime_train"
        ),
        node(
            func=create_arrivaltime_feature,
            inputs="X_val_validated",
            outputs="X_val_with_datetime",
            name="create_datetime_val"
        ),

        # Merge y with datetime info
        node(
            func=merge_target_with_datetime,
            inputs=["y_train_validated", "X_train_validated"],
            outputs="y_train_final",
            name="merge_datetime_train"
        ),
        node(
            func=merge_target_with_datetime,
            inputs=["y_val_validated", "X_val_validated"],
            outputs="y_val_final",
            name="merge_datetime_val"
        ),
    ])
