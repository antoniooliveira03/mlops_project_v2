from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_expectations_from_training, validate_and_save



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_expectations_from_training,
            inputs=["X_train_data", "y_train_data"],
            outputs="suite_name",
            name="generate_expectation_suite_node"
        ),
        node(
            func=validate_and_save,
            inputs=["X_val_data", "y_val_data", "suite_name"],
            outputs="validation_results",
            name="validate_val_data_node"
        ),
    ])

