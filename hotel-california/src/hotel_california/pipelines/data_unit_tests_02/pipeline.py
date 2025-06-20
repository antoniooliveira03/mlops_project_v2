from kedro.pipeline import Pipeline, node, pipeline
from .nodes import unit_test



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=unit_test,
            inputs="X_train_data",
            outputs="unit_test_result_train",
            name="unit_test_node_train"
        ),
        node(
            func=unit_test,
            inputs="X_train_data",
            outputs="unit_test_result_val",
            name="unit_test_node_val"
        ),
    ])

