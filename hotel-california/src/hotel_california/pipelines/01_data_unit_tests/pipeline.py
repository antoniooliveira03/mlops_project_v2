from kedro.pipeline import Pipeline, node, pipeline
from .nodes import  unit_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=unit_test,
                inputs= "Train_Data_Project_MLOPS",
                outputs= "",
                name= "unit_data_test",
            ),
        ]
    )


