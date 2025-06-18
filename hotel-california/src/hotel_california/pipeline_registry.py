from typing import Dict
from kedro.pipeline import Pipeline


from hotel_california.pipelines import (
    data_split_01 as data_split,

)

def register_pipelines() -> Dict[str, Pipeline]:

    data_split_pipeline = data_split.create_pipeline()
    # ...

    return {
        "data_split_01": data_split_pipeline,
    }