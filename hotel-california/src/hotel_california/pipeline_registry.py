from typing import Dict
from kedro.pipeline import Pipeline


from project_template.pipelines import (
    split as data_split,

)

def register_pipelines() -> Dict[str, Pipeline]:

    data_split = data_split.create_pipeline()
    # ...

    return {
        "split": data_split,
    }