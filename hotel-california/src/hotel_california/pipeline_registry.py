from typing import Dict
from kedro.pipeline import Pipeline


from hotel_california.pipelines import (
    data_split_01 as data_split,
    data_unit_tests_02 as data_unit_tests,
    data_preproc_03 as data_preproc,
    model_selection_0x as model_selection

)

def register_pipelines() -> Dict[str, Pipeline]:

    data_split_pipeline = data_split.create_pipeline()
    data_unit_tests_pipeline = data_unit_tests.create_pipeline()  
    data_preproc_pipeline = data_preproc.create_pipeline()
    model_selection_pipeline = model_selection.create_pipeline()
    # ...

    return {
        "split": data_split_pipeline,
        "unit_tests": data_unit_tests_pipeline,
        "preproc": data_preproc_pipeline,
        "model_selection": model_selection_pipeline,
        "__default__": data_split_pipeline + data_unit_tests_pipeline + data_preproc_pipeline,
        
    }