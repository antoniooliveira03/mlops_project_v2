from typing import Dict
from kedro.pipeline import Pipeline


from hotel_california.pipelines import (
    data_split_01 as data_split,
    data_unit_tests_02 as data_unit_tests,
    data_preproc_03 as data_preproc,
    data_unit_tests_afterprep_04 as data_unit_tests_afterprep,
    ingestion_05 as ingestion,

)

def register_pipelines() -> Dict[str, Pipeline]:

    data_split_pipeline = data_split.create_pipeline()
    data_unit_tests_pipeline = data_unit_tests.create_pipeline()  
    data_preproc_pipeline = data_preproc.create_pipeline()
    data_unit_tests_afterprep_pipeline = data_unit_tests_afterprep.create_pipeline()
    ingestion_pipeline = ingestion.create_pipeline() 
    # ...

    return {
        "split": data_split_pipeline,
        "unit_tests": data_unit_tests_pipeline,
        "preproc": data_preproc_pipeline,
        "unit_tests_afterprep": data_unit_tests_afterprep_pipeline,
        #"ingestion": ingestion_pipeline,
        "__default__": data_split_pipeline + data_unit_tests_pipeline + data_preproc_pipeline + data_unit_tests_afterprep_pipeline #+ ingestion_pipeline,
    }