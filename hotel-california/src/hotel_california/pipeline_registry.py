from typing import Dict
from kedro.pipeline import Pipeline

from project_template.pipelines import (
  #  pipeline name as name,

)

def register_pipelines() -> Dict[str, Pipeline]:

    unit_tests_stage = unit_tests.create_pipeline()
    feature_store = feature_store.create_pipeline()
    unit_tests_2 = unit_tests_2.create_pipeline()
    # ...

    return {
        "unit_tests": unit_tests_stage,
        "preprocessing": preprocessing_stage,
        "split_data": split_data_stage,
        "feature_selection": feature_selection_stage,
        "tuning" : model_tuning_stage,
        "train": model_train_stage,
        "predict": model_predict_stage, 
        "drift_test" : drift_test_stage, 
        "__default__": preprocessing_stage + split_data_stage + feature_selection_stage + model_tuning_stage + model_train_stage,
    }