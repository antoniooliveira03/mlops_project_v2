from typing import Dict
from kedro.pipeline import Pipeline


from hotel_california.pipelines import (
    data_split_01 as data_split,
    data_unit_tests_02 as data_unit_tests,
    data_preproc_03 as data_preproc,
    data_unit_tests_afterprep_04 as data_unit_tests_afterprep,
    ingestion_05 as ingestion,
    feature_selection_06 as feature_selection,
    model_selection_07 as model_selection,
    model_train_08 as model_train,
    model_predict_09 as model_predict
)

def register_pipelines() -> Dict[str, Pipeline]:

    data_split_pipeline = data_split.create_pipeline()
    data_unit_tests_pipeline = data_unit_tests.create_pipeline()  
    data_preproc_pipeline = data_preproc.create_pipeline()
    data_unit_tests_afterprep_pipeline = data_unit_tests_afterprep.create_pipeline()
    #ingestion_pipeline = ingestion.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    model_train_pipeline = model_train.create_pipeline()
    model_selection_pipeline = model_selection.create_pipeline()
    model_prediction_pipeline = model_predict.create_pipeline()

    return {
        "split": data_split_pipeline,
        "unit_tests": data_unit_tests_pipeline,
        "preproc": data_preproc_pipeline,
        "unit_tests_afterprep": data_unit_tests_afterprep_pipeline,
        "feature_selection": feature_selection_pipeline,
        "model_selection": model_selection_pipeline,
        "model_train": model_train_pipeline,
        "model_prediction": model_prediction_pipeline,
        #"ingestion": ingestion_pipeline,
        # The ingestion pipeline is commented out by default to avoid long execution times.
        # Uncomment it only if you need to run it and upload data to your Hopsworks workspace.
        "__default__":  data_split_pipeline + 
                        data_unit_tests_pipeline + 
                        data_preproc_pipeline + 
                        data_unit_tests_afterprep_pipeline + 
                        feature_selection_pipeline
                        #+ ingestion_pipeline,
    }