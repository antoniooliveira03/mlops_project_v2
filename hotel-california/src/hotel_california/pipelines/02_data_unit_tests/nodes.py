import pandas as pd
from .utils import ExpectationsReportV3
from great_expectations.data_context import DataContext
from great_expectations.checkpoint import Checkpoint


def generate_expectations_from_training(x_train: pd.DataFrame, y_train: pd.Series) -> str:
    df = x_train.copy()
    df["target"] = y_train

    data_context = DataContext()  # You can pass path if not in root
    report = ExpectationsReportV3()
    report.df = df

    report.config = type("Config", (), {"title": "training_data"})()  # Quick config mock
    report.to_expectation_suite(
        datasource_name="default_pandas_datasource",
        data_asset_name="training_asset",
        suite_name="training_suite",
        data_context=data_context,
        save_suite=True,
        run_validation=True,
        build_data_docs=True,
    )

    return "training_suite"  # optional output to link later


def validate_new_data_against_suite(x_val: pd.DataFrame, y_val: pd.Series, suite_name: str) -> dict:
    df = x_val.copy()
    df["target"] = y_val

    data_context = DataContext()
    datasource = data_context.get_datasource("default_pandas_datasource")
    data_asset = datasource.get_asset("validation_asset")

    batch_request = data_asset.build_batch_request()

    checkpoint_config = {
        "class_name": "Checkpoint",
        "validations": [
            {
                "batch_request": batch_request,
                "expectation_suite_name": suite_name,
            }
        ]
    }


    checkpoint = Checkpoint(
        f"_tmp_checkpoint_{suite_name}",
        data_context,
        name="_tmp_checkpoint",
        **checkpoint_config,
    )
    
    results = checkpoint.run()
    return results.to_json_dict()
