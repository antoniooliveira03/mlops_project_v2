import pandas as pd
from .utils import ExpectationsReportV3
import great_expectations as gx
from great_expectations.checkpoint import Checkpoint
from kedro.io import DataCatalog
from pathlib import Path
from ydata_profiling import ProfileReport
from ydata_profiling.expectations_report import ExpectationsReport



def generate_expectations_from_training(x_train: pd.DataFrame, y_train: pd.Series) -> str:

    data_context = gx.get_context(context_root_dir = "gx")

    profile_X = ProfileReport(x_train, 
                        title=f"X Profiling Report", 
                        minimal=True)

    profile_y = ProfileReport(y_train, 
                        title=f"Y Profiling Report", 
                        minimal=True)

    ExpectationsReport.to_expectation_suite = ExpectationsReportV3.to_expectation_suite

    # Combine into one DataFrame
    df = x_train.copy()
    df["target"] = y_train

    report = ExpectationsReportV3()
    report.df = df
    report.config = type("Config", (), {"title": "training_data"})()

    # Generate expectation suite
    report.to_expectation_suite(
        datasource_name="expectations_datasource",
        data_asset_name="training_asset",
        suite_name="training_suite",
        data_context=data_context,
        save_suite=True,
        run_validation=True,
        build_data_docs=True,
    )



import pandas as pd

def get_validation_results(checkpoint_result):
    # Extract  validation result
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))

    # validation_result_data contains validation details under 'validation_result'
    validation_result = validation_result_data.get('validation_result', {})

    results = validation_result.get("results", [])
    meta = validation_result.get("meta", {})
    expectation_suite_name = meta.get('expectation_suite_name', '')

    # Prepare empty DataFrame with desired columns
    df_validation = pd.DataFrame(columns=[
        "Success", "Expectation Type", "Column", "Column Pair", "Max Value",
        "Min Value", "Element Count", "Unexpected Count", "Unexpected Percent",
        "Value Set", "Unexpected Value", "Observed Value"
    ])

    for result in results:
        expectation_config = result.get('expectation_config', {})
        kwargs = expectation_config.get('kwargs', {})

        observed_value = result.get('result', {}).get('observed_value', None)
        value_set = kwargs.get('value_set', None)

        # Identify unexpected values if possible
        if isinstance(observed_value, list) and isinstance(value_set, (list, set)):
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value = []

        df_validation = pd.concat([
            df_validation,
            pd.DataFrame.from_dict([{
                "Success": result.get('success', None),
                "Expectation Type": expectation_config.get('expectation_type', None),
                "Column": kwargs.get('column', None),
                "Column Pair": (kwargs.get('column_A', None), kwargs.get('column_B', None)),
                "Max Value": kwargs.get('max_value', None),
                "Min Value": kwargs.get('min_value', None),
                "Element Count": result.get('result', {}).get('element_count', None),
                "Unexpected Count": result.get('result', {}).get('unexpected_count', None),
                "Unexpected Percent": result.get('result', {}).get('unexpected_percent', None),
                "Value Set": value_set,
                "Unexpected Value": unexpected_value,
                "Observed Value": observed_value
            }])
        ], ignore_index=True)

    return df_validation

# Validate and save the validation results
def validate_and_save(x_val, y_val, suite_name, catalog):
    import great_expectations as gx
    from great_expectations.checkpoint import Checkpoint

    data_context = gx.get_context(context_root_dir="gx")

    # Create batch request for validation asset (adjust asset name as needed)
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
        name=f"_tmp_checkpoint_{suite_name}",
        **checkpoint_config,
    )

    checkpoint_result = checkpoint.run()

    # Use your parser to get a dataframe with validation results
    df_validation = get_validation_results(checkpoint_result)

    # Check if all validations passed
    if df_validation["Success"].all():
        dataset_name = f"{suite_name}_validated"
        if dataset_name in catalog.list():
            catalog.save(dataset_name, df_validation)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in catalog.")
        return df_validation
    else:
        print("Validation failed; not saving dataset.")
        return None
