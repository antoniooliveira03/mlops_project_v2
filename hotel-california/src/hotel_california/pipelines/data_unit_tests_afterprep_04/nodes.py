import mlflow
import great_expectations as gx
import pandas as pd
import logging

def unit_test_final(df: pd.DataFrame, mlruns_path: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(mlruns_path)
    exp_name = "data_unit_tests_preprocessed_data"
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(exp_name)
    else:
        experiment_id = experiment.experiment_id

    if mlflow.active_run() is not None:
        mlflow.end_run()

    df = df.copy(deep=True)

    with mlflow.start_run(experiment_id=experiment_id, run_name="data_unit_tests_run", nested=True) as run:
        mlflow.set_tag("mlflow.runName", "verify_data_quality_preprocessed_data")

        pd_df_gx = gx.dataset.PandasDataset(df)
        gx_results_summary = {}

        def log_expectation(expectation_name: str, result: dict):
            gx_results_summary[expectation_name] = result["success"]
            path = f"expectation_results_prepdata/{expectation_name}.json"
            mlflow.log_dict(result.to_json_dict(), path)

        # Basic expectations
        result = pd_df_gx.expect_column_values_to_be_of_type('bookingid', 'int64')
        log_expectation("BookingID_type", result)
        assert result.success

        result = pd_df_gx.expect_column_values_to_be_unique('bookingid')
        log_expectation("BookingID_unique", result)
        assert result.success

        result = pd_df_gx.expect_column_values_to_be_between('arrivalyear', 2016, 2016)
        log_expectation("ArrivalYear_fixed_2016", result)
        assert result.success

        # All column expectations
        columns_and_expectations = [
            ("arrivalmonth", pd_df_gx.expect_column_values_to_be_between, {"min_value": 1, "max_value": 12}),
            ("arrivalweeknumber", pd_df_gx.expect_column_values_to_be_between, {"min_value": 1, "max_value": 53}),
            ("arrivaldayofmonth", pd_df_gx.expect_column_values_to_be_between, {"min_value": 1, "max_value": 31}),
            ("arrivalhour", pd_df_gx.expect_column_values_to_be_between, {"min_value": 14, "max_value": 24}),
            ("weekendstays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("weekdaystays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("adults", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("children", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("babies", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("firsttimeguest", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("affiliatedcustomer", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("previousreservations", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("previousstays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("previouscancellations", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("daysuntilconfirmation", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("onlinereservation", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("bookingchanges", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("bookingtoarrivaldays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 365}),
            ("parkingspacesbooked", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("specialrequests", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 5}),
            ("partofgroup", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("orderedmealsperday", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 3}),
            ("floorreserved", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 6}),
            ("floorassigned", pd_df_gx.expect_column_values_to_be_between, {"min_value": -1, "max_value": 6}),
            ("dailyrateeuros", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("dailyrateusd", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("percent_paid_in_advance", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 1}),
            ("country_income_euros_y2", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("country_income_euros_y1", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("country_hdi_y1", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 1}),

            # Engineered features
            ("hour", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 23}),
            ("minute", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 59}),
            ("totalstaydays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("totalguests", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("childrenratio", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 1}),
            ("babiesratio", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 1}),
            ("confirmationtoarrivaldays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 365}),
            ("isrepeatguest", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("incomechange", pd_df_gx.expect_column_values_to_be_between, {"min_value": -50000, "max_value": 50000}),
            ("arrivaltimeofday", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": ['night', 'morning', 'afternoon', 'evening']}),
        ]

        for col, func, kwargs in columns_and_expectations:
            result = func(col, **kwargs)
            key = f"{col}_{func.__name__.replace('expect_column_values_to_', '')}"
            log_expectation(key, result)
            assert result.success

        # Log aggregated summary
        mlflow.log_dict(gx_results_summary, "expectation_results_summary.json")

    mlflow.end_run()
    logging.getLogger(__name__).info("All GX expectations passed and logged.")
    return df