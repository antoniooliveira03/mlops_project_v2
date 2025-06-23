import great_expectations as gx
import mlflow
import pandas as pd
import logging


import great_expectations as gx
import mlflow
import pandas as pd
import logging

def unit_test(df: pd.DataFrame, mlruns_path: str) -> str:

    mlflow.set_tracking_uri(mlruns_path)
    exp_name = "data_unit_tests"
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(exp_name)
    else:
        experiment_id = experiment.experiment_id
    
    if mlflow.active_run() is not None:
        mlflow.end_run()

    df = df.copy(deep=True)

    with mlflow.start_run(experiment_id=experiment_id, run_name="data_unit_tests_run", nested=True) as run:
        mlflow.set_tag("mlflow.runName", "verify_data_quality")

        # Log basic stats
        mlflow.log_dict(df.describe(include='all').to_dict(), "describe_data_raw.json")

        pd_df_gx = gx.dataset.PandasDataset(df)
        gx_results_summary = {}

        def log_expectation(expectation_name: str, result: dict):
            gx_results_summary[expectation_name] = result["success"]
            path = f"expectation_results/{expectation_name}.json"
            mlflow.log_dict(result, path)

        # Each expectation block:
        result = pd_df_gx.expect_column_values_to_be_of_type('BookingID', 'int64')
        log_expectation("BookingID_type", result)
        assert result.success

        result = pd_df_gx.expect_column_values_to_be_unique('BookingID')
        log_expectation("BookingID_unique", result)
        assert result.success

        result = pd_df_gx.expect_column_values_to_be_between('ArrivalYear', 2016, 2016)
        log_expectation("ArrivalYear_fixed_2016", result)
        assert result.success

        # Add all remaining expectations just like above:
        columns_and_expectations = [
            ("ArrivalMonth", pd_df_gx.expect_column_values_to_be_between, {"min_value": 1, "max_value": 12}),
            ("ArrivalWeekNumber", pd_df_gx.expect_column_values_to_be_between, {"min_value": 1, "max_value": 53}),
            ("ArrivalDayOfMonth", pd_df_gx.expect_column_values_to_be_between, {"min_value": 1, "max_value": 31}),
            ("ArrivalHour", pd_df_gx.expect_column_values_to_be_between, {"min_value": 14, "max_value": 24}),
            ("WeekendStays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("WeekdayStays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("Adults", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("Children", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("Babies", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("FirstTimeGuest", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("AffiliatedCustomer", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("PreviousReservations", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("PreviousStays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("PreviousCancellations", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("DaysUntilConfirmation", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("OnlineReservation", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("BookingChanges", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("BookingToArrivalDays", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 365}),
            ("ParkingSpacesBooked", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("SpecialRequests", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 5}),
            ("PartOfGroup", pd_df_gx.expect_column_values_to_be_in_set, {"value_set": [0, 1]}),
            ("OrderedMealsPerDay", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 3}),
            ("FloorReserved", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 6}),
            ("FloorAssigned", pd_df_gx.expect_column_values_to_be_between, {"min_value": -1, "max_value": 6}),
            ("DailyRateEuros", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("DailyRateUSD", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("%PaidinAdvance", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 1}),
            ("CountryofOriginAvgIncomeEuros (Year-2)", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("CountryofOriginAvgIncomeEuros (Year-1)", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0}),
            ("CountryofOriginHDI (Year-1)", pd_df_gx.expect_column_values_to_be_between, {"min_value": 0, "max_value": 1}),
        ]

        for col, func, kwargs in columns_and_expectations:
            result = func(col, **kwargs)
            key = f"{col}_{func.__name__.replace('expect_column_values_to_', '')}"
            log_expectation(key, result)
            assert result.success

        # Log aggregated summary of expectation outcomes
        mlflow.log_dict(gx_results_summary, "expectation_results_summary.json")

        # Log cleaned data stats
        mlflow.log_dict(df.describe().to_dict(), "stats_data_cleaned.json")

    mlflow.end_run()
    logging.getLogger(__name__).info("All GX expectations passed and logged.")
    return df
    

def unit_test_y(y: pd.Series, mlruns_path: str) -> str:
    mlflow.set_tracking_uri(mlruns_path)
    exp_name = "label_data_tests"

    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(exp_name)
    else:
        experiment_id = experiment.experiment_id

    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Handle Series or DataFrame input
    if isinstance(y, pd.DataFrame):
        assert "Canceled" in y.columns, "'Canceled' column not found in y"
        label_series = y["Canceled"]
    elif isinstance(y, pd.Series):
        label_series = y
    else:
        raise ValueError("y must be a Series or a DataFrame")

    label_df = label_series.to_frame(name="Canceled")

    with mlflow.start_run(experiment_id=experiment_id, run_name="label_verification_run", nested=True):
        mlflow.set_tag("mlflow.runName", "verify_label_range")

        # Apply expectation
        gx_df = gx.dataset.PandasDataset(label_df)
        result = gx_df.expect_column_values_to_be_in_set("Canceled", [0, 1])

        # Log expectation result
        mlflow.log_dict(result, "expectation_results/Canceled_in_set.json")

        # Log summary
        summary = {"Canceled_in_set": result.success}
        mlflow.log_dict(summary, "label_expectation_summary.json")

        # Log label distribution statistics
        mlflow.log_dict(label_series.describe().to_dict(), "label_stats.json")

        # Assert to fail early if invalid
        assert result.success

    mlflow.end_run()
    logging.getLogger(__name__).info("Label expectations passed and logged.")
    return y