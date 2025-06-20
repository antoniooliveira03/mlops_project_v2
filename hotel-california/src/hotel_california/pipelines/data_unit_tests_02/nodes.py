import great_expectations as gx
import mlflow
import pandas as pd
import logging


def unit_test(df: pd.DataFrame, mlruns_path: str) -> str:

    mlflow.set_tracking_uri(mlruns_path)

    if mlflow.active_run():
        mlflow.end_run()

    df = df.copy(deep=True)
    mlflow.set_experiment("data_unit_tests")

    with mlflow.start_run(run_name="data_unit_tests_run_") as run:
        mlflow.set_tag("mlflow.runName", "verify_data_quality")

        # Log raw stats
        mlflow.log_dict(df.describe(include='all').to_dict(), "describe_data_raw.json")

        pd_df_gx = gx.dataset.PandasDataset(df)

        # BookingID: integer, unique
        assert pd_df_gx.expect_column_values_to_be_of_type('BookingID', 'int64').success
        assert pd_df_gx.expect_column_values_to_be_unique('BookingID').success

        # ArrivalYear: int, always 2016 (min=max=2016)
        assert pd_df_gx.expect_column_values_to_be_of_type('ArrivalYear', 'int64').success
        assert pd_df_gx.expect_column_values_to_be_between('ArrivalYear', 2016, 2016).success

        # ArrivalMonth: int 1-12
        assert pd_df_gx.expect_column_values_to_be_between('ArrivalMonth', 1, 12).success

        # ArrivalWeekNumber: int 1-53
        assert pd_df_gx.expect_column_values_to_be_between('ArrivalWeekNumber', 1, 53).success

        # ArrivalDayOfMonth: int 1-31
        assert pd_df_gx.expect_column_values_to_be_between('ArrivalDayOfMonth', 1, 31).success

        # ArrivalHour: float or int between 14 and 24
        assert pd_df_gx.expect_column_values_to_be_between('ArrivalHour', 14, 24).success

        # WeekendStays: int >=0 
        assert pd_df_gx.expect_column_values_to_be_between('WeekendStays', 
                                                           min_value=0, 
                                                           max_value=None).success

        # WeekdayStays: int >=0 
        assert pd_df_gx.expect_column_values_to_be_between('WeekdayStays',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # Adults: int >= 0
        assert pd_df_gx.expect_column_values_to_be_between('Adults',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # Children: int >= 0
        assert pd_df_gx.expect_column_values_to_be_between('Children',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # Babies: int >= 0
        assert pd_df_gx.expect_column_values_to_be_between('Babies',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # FirstTimeGuest: binary 0 or 1
        assert pd_df_gx.expect_column_values_to_be_in_set('FirstTimeGuest', [0, 1]).success

        # AffiliatedCustomer: binary 0 or 1
        assert pd_df_gx.expect_column_values_to_be_in_set('AffiliatedCustomer', [0, 1]).success

        # PreviousReservations: int >=0
        assert pd_df_gx.expect_column_values_to_be_between('PreviousReservations',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # PreviousStays: int >=0 
        assert pd_df_gx.expect_column_values_to_be_between('PreviousStays',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # PreviousCancellations: int >=0 
        assert pd_df_gx.expect_column_values_to_be_between('PreviousCancellations',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # DaysUntilConfirmation: int >=0
        assert pd_df_gx.expect_column_values_to_be_between('DaysUntilConfirmation',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # OnlineReservation: binary 0 or 1
        assert pd_df_gx.expect_column_values_to_be_in_set('OnlineReservation', [0, 1]).success

        # BookingChanges: int >=0
        assert pd_df_gx.expect_column_values_to_be_between('BookingChanges',
                                                           min_value=0, 
                                                           max_value=None).success
        # BookingToArrivalDays: int >=0 (max 365)
        assert pd_df_gx.expect_column_values_to_be_between('BookingToArrivalDays', 0, 365).success

        # ParkingSpacesBooked: binary 0 or 1
        assert pd_df_gx.expect_column_values_to_be_in_set('ParkingSpacesBooked', [0, 1]).success

        # SpecialRequests: int >=0 (max 5)
        assert pd_df_gx.expect_column_values_to_be_between('SpecialRequests', 0, 5).success

        # PartOfGroup: binary 0 or 1
        assert pd_df_gx.expect_column_values_to_be_in_set('PartOfGroup', [0, 1]).success

        # OrderedMealsPerDay: int >=0
        assert pd_df_gx.expect_column_values_to_be_between('OrderedMealsPerDay',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # FloorReserved: int 0-6
        assert pd_df_gx.expect_column_values_to_be_between('FloorReserved', 0, 6).success

        # FloorAssigned: int -1 to 6
        assert pd_df_gx.expect_column_values_to_be_between('FloorAssigned', -1, 6).success

        # DailyRateEuros: float 0 >=
        assert pd_df_gx.expect_column_values_to_be_between('DailyRateEuros',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # DailyRateUSD: float 0 >=
        assert pd_df_gx.expect_column_values_to_be_between('DailyRateUSD',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # %PaidinAdvance: float 0-1
        assert pd_df_gx.expect_column_values_to_be_between('%PaidinAdvance', 0, 1).success

        # CountryofOriginAvgIncomeEuros (Year-2): float >=0
        assert pd_df_gx.expect_column_values_to_be_between('CountryofOriginAvgIncomeEuros (Year-2)',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # CountryofOriginAvgIncomeEuros (Year-1): float 0 >=
        assert pd_df_gx.expect_column_values_to_be_between('CountryofOriginAvgIncomeEuros (Year-1)',
                                                           min_value=0, 
                                                           max_value=None).success
        
        # CountryofOriginHDI (Year-1): float 0 - 1
        assert pd_df_gx.expect_column_values_to_be_between('CountryofOriginHDI (Year-1)', 0, 1).success



        
         # Log the cleaned data statistics
        describe_to_dict=df.describe().to_dict()
        mlflow.log_dict(describe_to_dict,"stats_data_cleaned.json")
        
    mlflow.end_run()
    log = logging.getLogger(__name__)
    log.info("Success")

    return "All data quality tests passed successfully."
        
    

def unit_test_y(y: pd.Series, mlruns_path: str) -> str:
    
    mlflow.set_tracking_uri(mlruns_path)

    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_experiment("label_data_tests")

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    with mlflow.start_run(run_name="label_verification_run") as run:
        mlflow.set_tag("mlflow.runName", "verify_label_range")

        # Convert to DataFrame for GX
        df = pd.DataFrame({"label": y})
        gx_df = gx.dataset.PandasDataset(df)

        # Run expectation: only 0 or 1
        assert gx_df.expect_column_values_to_be_in_set('label', [0, 1]).success

        # Log basic stats
        mlflow.log_dict(df["label"].describe().to_dict(), "label_stats.json")

    mlflow.end_run()
    return "Label check passed: values are binary (0 or 1)."
