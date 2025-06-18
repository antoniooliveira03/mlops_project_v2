import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import great_expectations as ge
import mlflow

def unit_test(data: pd.DataFrame,): 

    if mlflow.active_run():
        mlflow.end_run()

    df=data.copy(deep=True)

    mlflow.set_experiment("data_unit_tests")

    with mlflow.start_run(run_name="verify_data_quality") as run:
        mlflow.set_tag("mlflow.runName", "verify_data_quality")
    
        # Log the raw data statistics
        describe_to_dict=df.describe().to_dict()
        mlflow.log_dict(describe_to_dict,"describe_data_raw.json")
    
        # Perform data quality checks using Great Expectations
        pd_df_ge = ge.from_pandas(df)
        assert pd_df_ge.expect_column_values_to_be_between('product_weight_g', min_value=0, max_value=50000, mostly=0.75).success == True
        assert pd_df_ge.expect_column_values_to_not_be_null('product_weight_g').success == False
        assert pd_df_ge.expect_column_values_to_be_between('product_length_cm', min_value=5, max_value=150, mostly=0.75).success == True
        assert pd_df_ge.expect_column_values_to_not_be_null('product_length_cm').success == False
        assert pd_df_ge.expect_column_values_to_be_between('product_width_cm', min_value=5, max_value=110, mostly=0.75).success == True
        assert pd_df_ge.expect_column_values_to_not_be_null('product_width_cm').success == False
        assert pd_df_ge.expect_column_values_to_be_between('product_height_cm', min_value=5, max_value=110, mostly=0.75).success == True
        assert pd_df_ge.expect_column_values_to_not_be_null('product_height_cm').success == False
        assert pd_df_ge.expect_column_to_exist('order_status').success == True
        assert pd_df_ge.expect_column_values_to_not_be_null('order_status').success == True
        assert pd_df_ge.expect_column_to_exist('price').success == True
        assert pd_df_ge.expect_column_values_to_not_be_null('customer_id').success == True
        assert pd_df_ge.expect_column_values_to_not_be_null('product_category_name').success == False
        assert pd_df_ge.expect_column_values_to_be_in_set("order_status", ['delivered', 'shipped', 'processing', 'canceled', 'invoiced', 'unavailable', 'approved']).success == False
        assert pd_df_ge.expect_column_values_to_be_in_type_list("customer_id", ["int", "int64"]).success == True
        
         # Log the cleaned data statistics
        describe_to_dict=df.describe().to_dict()
        mlflow.log_dict(describe_to_dict,"describe_data_cleaned.json")
    mlflow.end_run()


    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")

    return 0