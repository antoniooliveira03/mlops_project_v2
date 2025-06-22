import logging
import numpy as np
import pandas as pd
import hopsworks

from typing import Any, Dict, Tuple
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)

def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
    """
    Build an expectation suite for hotel booking features.
    
    Args:
        expectation_suite_name (str): Name for the expectation suite.
        feature_group (str): One of ['arrival_features', 'guest_features', 'booking_features', 'financial_features', 'target'].
    
    Returns:
        ExpectationSuite: Configured expectation suite for the feature group.
    """
    expectation_suite = ExpectationSuite(expectation_suite_name=expectation_suite_name)

    # Arrivaltime
    expectation_suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "arrivaltime"},
        )
    )

    if feature_group == 'arrival_features':
        for col in ['arrivalyear', 'arrivalmonth', 'arrivalweeknumber', 'arrivaldayofmonth', 'arrivalhour']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col},
                )
            )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "arrivalyear", "value_set": [2016]},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "arrivalmonth", "min_value": 1, "max_value": 12},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "arrivaldayofmonth", "min_value": 1, "max_value": 31},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "arrivalhour", "min_value": 0, "max_value": 24},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "arrivalweeknumber", "min_value": 1, "max_value": 53},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "totalstaydays",
                    "min_value": 0,
                    "strict_min": False,
                },
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "arrivaltimeofday",
                    "value_set": ["night", "morning", "afternoon", "evening"],
                },
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_columns_to_match_ordered_list",
                kwargs={"column_list": [
                    'bookingid', 'arrivaltime', 'arrivalyear', 'arrivalmonth', 'arrivalweeknumber',
                    'arrivaldayofmonth', 'arrivalhour', 'weekendstays', 'weekdaystays'
                    'totalstaydays', 'arrivaltimeofday'
                ]}
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 11}
            )
        )

    if feature_group == 'guest_features':
        for col in ['adults', 'children', 'babies']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "int64"},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": col, "min_value": 0},
                )
            )

        for col in ['firsttimeguest', 'affiliatedcustomer', 'partofgroup', 'isrepeatguest']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_distinct_values_to_be_in_set",
                    kwargs={"column": col, "value_set": [0, 1]},
                )
            )
        
        # childrenratio and babiesratio: must be between 0 and 1
        for col in ["childrenratio", "babiesratio"]:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": col,
                        "min_value": 0.0,
                        "max_value": 1.0,
                        "strict_min": False,
                        "strict_max": False,
                    },
                )
            )

        for col in ['previousreservations', 'previousstays', 'previouscancellations', 'totalguests']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": col, "min_value": 0},
                )
            )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_columns_to_match_ordered_list",
                kwargs={"column_list": [
                    'bookingid', 'arrivaltime', 'adults', 'children', 'babies', 'firsttimeguest', 'affiliatedcustomer',
                    'previousreservations', 'previousstays', 'previouscancellations',
                    'partofgroup', 'totalguests', 'childrenratio', 'babiesratio', 'isrepeatguest'
                ]}
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 15}
            )
        )

    if feature_group == 'booking_features':
        for col in ['daysuntilconfirmation', 'bookingtoarrivaldays', 'bookingchanges']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": col, "min_value": 0},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "int64"},
                )
            )

        for col in ['orderedmealsperday', 'floorreserved', 'floorassigned']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "int64"},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": col, "min_value": 0},
                )
            )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "onlinereservation", "value_set": [0, 1]},
            )
        )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "parkingspacesbooked", "min_value": 0},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "specialrequests", "min_value": 0},
            )
        )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "confirmationtoarrivaldays",
                    "min_value": 0,
                    "strict_min": False,
                },
            )
        )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_columns_to_match_ordered_list",
                kwargs={"column_list": [
                    'bookingid', 'arrivaltime','daysuntilconfirmation', 'onlinereservation', 'bookingchanges',
                    'bookingtoarrivaldays', 'parkingspacesbooked', 'specialrequests',
                    'orderedmealsperday', 'floorreserved', 'floorassigned', 'confirmationtoarrivaldays'
                ]}
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 12}
            )
        )

    if feature_group == 'financial_features':
        for col in ['dailyrateeuros', 'dailyrateusd']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "float64"},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": col, "min_value": 0},
                )
            )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "percent_paid_in_advance", "min_value": 0, "max_value": 100},
            )
        )

        for col in [
            'country_income_euros_y2',
            'country_income_euros_y1',
            'country_hdi_y1'
        ]:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": col, "min_value": 0},
                )
            )
        
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "incomechange",
                },
            )
        )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={
                    "column": "incomechange",
                    "type_": "float",
                },
            )
        )

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_columns_to_match_ordered_list",
                kwargs={"column_list": [
                    'bookingid', 'arrivaltime',
                    'dailyrateeuros', 'dailyrateusd', 'percent_paid_in_advance',
                    'country_income_euros_y2',
                    'country_income_euros_y1',
                    'country_hdi_y1', 'incomechange'
                ]}
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 9}
            )
        )

    # numerical features
    if feature_group == 'numerical_features':

        for i in ['children', 'adults','babies']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "int64"},
                )
            )


    if feature_group == 'categorical_features':

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "arrivaltimeofday", "value_set": ["night", "morning", "afternoon", "evening"]},
            )
        )

    if feature_group == 'target':
        target_column = 'canceled'
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": target_column, "value_set": [0, 1]},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": target_column},
            )
        )

        # Columns
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_columns_to_match_ordered_list",
                kwargs={"column_list": ['bookingid', target_column, 'arrivaltime']},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 3}
            )
        )

    return expectation_suite

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in Hopsworks.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        object_feature_group (FeatureGroup): The created feature group object.
    """

    # Connect to feature store
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Ensure the feature group is offline and does not rely on Kafka.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=["bookingid"],
        event_time="arrivaltime",  
        online_enabled=False,  
        expectation_suite=validation_expectation_suite,  # Expectation suite for data validation
    )

    try:
        # Upload the data to the feature group
        object_feature_group.insert(
            features=data,
            overwrite=False,
             storage="offline",
            write_options={
                "wait_for_job": True,
            },
        )

        # Add feature descriptions
        for feature_name, feature_desc in group_description.items():
            object_feature_group.update_feature_description(feature_name, feature_desc)

        # Configure and update statistics
        object_feature_group.statistics_config = {
            "enabled": True,
            "histograms": True,
            "correlations": True,
        }
        object_feature_group.update_statistics_config()
        object_feature_group.compute_statistics()

        # Return the feature group object
        return object_feature_group
    
    except Exception as e:
        print(f"Error during feature group creation or data insertion: {e}")
        raise e


def ingestion(df: pd.DataFrame, parameters: dict, target_df: pd.DataFrame = None, prefix: str = ""):
    """
    Ingest hotel booking data into the feature store with validation.

    Args:
        df (pd.DataFrame): DataFrame with hotel booking data.
        parameters (dict): Configuration parameters, including:
            - to_feature_store (bool)
            - feature_group_versions (dict)
            - feature_descriptions (dict)
            - settings (dict)

    Returns:
        pd.DataFrame: The processed DataFrame.
    """ 

    # Build expectation suites for each feature group
    validation_suites = {
        "arrival_features": build_expectation_suite("arrival_expectations", "arrival_features"),
        "guest_features": build_expectation_suite("guest_expectations", "guest_features"),
        "booking_features": build_expectation_suite("booking_expectations", "booking_features"),
        "financial_features": build_expectation_suite("financial_expectations", "financial_features"),
        "numerical_features": build_expectation_suite("numerical_expectations", "numerical_features"),
        "categorical_features": build_expectation_suite("categorical_expectations", "categorical_features"),
        "target": build_expectation_suite("target_expectations", "target"),
    }
    
    # Split the DataFrame into feature groups
    feature_groups = {
        "arrival_features": df[['bookingid', 'arrivaltime', 'arrivalyear', 'arrivalmonth', 'arrivalweeknumber',
                                'arrivaldayofmonth', 'arrivalhour', 'weekendstays', 'weekdaystays', 'totalstaydays',
                                'arrivaltimeofday']],
        "guest_features": df[['bookingid', 'arrivaltime', 'adults', 'children', 'babies', 'firsttimeguest',
                              'affiliatedcustomer', 'previousreservations', 'previousstays', 'previouscancellations',
                              'partofgroup', 'totalguests', 'childrenratio', 'babiesratio',
                              'isrepeatguest']],
        "booking_features": df[['bookingid', 'arrivaltime', 'daysuntilconfirmation', 'onlinereservation',
                               'bookingchanges', 'bookingtoarrivaldays', 'parkingspacesbooked', 'specialrequests',
                               'orderedmealsperday', 'floorreserved', 'floorassigned', 'confirmationtoarrivaldays']],
        "financial_features": df[['bookingid', 'arrivaltime', 'dailyrateeuros', 'dailyrateusd', 'percent_paid_in_advance',
                                  'country_income_euros_y2', 'country_income_euros_y1', 'country_hdi_y1', 'incomechange']],
        "numerical_features": df[['bookingid', 'arrivaltime', 'arrivalyear', 'arrivalmonth', 'arrivalweeknumber',
                                'arrivaldayofmonth', 'arrivalhour', 'weekendstays', 'weekdaystays', 'totalstaydays',
                                'adults', 'children', 'babies', 'previousreservations', 'previousstays',
                                'previouscancellations', 'totalguests', 'childrenratio', 'babiesratio', 'daysuntilconfirmation',
                                'bookingchanges', 'bookingtoarrivaldays', 'parkingspacesbooked','specialrequests',
                                'orderedmealsperday', 'floorreserved', 'floorassigned', 'confirmationtoarrivaldays',
                                'dailyrateeuros', 'dailyrateusd', 'percent_paid_in_advance',
                                'country_income_euros_y2', 'country_income_euros_y1', 'country_hdi_y1', 'incomechange']],
        "categorical_features": df[['bookingid', 'arrivaltime', 'arrivaltimeofday', 'firsttimeguest', 'affiliatedcustomer', 'partofgroup',
                                    'isrepeatguest', 'onlinereservation']],
        "target": pd.concat([
            df[["bookingid", "arrivaltime"]].reset_index(drop=True),
            target_df.reset_index(drop=True)[["canceled"]]
        ], axis=1)
        if target_df is not None else None,
    }

    for group_name, data in feature_groups.items():
        if data is None and 'arrivaltime' in data.columns:
            data['arrivaltime'] = pd.to_datetime(data['arrivaltime'], errors='coerce')
            full_group_name = f"{prefix}_{group_name}" if prefix else group_name
            to_feature_store(
                data=data,
                group_name=full_group_name)


    if parameters.get("to_feature_store", False):
        for group_name, data in feature_groups.items():
            to_feature_store(
                data=data,
                group_name=group_name,
                feature_group_version=parameters.get("feature_group_versions", {}).get(group_name, 1),
                description=parameters.get("feature_descriptions", {}).get(group_name, ""),
                group_description=parameters.get("feature_descriptions", {}).get(group_name, {}),
                validation_expectation_suite=validation_suites[group_name],
                credentials_input=credentials["feature_store"]
            )
    
    return df
