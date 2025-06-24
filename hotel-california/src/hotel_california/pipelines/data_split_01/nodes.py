from typing import Dict, Any, Tuple
import pandas as pd

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    df = data.copy()

    # Extract target + BookingID
    target = df[["BookingID", "Canceled"]]

    # Drop target from features
    df = df.drop(columns=["Canceled"])
    
    # Compute split indices
    train_end = int(0.8 * len(df))
    val_end = int(0.9 * len(df))

    # Features
    X_train_data = df.iloc[:train_end]
    X_val_data = df.iloc[train_end:val_end]
    X_test_data = df.iloc[val_end:]

    # Targets
    y_train_data = target["Canceled"].iloc[:train_end]
    y_val_data = target["Canceled"].iloc[train_end:val_end]
    y_test_data = target["Canceled"].iloc[val_end:]

    return X_train_data, X_val_data, X_test_data, y_train_data, y_val_data, y_test_data
