from typing import Dict, Any, Tuple
import pandas as pd

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    df = data.copy()
    # Extract target as Series
    target = df['Canceled']
    
    # Drop target from features
    df = df.drop(columns=["Canceled"])
    
    # Chronological split
    split_idx = int(0.8 * len(df))
    X_train_data = df.iloc[:split_idx]
    X_val_data = df.iloc[split_idx:]
    
    y_train_data = target.iloc[:split_idx]
    y_val_data = target.iloc[split_idx:]
    
    return X_train_data, X_val_data, y_train_data, y_val_data
