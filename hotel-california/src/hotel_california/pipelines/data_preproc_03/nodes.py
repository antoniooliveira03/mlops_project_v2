import pandas as pd

def drop(df: pd.DataFrame) -> pd.DataFrame:
    if 'CompanyReservation' in df.columns:
        df = df.drop(columns=['CompanyReservation'])
    return df
