import pandas as pd

def drop(df: pd.DataFrame) -> pd.DataFrame:
    if 'CompanyReservation' in df.columns:
        df = df.drop(columns=['CompanyReservation'])
    return df

def normalize_column_names(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lowercase all column names in both X and y."""
    X.columns = X.columns.str.lower()
    y.columns = y.columns.str.lower()
    return X, y


def rename_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Rename selected columns for clarity and consistency."""
    rename_map = {
        "%paidinadvance": "percent_paid_in_advance",
        "countryoforiginavgincomeeuros (year-2)": "country_income_euros_y2",
        "countryoforiginavgincomeeuros (year-1)": "country_income_euros_y1",
        "countryoforiginhdi (year-1)": "country_hdi_y1",
    }
    return X.rename(columns=rename_map)


def create_arrivaltime_feature(X: pd.DataFrame) -> pd.DataFrame:
    """Create a datetime column from arrival date and time components."""
    X["hour"] = X["arrivalhour"].astype(int)
    X["minute"] = ((X["arrivalhour"] - X["hour"]) * 60).round().astype(int)
    
    X["arrivaltime"] = (
        X["arrivalyear"].astype(str) + "-" +
        X["arrivalmonth"].astype(str) + "-" +
        X["arrivaldayofmonth"].astype(str) + " " +
        X["hour"].astype(str) + ":" +
        X["minute"].astype(str).str.zfill(2)
    )
    X["arrivaltime"] = pd.to_datetime(X["arrivaltime"], format="%Y-%m-%d %H:%M")

    print("First customer of the year:", X["arrivaltime"].min())
    print("Last customer of the year:", X["arrivaltime"].max())

    return X


def merge_target_with_datetime(y: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Merge target dataframe with arrival time."""
    return pd.merge(y, X[["bookingid", "arrivaltime"]], on="bookingid", how="inner")
