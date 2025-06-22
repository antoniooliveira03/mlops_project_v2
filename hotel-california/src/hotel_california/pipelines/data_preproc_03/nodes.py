import pandas as pd

def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    # Drop
    if 'CompanyReservation' in X.columns:
        X = X.drop(columns=['CompanyReservation'])

    # Normalize
    X.columns = X.columns.str.lower()

    # Rename
    rename_map = {
        "%paidinadvance": "percent_paid_in_advance",
        "countryoforiginavgincomeeuros (year-2)": "country_income_euros_y2",
        "countryoforiginavgincomeeuros (year-1)": "country_income_euros_y1",
        "countryoforiginhdi (year-1)": "country_hdi_y1",
    }
    X = X.rename(columns=rename_map)

    # Arrival time
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

    return X

def prepare_target(y: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    y.columns = y.columns.str.lower()
    return pd.merge(y, X[["bookingid", "arrivaltime"]], on="bookingid", how="inner")
