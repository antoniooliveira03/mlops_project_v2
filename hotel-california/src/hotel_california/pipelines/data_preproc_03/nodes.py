import pandas as pd

def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary columns
    if 'CompanyReservation' in X.columns:
        X = X.drop(columns=['CompanyReservation'])

    # Normalize column names
    X.columns = X.columns.str.lower()

    # Rename selected columns for consistency
    rename_map = {
        "%paidinadvance": "percent_paid_in_advance",
        "countryoforiginavgincomeeuros (year-2)": "country_income_euros_y2",
        "countryoforiginavgincomeeuros (year-1)": "country_income_euros_y1",
        "countryoforiginhdi (year-1)": "country_hdi_y1",
    }
    X = X.rename(columns=rename_map)

    # Ensure children are integers
    X["children"] = X["children"].astype(int)

    # Total stay days
    X["totalstaydays"] = X["weekendstays"] + X["weekdaystays"]

    # Total guests
    X["totalguests"] = X["adults"] + X["children"] + X["babies"]

    # Children and babies ratio
    total_guests_safe = X["totalguests"].replace(0, 1)
    X["childrenratio"] = X["children"] / total_guests_safe
    X["babiesratio"] = X["babies"] / total_guests_safe

    # Arrival time extraction
    X["hour"] = X["arrivalhour"].astype(int)
    X["minute"] = ((X["arrivalhour"] - X["hour"]) * 60).round().astype(int)
    X["arrivaltime"] = (
        X["arrivalyear"].astype(str) + "-" +
        X["arrivalmonth"].astype(str).str.zfill(2) + "-" +
        X["arrivaldayofmonth"].astype(str).str.zfill(2) + " " +
        X["hour"].astype(str).str.zfill(2) + ":" +
        X["minute"].astype(str).str.zfill(2)
    )
    X["arrivaltime"] = pd.to_datetime(X["arrivaltime"], format="%Y-%m-%d %H:%M")
    X["arrivaltime"] = pd.to_datetime(X["arrivaltime"])


    # Arrival time of day
    X["arrivaltimeofday"] = pd.cut(
        X["arrivalhour"],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        right=False,
        include_lowest=True
    )

    # Reservation gap feature
    X["confirmationtoarrivaldays"] = X["bookingtoarrivaldays"] - X["daysuntilconfirmation"]

    # Repeat guest indicator
    X["isrepeatguest"] = (X["previousreservations"] > 0).astype(int)

    # Socioeconomic feature: income change from Y-2 to Y-1
    X["incomechange"] = X["country_income_euros_y1"] - X["country_income_euros_y2"]

    return X

def prepare_target(y: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    y.columns = y.columns.str.lower()
    X["arrivaltime"] = pd.to_datetime(X["arrivaltime"])
    return pd.merge(y, X[["bookingid", "arrivaltime"]], on="bookingid", how="inner")
