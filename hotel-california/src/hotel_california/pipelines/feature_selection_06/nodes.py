import logging
import pandas as pd
import numpy as np
import json
import mlflow
import matplotlib.pyplot as plt
import hopsworks


from typing import Dict, Any, List, Tuple
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
import yaml


logger = logging.getLogger(__name__)

def load_features_node() -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open("conf/local/credentials.yml", "r") as f:
        credentials = yaml.safe_load(f)["feature_store"]

    project = hopsworks.login(
        api_key_value=credentials["FS_API_KEY"],
        project=credentials["FS_PROJECT_NAME"]
    )
    fs = project.get_feature_store()

    cat_fg = fs.get_feature_group('train_categorical_features', version=2)
    num_fg = fs.get_feature_group('train_numerical_features', version=2)

    categorical_features = cat_fg.read().columns.tolist()
    numerical_features = num_fg.read().columns.tolist()

    return categorical_features, numerical_features


def encode_features(X: pd.DataFrame, categorical_features: List[str]):
    """
    One-hot encodes only selected categorical features (e.g., 'arrivaltimeofday'),
    and keeps binary categorical features (0/1) as-is.
    """
    # Define which features to OHE
    ohe_features = ['arrivaltimeofday']
    
    # Keep the rest (0/1 features) as regular categoricals
    binary_features = [col for col in categorical_features if col not in ohe_features and col != "bookingid"]


    # OHE only the selected features
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_ohe = ohe.fit_transform(X[ohe_features])
    ohe_feature_names = ohe.get_feature_names_out(ohe_features)
    X_ohe_df = pd.DataFrame(X_ohe, columns=ohe_feature_names, index=X.index)
    X_ohe_df = X_ohe_df.apply(pd.to_numeric, errors='coerce')

    # Keep binary features as-is
    X_bin_df = X[binary_features]
    
    # Combine
    X_encoded = pd.concat([X_ohe_df, X_bin_df], axis=1)

    # Final list of features
    all_features = X_encoded.columns.tolist()
    return X_encoded, all_features


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    n_features: int = 10
) -> List[str]:
    """Apply feature selection based on method."""

    if method == "rfe":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(model, n_features_to_select=n_features)
        rfe.fit(X, y)
        return X.columns[rfe.support_].tolist()

    elif method == "feature_importance":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        return X.columns[indices].tolist()

    elif method == "mutual_info":
        mi = mutual_info_classif(X, y, discrete_features='auto')
        indices = np.argsort(mi)[::-1][:n_features]
        return X.columns[indices].tolist()

    else:
        raise ValueError(f"Unknown method: {method}")


def feature_selection_node(
    data: pd.DataFrame,
    mlruns_path: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str],
    numerical_features: List[str],
    parameters: Dict[str, Any]
) -> Tuple[List[str], Dict[str, int], plt.Figure]:

    mlflow.set_tracking_uri(mlruns_path)

    exp_name = "feature_selection_experiment"
    experiment = mlflow.get_experiment_by_name(exp_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    if mlflow.active_run() is not None:
        mlflow.end_run()

    logger.info("Starting feature selection node")

    if 'bookingid' in data.columns:
        data = data.set_index('bookingid')
    if 'bookingid' in X_train.columns:
        X_train = X_train.set_index('bookingid')

    methods = parameters.get("methods", ["rfe"])
    n_features = parameters.get("n_features", 10)
    encode_cats = parameters.get("encode_categoricals", True)

    with mlflow.start_run(experiment_id=experiment_id, run_name="feature_selection", nested=True):
        mlflow.log_params(parameters)

        if encode_cats:
            X_cat_encoded, _ = encode_features(X_train, categorical_features)
            X_cat_encoded = X_cat_encoded.apply(pd.to_numeric, errors='coerce')
            numerical_features = [col for col in numerical_features if col in X_train.columns]
            X_encoded = pd.concat([X_cat_encoded, X_train[numerical_features]], axis=1)
        else:
            numerical_features = [col for col in numerical_features if col in X_train.columns]
            X_encoded = X_train[numerical_features]

        X_encoded.drop(columns=[col for col in X_encoded.columns if col == 'arrivaltime'], errors='ignore', inplace=True)

        all_selected = []
        for method in methods:
            logger.info(f"Running feature selection with method: {method}")
            if isinstance(y_train, pd.DataFrame):
                if "canceled" in y_train.columns:
                    y_train = y_train["canceled"]
                else:
                    raise ValueError("y_train must contain a 'canceled' column.")
            selected = select_features(X_encoded, y_train, method, n_features=n_features)
            mlflow.log_param(f"{method}_selected", selected)
            all_selected.extend(selected)

        # Count feature selection votes
        counter = Counter(all_selected)
        best_features = [f for f, _ in counter.most_common(n_features)]

        logger.info(f"Final selected features: {best_features}")
        mlflow.log_param("final_selected_features", best_features)

        # Create final vote plot figure (returned to Kedro for saving)
        fig, ax = plt.subplots(figsize=(10, 6))
        feat_names, vote_counts = zip(*counter.most_common())
        ax.barh(feat_names[::-1], vote_counts[::-1])
        ax.set_xlabel("Vote Count")
        ax.set_title("Feature Selection Vote Count")
        fig.tight_layout()
        mlflow.log_figure(fig, "feature_votes_plot.png")

        return best_features, dict(counter), fig
