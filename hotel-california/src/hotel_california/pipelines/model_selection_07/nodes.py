import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
import optuna
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

models_dict = {
    "RandomForestClassifier": RandomForestClassifier(),
    #"GradientBoostingClassifier": GradientBoostingClassifier(),
    #"LogisticRegression": LogisticRegression(),
    #"XGBClassifier": XGBClassifier(),
    #"LGBMClassifier": LGBMClassifier()
}

def _get_or_create_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id

def register_model(
    model_path: str,
    model_name: str,
    model_tag: str = None,
    model_alias: str = None
) -> None:
    client = MlflowClient()
    version = mlflow.register_model(model_path, model_name)
    
    if any(var is not None for var in [model_tag, model_alias]):
        if model_tag:
            client.set_registered_model_tag(model_name, "task", "classification")
        if model_alias:
            client.set_registered_model_alias(model_name, model_alias, version.version)


def get_search_space(trial, model_name, param_grid):
    params = {}
    if model_name not in param_grid:
        return params
    
    for param, values in param_grid[model_name].items():
        if len(values) == 1:
            params[param] = values[0]
        else:
            params[param] = trial.suggest_categorical(param, values)
    return params


def objective(trial, X_train, X_val, y_train, y_val, model_name, base_model, param_grid, best_columns=None):
    params = get_search_space(trial, model_name, param_grid)
    model = base_model.__class__(**params)

    if best_columns is not None:
        # If best_columns is provided, filter the training and validation sets
        X_train_ = X_train[best_columns]
        X_val_ = X_val[best_columns]
    else:
        # If no best_columns, use the full dataset
        X_train_ = X_train
        X_val_ = X_val

    model.fit(X_train_, y_train)
    y_val_pred = model.predict(X_val_)
    f1 = f1_score(y_val, y_val_pred)
    
    # Log metrics and parameters to MLflow for this trial
    mlflow.log_params(params)
    mlflow.log_metric("val_f1_score", f1)

    return f1

def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            logger.info(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            logger.info(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

def update_parameters_yaml(yaml_path: str, new_model_name: str, new_params: dict, use_feature_selection: bool = True) -> None:

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"{yaml_path} not found")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Update with new model info
    config['model'] = new_model_name
    config['model_params'] = new_params
    config['use_feature_selection'] = use_feature_selection

    with open(path, "w") as f:
        yaml.safe_dump(config, f)

    logger.info(f"Updated {yaml_path} with new model '{new_model_name}', params, and use_feature_selection={use_feature_selection}.")

def register_model(
    model_path: str,
    model_name: str,
    register_as_champion: bool = False
) -> None:
    client = MlflowClient()
    version = mlflow.register_model(model_path, model_name).version
    
    # Always tag the model for clarity
    client.set_model_version_tag(model_name, version, "task", "classification")
    
    if register_as_champion:
        client.set_registered_model_alias(model_name, "champion", version)
        logger.info(f"Model version {version} registered and aliased as 'champion'.")
    else:
        logger.info(f"Model version {version} registered without alias.")


def model_selection(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    target_name: str,
                    champion_dict: Dict[str, Any],
                    champion_model: Any,
                    parameters_grid: Dict[str, Any],
                    best_columns,
                    use_feature_selection,
                    n_trials: int = 20) -> Any:

    y_train = np.ravel(y_train[target_name])
    y_test = np.ravel(y_test[target_name])

    initial_results = {}

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.SafeLoader)['tracking']['experiment']['name']
    experiment_id = _get_or_create_experiment_id(experiment_name)
    logger.info(f"Experiment ID: {experiment_id}")

    logger.info('Starting hyperparameter tuning for all models with Optuna')

    best_score = 0
    best_model = None
    best_model_name = None
    best_params = None
    best_run_id = None

    for model_name, base_model in models_dict.items():
        param_grid = parameters_grid[model_name]

        def objective_wrapper(trial):
            return objective(
                trial,
                X_train,
                X_test,
                y_train,
                y_test,
                model_name,
                base_model,
                param_grid,
                best_columns=best_columns
            )

        study = optuna.create_study(direction='maximize')
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
            
            # Log tags
            mlflow.set_tag("task", "Classification")
            mlflow.set_tag("stage", "hyperparameter_tuning_optuna")
            mlflow.set_tag("optimizer_engine", "optuna")
            mlflow.set_tag("model_family", model_name)
            mlflow.set_tag("feature_set_version", 1)
            run_id = mlflow.active_run().info.run_id
            
            # Use champion_callback here to log improvements during tuning
            study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[champion_callback])

            if use_feature_selection:
                X_train_fit = X_train[best_columns]
                X_test_fit = X_test[best_columns]
            else:
                X_train_fit = X_train
                X_test_fit = X_test

            model = base_model.__class__(**study.best_params)
            logger.info(f"Training {model_name} with best parameters: {study.best_params}")
            model.fit(X_train_fit, y_train)

            y_train_pred = model.predict(X_train_fit)
            y_test_pred = model.predict(X_test_fit)

            metrics = {
                'accuracy_train': accuracy_score(y_train, y_train_pred),
                'accuracy_test': accuracy_score(y_test, y_test_pred),
                'f1_score_train': f1_score(y_train, y_train_pred),
                'f1_score_test': f1_score(y_test, y_test_pred),
                'recall_test': recall_score(y_test, y_test_pred),
                'precision_test': precision_score(y_test, y_test_pred)
            }

            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log the model artifact, signature and input example
            signature = infer_signature(X_train_fit, model.predict(X_train_fit))
            input_example = X_train_fit.iloc[:5]
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            if metrics['f1_score_test'] > best_score:
                best_score = metrics['f1_score_test']
                best_model = model
                best_model_name = model_name
                best_params = model.get_params()
                best_run_id = run.info.run_id

            logger.info(f"Best model training complete. Run ID: {run_id}")
            logger.info("Best model accuracy on training set: %0.2f%%", metrics['accuracy_train'] * 100)
            logger.info("Best model accuracy on validation set: %0.2f%%", metrics['accuracy_test'] * 100)
            logger.info("Best model f1score on training set: %0.2f%%", metrics['f1_score_train'] * 100)
            logger.info("Best model f1score on validation set: %0.2f%%", metrics['f1_score_test'] * 100)
            logger.info("Best model precision on validation set: %0.2f%%", metrics['precision_test'] * 100)
            logger.info("Best model recall on validation set: %0.2f%%", metrics['recall_test'] * 100)

            if champion_dict['f1_score_test'] < best_score:
                logger.info(f"New champion model: {best_model_name} with test F1 {best_score:.4f} (previous {champion_dict['f1_score_test']:.4f})")

                # Update YAML
                update_parameters_yaml(
                    yaml_path="conf/base/parameters.yml",
                    new_model_name=best_model_name,
                    new_params=best_params
                )

                # Register the new champion model in MLflow Model Registry
                register_model(
                    model_path=f"runs:/{best_run_id}/model",
                    model_name="hotel_california_model",
                    register_as_champion=True  # ✅ set alias ONLY if truly best
                )

                return best_model, best_columns, metrics

            else:
                logger.info(f"Champion model remains with test F1 {champion_dict['f1_score_test']:.4f}")

                # Optional: Register the current (non-champion) model without alias for audit trail
                register_model(
                    model_path=f"runs:/{best_run_id}/model",
                    model_name="hotel_california_model",
                    register_as_champion=False
                )

                return champion_model, best_columns, {"status": "unchanged"}

