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
import shap
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
import optuna

logger = logging.getLogger(__name__)

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
        X_train_ = X_train[best_columns]
        X_val_ = X_val[best_columns]
    else:
        X_train_, X_val_ = X_train, X_val

    model.fit(X_train_, y_train)
    y_val_pred = model.predict(X_val_)
    f1 = f1_score(y_val, y_val_pred)
    
    # Log metrics and parameters to MLflow for this trial
    mlflow.log_params(params)
    mlflow.log_metric("val_f1_score", f1)

    return f1

def model_selection(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    target_name: str,
                    champion_dict: Dict[str, Any],
                    champion_model: Any,
                    parameters_grid: Dict[str, Any],
                    best_columns,
                    n_trials: int = 20) -> Any:
    
    models_dict = {
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
    }

    initial_results = {}   


    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.SafeLoader)['tracking']['experiment']['name']
    experiment_id = _get_or_create_experiment_id(experiment_name)
    logger.info(f"Experiment ID: {experiment_id}")

    logger.info('Starting first step of model selection: Comparing base models')

    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            y_train = np.ravel(y_train[target_name])
            y_test = np.ravel(y_test[target_name])
            model.fit(X_train, y_train)
            test_score = model.score(X_test, y_test)
            initial_results[model_name] = test_score
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Logged model: {model_name} with test score {test_score:.4f} in run {run_id}")

    best_model_name = max(initial_results, key=initial_results.get)
    logger.info(f"Best base model: {best_model_name} with score {initial_results[best_model_name]:.4f}")
    base_model = models_dict[best_model_name]

    logger.info('Starting hyperparameter tuning on best model with Optuna')

    # Perform hyperparameter tuning using Optuna
    param_grid = parameters_grid['hyperparameters'].get(best_model_name, {})
    def objective_wrapper(trial):
        return objective(
            trial,
            X_train,
            X_test,
            y_train,
            y_test,
            best_model_name,
            base_model,
            parameters_grid,
            best_columns=best_columns
        )
    
    # Set up Optuna study with MLflow integration
    study = optuna.create_study(direction='maximize')
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        y_train = np.ravel(y_train[target_name])
        y_test = np.ravel(y_test[target_name])
        study.optimize(objective_wrapper, n_trials=n_trials)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value

        logger.info(f"Optuna tuning complete. Best F1 score: {best_score:.4f} with params: {best_params}")

        # Refit best model on full training data
        best_model = base_model.__class__(**best_params)
        if best_columns is not None:
            X_train_ = X_train[best_columns]
        else:
            X_train_ = X_train
        best_model.fit(X_train_, y_train)

        pred_score = accuracy_score(y_test, best_model.predict(X_test))
        logger.info(f"Tuned model test accuracy: {pred_score:.4f}")

        mlflow.log_params(best_params)
        mlflow.log_metric("optuna_best_f1_score", best_score)
        mlflow.log_metric("test_accuracy", pred_score)

    if champion_dict['test_score'] < pred_score:
        logger.info(f"New champion model: {best_model_name} with test accuracy {pred_score:.4f} (previous {champion_dict['test_score']:.4f})")
        return {
            'model': best_model,
            'test_score': pred_score,
            'model_name': best_model_name,
            'parameters': best_params
        }
    else:
        logger.info(f"Champion model remains: {champion_dict['model_name']} with test accuracy {champion_dict['test_score']:.4f} (candidate {pred_score:.4f})")
        return champion_dict