import pandas as pd
import logging
from typing import Dict, Any
import numpy as np
import yaml
import pickle
import warnings
warnings.filterwarnings("ignore", category=Warning)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import mlflow
import optuna

logger = logging.getLogger(__name__)

# Get or create an MLflow experiment ID
def _get_or_create_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id


def model_selection(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.DataFrame,
                    y_test: pd.DataFrame,
                    champion_dict: Dict[str, Any],
                    champion_model: pickle.Pickler,
                    parameters: Dict[str, Any]):

    # Models that are going to be explored
    models_dict = {
        'RandomForestClassifier': RandomForestClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
    }

    initial_results = {}

    # Load MLflow experiment name and get its ID
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = _get_or_create_experiment_id(experiment_name)

    # Load model hyperparameter search space
    with open('conf/base/param_grid.yaml') as f:
        param_grid = yaml.safe_load(f)['hyperparameters']

    logger.info('Starting first step of model selection : Comparing between model types')

    # Train and evaluate each model with default parameters
    for model_name, ModelClass in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            model = ModelClass()
            y_train = np.ravel(y_train)
            model.fit(X_train, y_train)
            initial_results[model_name] = model.score(X_test, y_test)
            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}")

    # Select best performing model by initial accuracy
    best_model_name = max(initial_results, key=initial_results.get)
    best_model_class = models_dict[best_model_name]

    logger.info(f"Best model is {best_model_name} with score {initial_results[best_model_name]}")
    logger.info('Starting second step of model selection : Hyperparameter tuning using Optuna')

    # Define search space from param_grid
    search_space = param_grid[best_model_name]

    # Optuna objective function to optimize accuracy
    def objective(trial):
        trial_params = {
            param: trial.suggest_categorical(param, values)
            for param, values in search_space.items()
        }
        model = best_model_class(**trial_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    # Run Optuna study
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        best_params['random_state'] = 42

        best_model = best_model_class(**best_params)
        best_model.fit(X_train, y_train)

        # Log best parameters and score to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", study.best_value)

    # Final evaluation of best model
    pred_score = accuracy_score(y_test, best_model.predict(X_test))

    # Compare with previous champion model
    if champion_dict['test_score'] < pred_score:
        logger.info(f"New champion model is {best_model_name} with score: {pred_score} vs {champion_dict['test_score']}")
        return best_model
    else:
        logger.info(f"Champion model is still {champion_dict['regressor']} with score: {champion_dict['test_score']} vs {pred_score}")
        return champion_model
