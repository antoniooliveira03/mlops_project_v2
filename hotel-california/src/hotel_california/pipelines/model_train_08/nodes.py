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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from mlflow.models.signature import infer_signature

logger = logging.getLogger(__name__)

model_registry = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression
}

def register_model(
    model_path: str,
    model_name: str,
    model_tag: str = None,
    model_alias: str = None
) -> None:
    """
    Register a model in the MLflow Model Registry.
    
    Args:
        model_path (str): The path to the model to register.
        model_name (str): The name of the model to register.
        model_tag (str): A tag to add to the model.
        model_version (int): The version of the model to register.
        model_alias (str): An alias to add to the model.
    """
    
    client = MlflowClient()
    
    # Registering the model
    version = mlflow.register_model(
        model_path, model_name
    )
    
    if any(var is not None for var in [model_tag, model_alias]):
    
        if model_tag:
            client.set_registered_model_tag(
                model_name, "task", "classification"
            )
        
        # Creating alias
        if model_alias:
            client.set_registered_model_alias(
                model_name, model_alias, version.version
            )


def model_train(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.DataFrame, 
    y_test: pd.DataFrame,
    target_name: str, 
    use_feature_selection: bool,
    model_name: str,
    parameters: Dict[str, Any],
    best_columns
):

    # Load MLflow configuration
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.safe_load(f)['tracking']['experiment']['name']
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    try:
        with open(os.path.join('data', '06_models', 'champion_model.pkl'), 'rb') as f:
            classifier = pickle.load(f)
    except:
        model_class = model_registry[model_name]
        classifier = model_class(**parameters)

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
        run_id = run.info.run_id

        # Log datasets
        mlflow.log_input(mlflow.data.from_pandas(X_train, name="X_train_final"), context="training_X_train")
        mlflow.log_input(mlflow.data.from_pandas(X_test, name="X_val_final"), context="training_X_val")
        mlflow.log_input(mlflow.data.from_pandas(y_train, name="y_train_final"), context="training_y_train")
        mlflow.log_input(mlflow.data.from_pandas(y_test, name="y_val_final"), context="training_y_val")

        mlflow.set_tag("model_name", classifier.__class__.__name__)
        mlflow.set_tag("experiment_name", experiment_name)
        mlflow.set_tag("num_train_samples", str(X_train.shape[0]))
        mlflow.set_tag("num_test_samples", str(X_test.shape[0]))

        if use_feature_selection:
            mlflow.set_tag("selected_features", ", ".join(best_columns))
            X_train = X_train[best_columns]
            X_test = X_test[best_columns]
            mlflow.set_tag("feature_selection_used", str(best_columns))
            mlflow.set_tag("num_features", str(X_train.shape[1]))

            # Save feature list
            feature_path = "selected_features.pkl"
            with open(feature_path, "wb") as f:
                pickle.dump(best_columns, f)
            mlflow.log_artifact(feature_path)

        y_train = np.ravel(y_train[target_name])
        y_test = np.ravel(y_test[target_name])

        logger.info("Training the model...")
        model = classifier.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

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

        # Create figure with confusion matrix and feature importance
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[0], colorbar=False)
        axes[0].set_title("Confusion Matrix")

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False).head(15)

            axes[1].barh(importance_df['feature'], importance_df['importance'])
            axes[1].set_title("Top 15 Feature Importances")
            axes[1].invert_yaxis()
        else:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'No feature_importances_ attribute',
                        ha='center', va='center', fontsize=12)

        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix_feature_importance.png")

        # SHAP explanations
        logger.info("Generating SHAP explanations...")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        shap.initjs()
        shap_fig = plt.figure()
        shap.summary_plot(shap_values[:,:,1], X_train, show=False)
        plt.tight_layout()
        shap_fig_path = "shap_summary_plot.png"
        plt.savefig(shap_fig_path)
        mlflow.log_artifact(shap_fig_path)
        plt.close(shap_fig)

        # Log model explicitly with signature
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Register model in MLflow registry
        register_model(
            model_path=f"runs:/{run_id}/model",
            model_name="final_model",
            model_tag="production",
            model_alias="champion"
        )

        logger.info(f"Model training complete. Run ID: {run_id}")
        logger.info("Model accuracy on training set: %0.2f%%", metrics['accuracy_train'] * 100)
        logger.info("Model accuracy on validation set: %0.2f%%", metrics['accuracy_test'] * 100)
        logger.info("Model f1score on training set: %0.2f%%", metrics['f1_score_train'] * 100)
        logger.info("Model f1score on validation set: %0.2f%%", metrics['f1_score_test'] * 100)
        logger.info("Model precision on validation set: %0.2f%%", metrics['precision_test'] * 100)
        logger.info("Model recall on validation set: %0.2f%%", metrics['recall_test'] * 100)

    return model, X_train.columns, metrics, fig, shap_fig
