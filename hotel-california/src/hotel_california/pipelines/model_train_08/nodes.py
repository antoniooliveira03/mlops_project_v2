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
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from mlflow.models.signature import infer_signature

logger = logging.getLogger(__name__)

model_registry = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
    "AdaBoostClassifier": AdaBoostClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier,
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
        mlflow.get_parameters(parameters)

    with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
        run_id = run.info.run_id


        mlflow.set_tag("model_name", classifier.__class__.__name__)
        mlflow.set_tag("experiment_name", experiment_name)
        mlflow.set_tag("num_train_samples", str(X_train.shape[0]))
        mlflow.set_tag("num_test_samples", str(X_test.shape[0]))

        # Log model parameters
        if parameters is not None:
            mlflow.log_params(parameters)

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

        logger.info("Starting feature selection and data preparation...")
        if use_feature_selection:
            logger.info(f"Selected features: {best_columns}")
        else:
            logger.info("No feature selection applied. Using all features.")

        logger.info("Training the model...")
        model = classifier.fit(X_train, y_train)
        logger.info("Model training complete.")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        logger.info("Predictions complete.")

        metrics = {
            'accuracy_train': accuracy_score(y_train, y_train_pred),
            'accuracy_test': accuracy_score(y_test, y_test_pred),
            'f1_score_train': f1_score(y_train, y_train_pred),
            'f1_score_test': f1_score(y_test, y_test_pred),
            'recall_test': recall_score(y_test, y_test_pred),
            'precision_test': precision_score(y_test, y_test_pred)
        }

        logger.info("Logging metrics to MLflow...")
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Create figure with confusion matrix and feature importance
        logger.info("Creating confusion matrix and feature importance plots...")
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
        logger.info("Confusion matrix and feature importance plots created.")
        mlflow.log_figure(fig, "confusion_matrix_feature_importance.png")
        logger.info("Figure logged to MLflow.")
        extra_plot = fig

        # SHAP
        logger.info("Starting SHAP explainer selection...")
        # Auto-select appropriate SHAP explainer
        try:
            if hasattr(model, "predict_proba") and "tree" in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
                logger.info("TreeExplainer selected for SHAP.")
            else:
                explainer = shap.Explainer(model, X_train)
                logger.info("Generic SHAP Explainer selected.")
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            return None

        logger.info("Computing SHAP values...")
        n_shap_features = 15

        # Ensure sample_X columns and dtypes match training data exactly
        sample_X = X_train.copy()
        if sample_X.shape[0] > 1000:
            sample_X = sample_X.sample(n=1000, random_state=42)
        sample_X = sample_X[X_train.columns]  # enforce column order
        sample_X = sample_X.astype(X_train.dtypes.to_dict())  # enforce dtypes

        shap_values = explainer(sample_X)
        logger.info("SHAP values computed.")

        # Handle binary classification with 3D shap_values (samples, features, classes)
        if len(shap_values.values.shape) == 3:
            shap_vals = shap_values.values[:, :, 1]
        else:
            shap_vals = shap_values.values

        feature_importance = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": sample_X.columns,
            "importance": feature_importance
        }).sort_values(by="importance", ascending=False)

        top_features = importance_df.head(n_shap_features)["feature"].tolist()
        logger.info(f"Top {n_shap_features} features for SHAP: {top_features}")

        logger.info("Plotting SHAP summary and feature importance...")
        shap.initjs()
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # SHAP summary plot (top features only)
        plt.sca(axs[0])
        shap.summary_plot(
            shap_vals[:, [sample_X.columns.get_loc(f) for f in top_features]],
            sample_X[top_features],
            feature_names=top_features,
            show=False,
            plot_size=None,
            color_bar=True
        )
        axs[0].set_title("SHAP Summary Plot (Top 10 Features)")

        importance_df.head(n_shap_features).plot(
            kind="barh",
            x="feature",
            y="importance",
            ax=axs[1],
            legend=False
        )
        axs[1].invert_yaxis()
        axs[1].set_title("Mean Absolute SHAP Feature Importance (Top 10)")
        axs[1].set_xlabel("Importance")

        plt.tight_layout()
        shap_fig = fig

        # Log model explicitly with signature
        logger.info("Logging model to MLflow with signature...")
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        logger.info("Model logged to MLflow.")

        # Register model in MLflow registry
        logger.info("Registering model in MLflow registry...")
        register_model(
            model_path=f"runs:/{run_id}/model",
            model_name="hotel_california_model",
            model_tag="production",
            model_alias="champion"
        )
        logger.info("Model registered in MLflow registry.")

        logger.info(f"Model training complete. Run ID: {run_id}")
        logger.info("Model accuracy on training set: %0.2f%%", metrics['accuracy_train'] * 100)
        logger.info("Model accuracy on validation set: %0.2f%%", metrics['accuracy_test'] * 100)
        logger.info("Model f1score on training set: %0.2f%%", metrics['f1_score_train'] * 100)
        logger.info("Model f1score on validation set: %0.2f%%", metrics['f1_score_test'] * 100)
        logger.info("Model precision on validation set: %0.2f%%", metrics['precision_test'] * 100)
        logger.info("Model recall on validation set: %0.2f%%", metrics['recall_test'] * 100)

    return model, X_train.columns, metrics, extra_plot, shap_fig