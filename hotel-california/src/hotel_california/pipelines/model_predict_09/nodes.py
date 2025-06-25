import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)

def model_predict(X: pd.DataFrame, y, model, columns) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict using the trained model and evaluate metrics if ground truth is provided.

    Args:
        X (pd.DataFrame): Serving observations.
        y: Ground truth labels (optional, can be None).
        model: Trained model with a predict method (e.g., scikit-learn estimator).
        columns: List of feature column names to use for prediction.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame with predictions and a dictionary with descriptive statistics and metrics if y is provided.
    """
    # Make a copy to avoid modifying the original DataFrame
    X_pred = X.copy()

    # Predict
    y_pred = model.predict(X_pred[columns])

    # Create dataframe with predictions
    X_pred['y_pred'] = y_pred
    
    # Create dictionary with predictions
    describe_servings = X_pred.describe().to_dict()

    # If y is provided, calculate metrics (for classification)
    if y is not None:
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'precision': precision_score(y, y_pred)
        }
        describe_servings.update(metrics)
        logger.info('Accuracy: %.4f', metrics['accuracy'])
        logger.info('F1 Score: %.4f', metrics['f1_score'])
        logger.info('Recall: %.4f', metrics['recall'])
        logger.info('Precision: %.4f', metrics['precision'])

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))
    return X_pred, describe_servings