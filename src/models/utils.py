# src/models/utils.py

"""
Utility functions for model training, evaluation, and metrics.
"""

import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Simple utility to evaluate a trained model on test data.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# Could add more specialized metrics, data transformations, or pipeline helpers here.
