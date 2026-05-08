# src/data_processing.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Support both package-relative and direct invocation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_FRACTION, TEST_SIZE


def load_and_sample_data(
    csv_path=None, target_col=None,
    frac=TRAIN_FRACTION, test_size=TEST_SIZE,
    random_state=42, dataset=None,
):
    """
    Load a dataset either from a sklearn Bunch (dataset=) or from a CSV file.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    if dataset is not None:
        X = dataset.data.copy()
        y = dataset.target.copy()
        feature_names = list(dataset.feature_names)
    else:
        df = pd.read_csv(csv_path)
        df = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
        feature_names = df.drop(columns=[target_col]).columns.tolist()
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


def _add_noise_and_redundancy(X_orig, feature_names_orig, random_state=42):
    """
    Append two noise columns and one redundant column to X_orig, then standard-scale.
    Returns (X_scaled, feature_names).
    """
    n_samples = X_orig.shape[0]
    rng = np.random.RandomState(random_state)

    std_mean = float(np.std(X_orig, axis=0).mean()) if n_samples > 1 else 1.0
    noisy1 = rng.randn(n_samples, 1) * std_mean * 0.5
    noisy2 = rng.uniform(-1, 1, size=(n_samples, 1)) * std_mean * 0.5

    primary_idx = int(np.argmax(np.var(X_orig, axis=0)))
    primary_col = X_orig[:, primary_idx].reshape(-1, 1)
    std_primary = float(np.std(primary_col)) if n_samples > 1 else 1.0
    redundant = primary_col * 0.7 + rng.normal(
        0, 0.05 * max(std_primary, 1e-9), size=(n_samples, 1)
    )

    short_name = feature_names_orig[primary_idx][:10]
    X_out = np.hstack((X_orig, noisy1, noisy2, redundant))
    names_out = list(feature_names_orig) + [
        'noisy_gaussian', 'noisy_uniform', f'redundant_{short_name}'
    ]

    scaler = StandardScaler()
    return scaler.fit_transform(X_out), names_out


def load_iris_noisy(test_size=0.3, random_state=42):
    """Iris dataset with two noise features and one redundant feature appended."""
    data = load_iris()
    X_scaled, feature_names = _add_noise_and_redundancy(
        data.data, list(data.feature_names), random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, data.target, test_size=test_size,
        random_state=random_state, stratify=data.target,
    )
    return X_train, X_test, y_train, y_test, feature_names


def load_wine_noisy(test_size=0.3, random_state=42):
    """Wine dataset with two noise features and one redundant feature appended."""
    data = load_wine()
    X_scaled, feature_names = _add_noise_and_redundancy(
        data.data, list(data.feature_names), random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, data.target, test_size=test_size,
        random_state=random_state, stratify=data.target,
    )
    return X_train, X_test, y_train, y_test, feature_names
