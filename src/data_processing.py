# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import TRAIN_FRACTION, TEST_SIZE

def load_and_sample_data(csv_path, target_col, frac=TRAIN_FRACTION, test_size=TEST_SIZE, random_state=42):
    """
    Loads CSV, samples a fraction of rows, and splits into train/test.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(csv_path)
    df = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    feature_names = df.drop(columns=[target_col]).columns.tolist()

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Convert labels to numeric if necessary
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names
