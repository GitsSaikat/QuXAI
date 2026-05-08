"""Unit tests for src/data_processing.py"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from sklearn.datasets import load_iris

from data_processing import (
    load_and_sample_data,
    _add_noise_and_redundancy,
    load_iris_noisy,
    load_wine_noisy,
)


def test_load_and_sample_data_with_dataset():
    data = load_iris()
    X_train, X_test, y_train, y_test, feature_names = load_and_sample_data(dataset=data)
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert len(feature_names) == data.data.shape[1]
    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(X_test))


def test_load_and_sample_data_labels_numeric():
    data = load_iris()
    _, _, y_train, y_test, _ = load_and_sample_data(dataset=data)
    assert np.issubdtype(y_train.dtype, np.integer) or np.issubdtype(y_train.dtype, np.floating)


def test_add_noise_and_redundancy_output_shape():
    data = load_iris()
    X_scaled, names = _add_noise_and_redundancy(data.data, list(data.feature_names))
    assert X_scaled.shape == (data.data.shape[0], data.data.shape[1] + 3)
    assert len(names) == data.data.shape[1] + 3


def test_add_noise_and_redundancy_no_nans():
    data = load_iris()
    X_scaled, _ = _add_noise_and_redundancy(data.data, list(data.feature_names))
    assert not np.any(np.isnan(X_scaled))


def test_load_iris_noisy_shapes():
    X_train, X_test, y_train, y_test, feature_names = load_iris_noisy()
    n_features = 4 + 3  # original 4 + 3 augmented
    assert X_train.shape[1] == n_features
    assert X_test.shape[1] == n_features
    assert len(feature_names) == n_features


def test_load_wine_noisy_shapes():
    X_train, X_test, y_train, y_test, feature_names = load_wine_noisy()
    n_wine_orig = 13
    assert X_train.shape[1] == n_wine_orig + 3
    assert len(feature_names) == n_wine_orig + 3


def test_load_iris_noisy_no_nans():
    X_train, X_test, y_train, y_test, _ = load_iris_noisy()
    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(X_test))
