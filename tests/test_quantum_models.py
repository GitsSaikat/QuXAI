"""Unit tests for src/models/quantum_models.py"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from models.quantum_models import (
    sample_quantum_feature_map,
    _train_amplitude_model_generic,
    train_qrf, train_qdt, train_qnb, train_qlogistic,
    MAX_QUBITS,
)
from sklearn.ensemble import RandomForestClassifier

# Toy dataset: 4 features, 20 samples, 2 classes
rng = np.random.RandomState(0)
X_toy = rng.randn(20, 4).astype(np.float64)
y_toy = (X_toy[:, 0] > 0).astype(int)


def test_max_qubits_constant():
    assert MAX_QUBITS == 8


def test_train_qrf_attributes():
    model = train_qrf(X_toy, y_toy)
    assert hasattr(model, 'num_qubits')
    assert hasattr(model, 'quantum_feature_map')
    assert hasattr(model, 'hqml_model_type')
    assert model.hqml_model_type == 'qrf'


def test_train_qrf_num_qubits_capped():
    X_big = rng.randn(20, 12).astype(np.float64)
    model = train_qrf(X_big, y_toy)
    assert model.num_qubits <= MAX_QUBITS


def test_train_qrf_predict_shape():
    model = train_qrf(X_toy, y_toy)
    import pennylane as qml
    num_qubits = model.num_qubits
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(x):
        model.quantum_feature_map(x, num_qubits)
        return qml.state()

    X_q = np.array([np.abs(circuit(x)) ** 2 for x in X_toy[:, :num_qubits]])
    preds = model.predict(X_q)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})


def test_train_qdt_attributes():
    model = train_qdt(X_toy, y_toy)
    assert model.hqml_model_type == 'qdt'
    assert model.num_qubits == min(X_toy.shape[1], MAX_QUBITS)


def test_train_qnb_attributes():
    model = train_qnb(X_toy, y_toy)
    assert model.hqml_model_type == 'qnb'


def test_train_qlogistic_attributes():
    model = train_qlogistic(X_toy, y_toy)
    assert model.hqml_model_type == 'qlogistic'


def test_amplitude_probabilities_sum_to_one():
    """Squared amplitudes must sum to 1 for each sample."""
    import pennylane as qml
    model = train_qrf(X_toy, y_toy)
    num_qubits = model.num_qubits
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(x):
        model.quantum_feature_map(x, num_qubits)
        return qml.state()

    for x in X_toy[:5, :num_qubits]:
        probs = np.abs(circuit(x)) ** 2
        assert abs(probs.sum() - 1.0) < 1e-6, f"Probabilities do not sum to 1: {probs.sum()}"
