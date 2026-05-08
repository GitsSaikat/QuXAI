# src/models/quantum_models.py

import numpy as np
import pennylane as qml
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

MAX_QUBITS = 8

CLASSICAL_PARAMS = {
    "RandomForestClassifier":       {'n_estimators': 100, 'random_state': 42},
    "LogisticRegression":           {'max_iter': 1000, 'random_state': 42},
    "DecisionTreeClassifier":       {'random_state': 42},
    "GaussianNB":                   {},
    "AdaBoostClassifier":           {'n_estimators': 50, 'random_state': 42, 'algorithm': 'SAMME'},
    "GradientBoostingClassifier":   {'n_estimators': 100, 'random_state': 42},
    "LinearDiscriminantAnalysis":   {},
    "Perceptron":                   {'max_iter': 1000, 'random_state': 42},
    "RidgeClassifier":              {'random_state': 42},
    "ExtraTreesClassifier":         {'n_estimators': 100, 'random_state': 42},
}


def sample_quantum_feature_map(x, num_qubits):
    for i in range(num_qubits):
        qml.RX(float(x[i]), wires=i)


def _train_amplitude_model_generic(
    X_train, y_train, clf_class, clf_params, model_type_str,
    feature_map=sample_quantum_feature_map,
):
    num_qubits = min(X_train.shape[1], MAX_QUBITS)
    X_sub = X_train[:, :num_qubits]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(x):
        feature_map(x, num_qubits)
        return qml.state()

    # Use squared amplitudes (probabilities) — physically interpretable and sum to 1
    X_q = np.array([np.abs(circuit(x)) ** 2 for x in X_sub])
    clf = clf_class(**clf_params)
    clf.fit(X_q, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    clf.hqml_model_type = model_type_str
    return clf


# ---------------------------------------------------------------------------
# Amplitude-encoded classifiers (one-liners via the generic trainer)
# ---------------------------------------------------------------------------

def train_qrf(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, RandomForestClassifier,
        CLASSICAL_PARAMS["RandomForestClassifier"], 'qrf', feature_map)

def train_qlogistic(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, LogisticRegression,
        CLASSICAL_PARAMS["LogisticRegression"], 'qlogistic', feature_map)

def train_qdt(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, DecisionTreeClassifier,
        CLASSICAL_PARAMS["DecisionTreeClassifier"], 'qdt', feature_map)

def train_qnb(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, GaussianNB,
        CLASSICAL_PARAMS["GaussianNB"], 'qnb', feature_map)

def train_qada(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, AdaBoostClassifier,
        CLASSICAL_PARAMS["AdaBoostClassifier"], 'qada', feature_map)

def train_qgb(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, GradientBoostingClassifier,
        CLASSICAL_PARAMS["GradientBoostingClassifier"], 'qgb', feature_map)

def train_qlda(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, LinearDiscriminantAnalysis,
        CLASSICAL_PARAMS["LinearDiscriminantAnalysis"], 'qlda', feature_map)

def train_qperceptron(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, Perceptron,
        CLASSICAL_PARAMS["Perceptron"], 'qperceptron', feature_map)

def train_qridge(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, RidgeClassifier,
        CLASSICAL_PARAMS["RidgeClassifier"], 'qridge', feature_map)

def train_qextra(X_train, y_train, feature_map=sample_quantum_feature_map):
    return _train_amplitude_model_generic(
        X_train, y_train, ExtraTreesClassifier,
        CLASSICAL_PARAMS["ExtraTreesClassifier"], 'qextra', feature_map)


# ---------------------------------------------------------------------------
# Kernel-encoded classifiers
# ---------------------------------------------------------------------------

def _make_kernel_fn(feature_map_circuit):
    def kernel_fn(x1, x2):
        s1 = feature_map_circuit(x1)
        s2 = feature_map_circuit(x2)
        return np.abs(np.dot(s1.conj(), s2)) ** 2
    return kernel_fn


def train_qsvm(X_train, y_train, feature_map=sample_quantum_feature_map):
    num_qubits = min(X_train.shape[1], MAX_QUBITS)
    X_sub = X_train[:, :num_qubits]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def fmc(x):
        feature_map(x, num_qubits)
        return qml.state()

    train_kernel = qml.kernels.square_kernel_matrix(X_sub, _make_kernel_fn(fmc))
    svm = SVC(kernel='precomputed')
    svm.fit(train_kernel, y_train)
    svm.num_qubits = num_qubits
    svm.quantum_feature_map = feature_map
    svm.hqml_model_type = 'qsvm'
    return svm


def train_qknn(X_train, y_train, feature_map=sample_quantum_feature_map, n_neighbors=3):
    num_qubits = min(X_train.shape[1], MAX_QUBITS)
    X_sub = X_train[:, :num_qubits]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def fmc(x):
        feature_map(x, num_qubits)
        return qml.state()

    train_kernel = qml.kernels.square_kernel_matrix(X_sub, _make_kernel_fn(fmc))
    dist_matrix = np.abs(1 - train_kernel)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='precomputed')
    knn.fit(dist_matrix, y_train)
    knn.num_qubits = num_qubits
    knn.quantum_feature_map = feature_map
    knn.hqml_model_type = 'qknn'
    return knn


def train_qsvc_poly(X_train, y_train, feature_map=sample_quantum_feature_map):
    num_qubits = min(X_train.shape[1], MAX_QUBITS)
    X_sub = X_train[:, :num_qubits]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def fmc(x):
        feature_map(x, num_qubits)
        return qml.state()

    def poly_kernel(x1, x2):
        base = np.abs(np.dot(fmc(x1).conj(), fmc(x2))) ** 2
        return (base + 1) ** 3

    train_kernel = qml.kernels.square_kernel_matrix(X_sub, poly_kernel)
    clf = SVC(kernel='precomputed')
    clf.fit(train_kernel, y_train)
    clf.num_qubits = num_qubits
    clf.quantum_feature_map = feature_map
    clf.hqml_model_type = 'qsvc_poly'
    return clf
