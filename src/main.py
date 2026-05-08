# src/main.py

import os
import sys

# Allow running as `python src/main.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pennylane as qml
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from data_processing import load_and_sample_data
from models.quantum_models import (
    train_qrf, train_qsvm, train_qknn,
    train_qlogistic, train_qdt, train_qnb, train_qada, train_qgb,
    train_qlda, train_qperceptron, train_qridge, train_qsvc_poly, train_qextra,
)
from models.utils import evaluate_model
from explainers.medley_explainer import MEDLEY
from plotting.visualizations import plot_medley_scores

_TRAINERS = {
    'qrf': train_qrf, 'qsvm': train_qsvm, 'qknn': train_qknn,
    'qlogistic': train_qlogistic, 'qdt': train_qdt, 'qnb': train_qnb,
    'qada': train_qada, 'qgb': train_qgb, 'qlda': train_qlda,
    'qperceptron': train_qperceptron, 'qridge': train_qridge,
    'qsvc_poly': train_qsvc_poly, 'qextra': train_qextra,
}

_AMPLITUDE_MODELS = {
    'qrf', 'qlogistic', 'qdt', 'qnb', 'qada',
    'qgb', 'qlda', 'qperceptron', 'qridge', 'qextra',
}
_KERNEL_MODELS = {'qsvm', 'qsvc_poly'}


def evaluate_model_generic(model, model_type, X_train, X_test, y_train, y_test):
    num_qubits = model.num_qubits
    qfm = model.quantum_feature_map
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(x):
        qfm(x, num_qubits)
        return qml.state()

    if model_type in _AMPLITUDE_MODELS:
        X_test_sub = X_test[:, :num_qubits]
        X_test_q = np.array([np.abs(circuit(x)) ** 2 for x in X_test_sub])
        return accuracy_score(y_test, model.predict(X_test_q))

    def kernel_fn(x1, x2):
        s1 = circuit(x1)
        s2 = circuit(x2)
        return np.abs(np.dot(s1.conj(), s2)) ** 2

    if model_type in _KERNEL_MODELS:
        X_train_sub = X_train[:, :num_qubits]
        X_test_sub = X_test[:, :num_qubits]
        test_kernel = qml.kernels.kernel_matrix(X_test_sub, X_train_sub, kernel_fn)
        return accuracy_score(y_test, model.predict(test_kernel))

    if model_type == 'qknn':
        X_train_sub = X_train[:, :num_qubits]
        X_test_sub = X_test[:, :num_qubits]
        test_kernel = qml.kernels.kernel_matrix(X_test_sub, X_train_sub, kernel_fn)
        dist_matrix = np.abs(1 - test_kernel)
        return accuracy_score(y_test, model.predict(dist_matrix))

    return accuracy_score(y_test, model.predict(X_test))


def main():
    # 1. Load data from sklearn (iris.csv was removed from the repo)
    X_train, X_test, y_train, y_test, feature_names = load_and_sample_data(
        dataset=load_iris()
    )

    # 2. Choose a model
    chosen_model_type = 'qada'
    if chosen_model_type not in _TRAINERS:
        raise ValueError(f"Unknown model type '{chosen_model_type}'. Options: {list(_TRAINERS)}")
    model = _TRAINERS[chosen_model_type](X_train, y_train)

    # 3. Evaluate
    acc = evaluate_model_generic(model, chosen_model_type, X_train, X_test, y_train, y_test)
    print(f"Test Accuracy for {chosen_model_type.upper()}: {acc:.3f}")

    # 4. Explain with MEDLEY
    explainer = MEDLEY(
        pretrained_model=model,
        model_type=chosen_model_type,
        quantum_feature_map=model.quantum_feature_map,
        n_repeats=5,
    )
    explainer.fit(X_train, y_train)

    sample = X_test[2].reshape(1, -1)
    importances = explainer.interpret(sample)
    print("MEDLEY Importances:", importances)

    # 5. Plot
    plot_medley_scores(
        feature_names, importances,
        title=f"{chosen_model_type.upper()} MEDLEY",
        outfile="MEDLEY_Score.png",
    )


if __name__ == "__main__":
    main()
