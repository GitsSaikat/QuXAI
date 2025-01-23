# src/main.py

import numpy as np
from data_processing import load_and_sample_data
from models.quantum_models import train_qrf, train_qsvm, train_qknn
from models.utils import evaluate_model
from explainers.medley_explainer import MEDLEY
from plotting.visualizations import plot_medley_scores

def main():
    """
    Main entry point for user interactions:
      1. Load data
      2. Train model
      3. Evaluate
      4. Explain (MEDLEY)
      5. Visualize explanations
    """

    # 1. Load Data
    csv_path = "winequality-white.csv"
    target_col = "quality"
    X_train, X_test, y_train, y_test, feature_names = load_and_sample_data(csv_path, target_col)

    # 2. Choose a model
    chosen_model_type = 'qknn'  # 'qrf', 'qsvm', or 'qknn'
    if chosen_model_type == 'qrf':
        model = train_qrf(X_train, y_train)
    elif chosen_model_type == 'qsvm':
        model = train_qsvm(X_train, y_train)
    elif chosen_model_type == 'qknn':
        model = train_qknn(X_train, y_train)
    else:
        raise ValueError("Invalid model type. Choose from: 'qrf', 'qsvm', 'qknn'.")

    # 3. Evaluate model on test set
    # Note: For QRF, QSVM, QKNN, you might need to handle 
    # the conversion to quantum kernel or amplitude states
    # in evaluate_model or in a custom function. 
    acc = evaluate_model_generic(model, chosen_model_type, X_train, X_test, y_train, y_test)
    print(f"Test Accuracy for {chosen_model_type.upper()}: {acc:.3f}")

    # 4. Explain with MEDLEY
    explainer = MEDLEY(pretrained_model=model,
                       model_type=chosen_model_type,
                       quantum_feature_map=model.quantum_feature_map,
                       n_repeats=5)
    explainer.fit(X_train, y_train)

    sample_idx = 2
    sample = X_test[sample_idx].reshape(1, -1)
    importances = explainer.interpret(sample)
    print("MEDLEY Importances:", importances)

    # 5. Plot results
    plot_medley_scores(feature_names, importances,
                       title=f"{chosen_model_type.upper()} MEDLEY",
                       outfile="MEDLEY_Score.png")


def evaluate_model_generic(model, model_type, X_train, X_test, y_train, y_test):
    """
    Example of a custom evaluation routine that handles
    QRF, QSVM, or QKNN appropriately.
    """
    from sklearn.metrics import accuracy_score
    import numpy as np
    import pennylane as qml

    # If QRF => convert X_test to amplitude vectors
    if model_type == 'qrf':
        num_qubits = model.num_qubits
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def circuit(x):
            model.quantum_feature_map(x, num_qubits)
            return qml.state()

        X_test_q = np.array([np.abs(circuit(x)) for x in X_test])
        y_pred = model.predict(X_test_q)
        return accuracy_score(y_test, y_pred)

    # If QSVM => build NxN kernel for X_test vs X_train
    elif model_type == 'qsvm':
        num_qubits = model.num_qubits
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def feature_map_circuit(x):
            model.quantum_feature_map(x, num_qubits)
            return qml.state()

        def kernel_fn(x1, x2):
            s1 = feature_map_circuit(x1)
            s2 = feature_map_circuit(x2)
            return np.abs(np.dot(s1.conj(), s2)) ** 2

        test_kernel = qml.kernels.kernel_matrix(X_test, X_train, kernel_fn)
        y_pred = model.predict(test_kernel)
        return accuracy_score(y_test, y_pred)

    # If QKNN => distance = 1 - kernel
    elif model_type == 'qknn':
        num_qubits = model.num_qubits
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def feature_map_circuit(x):
            model.quantum_feature_map(x, num_qubits)
            return qml.state()

        def kernel_fn(x1, x2):
            s1 = feature_map_circuit(x1)
            s2 = feature_map_circuit(x2)
            return np.abs(np.dot(s1.conj(), s2)) ** 2

        test_kernel = qml.kernels.kernel_matrix(X_test, X_train, kernel_fn)
        dist_matrix = 1 - test_kernel
        dist_matrix = np.abs(dist_matrix)  # clamp negativity
        y_pred = model.predict(dist_matrix)
        return accuracy_score(y_test, y_pred)

    else:
        # If we had a classical model or fallback scenario
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    main()
