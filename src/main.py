# src/main.py

import numpy as np
from data_processing import load_and_sample_data
from models.quantum_models import (train_qrf, train_qsvm, train_qknn, 
    train_qlogistic, train_qdt, train_qnb, train_qada, train_qgb,
    train_qlda, train_qperceptron, train_qridge, train_qsvc_poly, train_qextra)
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
    chosen_model_type = 'qridge'  # Options: qrf, qsvm, qknn, qlogistic, qdt, qnb, qada, qgb, qlda, qperceptron, qridge, qsvc_poly, qextra
    if chosen_model_type == 'qrf':
        model = train_qrf(X_train, y_train)
    elif chosen_model_type == 'qsvm':
        model = train_qsvm(X_train, y_train)
    elif chosen_model_type == 'qknn':
        model = train_qknn(X_train, y_train)
    elif chosen_model_type == 'qlogistic':
        model = train_qlogistic(X_train, y_train)
    elif chosen_model_type == 'qdt':
        model = train_qdt(X_train, y_train)
    elif chosen_model_type == 'qnb':
        model = train_qnb(X_train, y_train)
    elif chosen_model_type == 'qada':
        model = train_qada(X_train, y_train)
    elif chosen_model_type == 'qgb':
        model = train_qgb(X_train, y_train)
    elif chosen_model_type == 'qlda':
        model = train_qlda(X_train, y_train)
    elif chosen_model_type == 'qperceptron':
        model = train_qperceptron(X_train, y_train)
    elif chosen_model_type == 'qridge':
        model = train_qridge(X_train, y_train)
    elif chosen_model_type == 'qsvc_poly':
        model = train_qsvc_poly(X_train, y_train)
    elif chosen_model_type == 'qextra':
        model = train_qextra(X_train, y_train)
    else:
        raise ValueError("Invalid model type. Check available options.")

    # 3. Evaluate model on test set
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
    Custom evaluation routine handling amplitude and kernel-based classifiers.
    """
    from sklearn.metrics import accuracy_score
    import numpy as np
    import pennylane as qml

    # Group for amplitude-based classifiers
    amplitude_models = ['qrf', 'qlogistic', 'qdt', 'qnb', 'qada', 'qgb', 'qlda', 'qperceptron', 'qridge', 'qextra']

    # Kernel-based models
    kernel_models = ['qsvm', 'qsvc_poly']
    
    if model_type in amplitude_models:
        num_qubits = model.num_qubits
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def circuit(x):
            model.quantum_feature_map(x, num_qubits)
            return qml.state()
        X_test_q = np.array([np.abs(circuit(x)) for x in X_test])
        y_pred = model.predict(X_test_q)
        return accuracy_score(y_test, y_pred)

    elif model_type in kernel_models:
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
        dist_matrix = np.abs(dist_matrix)
        y_pred = model.predict(dist_matrix)
        return accuracy_score(y_test, y_pred)

    else:
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)


def _predict(self, X):
    """
    Predict on classical X by converting or building kernel as needed.
    """
    if self.model_type == 'qrf':
        X_q = self._encode_qrf(X)
        return self.pretrained_model.predict(X_q)
    elif self.model_type == 'qnb':  # Added branch for quantum Naive Bayes.
        X_q = self._encode_qrf(X)
        return self.pretrained_model.predict(X_q)
    elif self.model_type == 'qsvm':
        if X.shape == self.X_ref_.shape and np.allclose(X, self.X_ref_):
            return self.pretrained_model.predict(self.kernel_ref_)
        else:
            return self._predict_kernel(X, is_knn=False)
    elif self.model_type == 'qknn':
        if X.shape == self.X_ref_.shape and np.allclose(X, self.X_ref_):
            dist_matrix = 1 - self.kernel_ref_
            dist_matrix = np.abs(dist_matrix)
            return self.pretrained_model.predict(dist_matrix)
        else:
            return self._predict_kernel(X, is_knn=True)
    else:
        return self.pretrained_model.predict(X)


if __name__ == "__main__":
    main()
