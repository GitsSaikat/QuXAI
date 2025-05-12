# src/explainers/medley_explainer.py

import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score

class MEDLEY:

    def __init__(self, pretrained_model, model_type, quantum_feature_map=None, n_repeats=5):
        self.pretrained_model = pretrained_model
        self.model_type = model_type
        self.quantum_feature_map = quantum_feature_map
        self.n_repeats = n_repeats

        self.X_ref_ = None
        self.y_ref_ = None
        self.num_features_ = None
        self.num_qubits_ = None
        self.kernel_ref_ = None

    def fit(self, X, y):
        self.X_ref_ = X
        self.y_ref_ = y
        self.num_features_ = X.shape[1]
        self.num_qubits_ = self.num_features_

        # If QKNN/QSVM, precompute NxN kernel for reference
        if self.model_type in ['qsvm', 'qknn'] and self.quantum_feature_map:
            dev = qml.device("default.qubit", wires=self.num_qubits_)

            @qml.qnode(dev)
            def feature_map_circuit(x):
                self.quantum_feature_map(x, self.num_qubits_)
                return qml.state()

            def kernel_fn(x1, x2):
                state1 = feature_map_circuit(x1)
                state2 = feature_map_circuit(x2)
                return np.abs(np.dot(state1.conj(), state2)) ** 2

            self.kernel_ref_ = qml.kernels.kernel_matrix(X, X, kernel_fn)

        return self

    def _encode_qrf(self, X):
        """
        For QRF, convert X to quantum states of shape [n, 2^num_qubits].
        """
        dev = qml.device("default.qubit", wires=self.num_qubits_)

        @qml.qnode(dev)
        def circuit(x):
            self.quantum_feature_map(x, self.num_qubits_)
            return qml.state()

        X_q = np.array([np.abs(circuit(x)) for x in X])
        return X_q

    def _predict(self, X):
        """
        Predict on input X by building the quantum representation for new models.
        """
        import numpy as np
        import pennylane as qml

        # Define groups for different model types.
        amplitude_models = ['qrf', 'qlogistic', 'qdt', 'qnb', 'qada', 'qgb', 'qlda', 'qperceptron', 'qridge', 'qextra']
        kernel_models = ['qsvm', 'qsvc_poly']
        
        if self.model_type in amplitude_models:
            num_qubits = self.pretrained_model.num_qubits
            dev = qml.device("default.qubit", wires=num_qubits)

            @qml.qnode(dev)
            def circuit(x):
                self.pretrained_model.quantum_feature_map(x, num_qubits)
                return qml.state()
            X_q = np.array([np.abs(circuit(x)) for x in X])
            return self.pretrained_model.predict(X_q)

        elif self.model_type in kernel_models:
            num_qubits = self.pretrained_model.num_qubits
            dev = qml.device("default.qubit", wires=num_qubits)

            @qml.qnode(dev)
            def feature_map_circuit(x):
                self.pretrained_model.quantum_feature_map(x, num_qubits)
                return qml.state()
                
            def kernel_fn(x1, x2):
                s1 = feature_map_circuit(x1)
                s2 = feature_map_circuit(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2
            
            test_kernel = qml.kernels.kernel_matrix(X, self.X_ref_, kernel_fn)
            return self.pretrained_model.predict(test_kernel)

        elif self.model_type == 'qknn':
            num_qubits = self.pretrained_model.num_qubits
            dev = qml.device("default.qubit", wires=num_qubits)

            @qml.qnode(dev)
            def feature_map_circuit(x):
                self.pretrained_model.quantum_feature_map(x, num_qubits)
                return qml.state()

            def kernel_fn(x1, x2):
                s1 = feature_map_circuit(x1)
                s2 = feature_map_circuit(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2

            test_kernel = qml.kernels.kernel_matrix(X, self.X_ref_, kernel_fn)
            dist_matrix = 1 - test_kernel
            dist_matrix = np.abs(dist_matrix)
            return self.pretrained_model.predict(dist_matrix)

        else:
            return self.pretrained_model.predict(X)

    def _predict_kernel(self, X, is_knn=False):
        """
        Build a kernel for (X, X_ref_), then transform to distances if is_knn=True.
        """
        dev = qml.device("default.qubit", wires=self.num_qubits_)

        @qml.qnode(dev)
        def feature_map_circuit(x):
            self.quantum_feature_map(x, self.num_qubits_)
            return qml.state()

        def kernel_fn(x1, x2):
            state1 = feature_map_circuit(x1)
            state2 = feature_map_circuit(x2)
            return np.abs(np.dot(state1.conj(), state2)) ** 2

        K_new = qml.kernels.kernel_matrix(X, self.X_ref_, kernel_fn)

        if is_knn:
            dist_matrix = 1 - K_new
            dist_matrix = np.abs(dist_matrix)
            return self.pretrained_model.predict(dist_matrix)
        else:
            return self.pretrained_model.predict(K_new)

    def interpret(self, x):
        """
        Compute feature importance for a single instance x by:
          A) Drop-Column on entire reference set
          B) Permutation on entire reference set
        Then average.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        baseline_acc = accuracy_score(self.y_ref_, self._predict(self.X_ref_))

       
        drop_importances = []
        for i in range(self.num_features_):
            X_drop = self.X_ref_.copy()
            X_drop[:, i] = 0.0
            drop_acc = accuracy_score(self.y_ref_, self._predict(X_drop))
            drop_importances.append(baseline_acc - drop_acc)

       
        perm_importances = []
        for i in range(self.num_features_):
            scores = []
            for _ in range(self.n_repeats):
                X_perm = self.X_ref_.copy()
                np.random.shuffle(X_perm[:, i])  # shuffle one column
                perm_acc = accuracy_score(self.y_ref_, self._predict(X_perm))
                scores.append(perm_acc)
            mean_acc = np.mean(scores)
            perm_importances.append(baseline_acc - mean_acc)

        # Combine the two
        final_importances = [(d + p) / 2.0 for d, p in zip(drop_importances, perm_importances)]
        return final_importances
