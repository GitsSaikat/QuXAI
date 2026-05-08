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
        self.num_qubits_ = getattr(self.pretrained_model, 'num_qubits', self.num_features_)

        if self.model_type in ('qsvm', 'qknn') and self.quantum_feature_map:
            dev = qml.device("default.qubit", wires=self.num_qubits_)

            @qml.qnode(dev)
            def fmc(x):
                self.quantum_feature_map(x, self.num_qubits_)
                return qml.state()

            def kernel_fn(x1, x2):
                s1 = fmc(x1)
                s2 = fmc(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2

            self.kernel_ref_ = qml.kernels.kernel_matrix(X, X, kernel_fn)

        return self

    def _predict(self, X):
        amplitude_models = {
            'qrf', 'qlogistic', 'qdt', 'qnb', 'qada',
            'qgb', 'qlda', 'qperceptron', 'qridge', 'qextra',
        }
        kernel_models = {'qsvm', 'qsvc_poly'}

        if self.model_type == 'classical':
            return self.pretrained_model.predict(X)

        num_qubits = self.pretrained_model.num_qubits
        qfm = self.pretrained_model.quantum_feature_map
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def circuit(x):
            qfm(x, num_qubits)
            return qml.state()

        if self.model_type in amplitude_models:
            X_sub = X[:, :num_qubits]
            # Use squared amplitudes (probabilities) — consistent with training
            X_q = np.array([np.abs(circuit(x)) ** 2 for x in X_sub])
            return self.pretrained_model.predict(X_q)

        if self.model_type in kernel_models:
            X_sub = X[:, :num_qubits]

            def kernel_fn(x1, x2):
                s1 = circuit(x1)
                s2 = circuit(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2

            test_kernel = qml.kernels.kernel_matrix(X_sub, self.X_ref_[:, :num_qubits], kernel_fn)
            return self.pretrained_model.predict(test_kernel)

        if self.model_type == 'qknn':
            X_sub = X[:, :num_qubits]

            def kernel_fn(x1, x2):
                s1 = circuit(x1)
                s2 = circuit(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2

            test_kernel = qml.kernels.kernel_matrix(X_sub, self.X_ref_[:, :num_qubits], kernel_fn)
            dist_matrix = np.abs(1 - test_kernel)
            return self.pretrained_model.predict(dist_matrix)

        return self.pretrained_model.predict(X)

    def interpret(self, x=None):
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
                np.random.shuffle(X_perm[:, i])
                perm_acc = accuracy_score(self.y_ref_, self._predict(X_perm))
                scores.append(perm_acc)
            perm_importances.append(baseline_acc - np.mean(scores))

        return np.array([(d + p) / 2.0 for d, p in zip(drop_importances, perm_importances)])
