import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LogisticRegression, Perceptron, RidgeClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



import pennylane as qml
from pennylane import numpy as pnp

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
        if hasattr(self.pretrained_model, 'num_qubits'):
            self.num_qubits_ = self.pretrained_model.num_qubits
        else:
            print(f"Warning: 'num_qubits' not found on model {self.model_type}. Defaulting to num_features: {self.num_features_}")
            self.num_qubits_ = self.num_features_
        return self

    def _predict(self, X_perturbed):
        model = self.pretrained_model
        model_type_local = self.model_type
        num_qubits = model.num_qubits
        qfm_to_use = model.quantum_feature_map
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def q_node_for_prediction(x_input):
            qfm_to_use(x_input, num_qubits)
            return qml.state()

        amplitude_models = ['qrf', 'qlogistic', 'qdt', 'qnb', 'qada', 'qgb', 'qlda', 'qperceptron', 'qridge', 'qextra']
        kernel_models_direct = ['qsvm', 'qsvc_poly'] # Defined, but won't be used in this run

        if model_type_local in amplitude_models:
            X_q = np.array([np.abs(q_node_for_prediction(x))**2 for x in X_perturbed])
            return model.predict(X_q)
        elif model_type_local in kernel_models_direct: # Logic remains for completeness of MEDLEY class
            def kernel_fn_pred(x1, x2):
                s1 = q_node_for_prediction(x1)
                s2 = q_node_for_prediction(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2
            test_kernel_matrix = qml.kernels.kernel_matrix(X_perturbed, self.X_ref_, kernel_fn_pred)
            return model.predict(test_kernel_matrix)
        elif model_type_local == 'qknn': # Logic remains for completeness of MEDLEY class
            def kernel_fn_pred_knn(x1, x2):
                s1 = q_node_for_prediction(x1)
                s2 = q_node_for_prediction(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2
            test_kernel_matrix = qml.kernels.kernel_matrix(X_perturbed, self.X_ref_, kernel_fn_pred_knn)
            dist_matrix = 1 - np.abs(test_kernel_matrix)
            return model.predict(dist_matrix)
        else:
            print(f"Warning: Model type '{model_type_local}' not recognized as HQML for MEDLEY._predict. Attempting direct prediction.")
            return model.predict(X_perturbed)

    def interpret(self, x_dummy=None):
        if self.X_ref_ is None or self.y_ref_ is None:
            raise ValueError("MEDLEY explainer must be fitted first with reference data.")
        baseline_acc = accuracy_score(self.y_ref_, self._predict(self.X_ref_))
        drop_importances = []
        for i in range(self.num_features_):
            X_drop = self.X_ref_.copy(); X_drop[:, i] = 0.0
            drop_acc = accuracy_score(self.y_ref_, self._predict(X_drop))
            drop_importances.append(baseline_acc - drop_acc)
        perm_importances = []
        for i in range(self.num_features_):
            scores = [];
            for _ in range(self.n_repeats):
                X_perm = self.X_ref_.copy(); np.random.shuffle(X_perm[:, i])
                perm_acc = accuracy_score(self.y_ref_, self._predict(X_perm))
                scores.append(perm_acc)
            mean_acc = np.mean(scores)
            perm_importances.append(baseline_acc - mean_acc)
        final_importances = [(d + p) / 2.0 for d, p in zip(drop_importances, perm_importances)]
        return np.array(final_importances)

# --- Quantum Feature Map ---
def sample_quantum_feature_map(x, num_qubits):
    for i in range(num_qubits):
        qml.RX(float(x[i]), wires=i)

# --- Simplified HQML Model Training Functions (Amplitude-Based Only) ---
def _train_amplitude_model(X_train, y_train, classical_model_class, model_params, hqml_model_type_str, feature_map_func=sample_quantum_feature_map):
    num_qubits = X_train.shape[1]
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(x):
        feature_map_func(x, num_qubits)
        return qml.state()
    X_train_q = np.array([np.abs(circuit(x))**2 for x in X_train])
    model = classical_model_class(**model_params)
    model.fit(X_train_q, y_train)
    model.num_qubits = num_qubits
    model.quantum_feature_map = feature_map_func
    model.hqml_model_type = hqml_model_type_str
    return model

def train_hqml_qrf(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, RandomForestClassifier, {'n_estimators': 50, 'random_state': 42, 'class_weight': 'balanced'}, 'qrf', feature_map_func)
def train_hqml_qlogistic(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, LogisticRegression, {'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'}, 'qlogistic', feature_map_func)
def train_hqml_qdt(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, DecisionTreeClassifier, {'random_state': 42, 'class_weight': 'balanced'}, 'qdt', feature_map_func)
def train_hqml_qnb(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, GaussianNB, {}, 'qnb', feature_map_func)
def train_hqml_qada(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, AdaBoostClassifier, {'n_estimators': 50, 'random_state': 42}, 'qada', feature_map_func)
def train_hqml_qgb(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, GradientBoostingClassifier, {'n_estimators': 50, 'random_state': 42}, 'qgb', feature_map_func)
def train_hqml_qlda(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, LinearDiscriminantAnalysis, {}, 'qlda', feature_map_func)
def train_hqml_qperceptron(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, Perceptron, {'random_state': 42, 'class_weight': 'balanced'}, 'qperceptron', feature_map_func)
def train_hqml_qridge(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, RidgeClassifier, {'random_state': 42, 'class_weight': 'balanced'}, 'qridge', feature_map_func)
def train_hqml_qextra(X_train, y_train, feature_map_func=sample_quantum_feature_map):
    return _train_amplitude_model(X_train, y_train, ExtraTreesClassifier, {'n_estimators': 50, 'random_state': 42, 'class_weight': 'balanced'}, 'qextra', feature_map_func)

# Kernel-Based Model training functions are now removed

# --- Data Loading and Preprocessing ---
def _add_noise_and_redundancy(X_orig, feature_names_orig, random_state=42):
    n_samples = X_orig.shape[0]
    rng = np.random.RandomState(random_state)
    if X_orig.shape[1] == 0:
        noisy_feat1 = rng.randn(n_samples, 1); noisy_feat2 = rng.uniform(-1,1,size=(n_samples,1)); redundant_feat = rng.randn(n_samples,1)
        X_processed = np.hstack((noisy_feat1, noisy_feat2, redundant_feat))
        feature_names_processed = ['noisy_g', 'noisy_u', 'redundant_g']
    else:
        std_mean = np.std(X_orig, axis=0).mean() if X_orig.shape[0] > 1 else 1.0
        noisy_feat1 = rng.randn(n_samples, 1) * std_mean * 0.5
        noisy_feat2 = rng.uniform(-1, 1, size=(n_samples, 1)) * std_mean * 0.5
        primary_feat_idx = np.argmax(np.var(X_orig, axis=0)) if X_orig.shape[0] > 1 else 0
        primary_feat_for_redundancy = X_orig[:, primary_feat_idx].reshape(-1, 1)
        std_primary = np.std(primary_feat_for_redundancy) if X_orig.shape[0] > 1 else 1.0
        redundant_feat = primary_feat_for_redundancy * 0.7 + rng.normal(0, 0.05 * std_primary, size=(n_samples, 1))
        original_feat_name_short = feature_names_orig[primary_feat_idx]
        if len(original_feat_name_short) > 10:
            original_feat_name_short = original_feat_name_short[:7] + "..."
        redundant_name = f'redundant_{original_feat_name_short}'
        X_processed = np.hstack((X_orig, noisy_feat1, noisy_feat2, redundant_feat))
        feature_names_processed = feature_names_orig + ['noisy_g', 'noisy_u', redundant_name]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    return X_scaled, feature_names_processed

def load_and_prep_iris_noisy_for_hqml():
    data = load_iris()
    X_scaled, feature_names = _add_noise_and_redundancy(data.data, list(data.feature_names))
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Q-MEDLEY application on Amplitude HQML models (Noisy Iris)...")
    X_train, X_test, y_train, y_test, feature_names = load_and_prep_iris_noisy_for_hqml()

    hqml_model_trainers = {
        "Q.RandomForest": train_hqml_qrf,
        "Q.LogRegression": train_hqml_qlogistic,
        "Q.DecisionTree": train_hqml_qdt,
        "Q.NaiveBayes": train_hqml_qnb,
        "Q.AdaBoost": train_hqml_qada,
        "Q.GradBoost": train_hqml_qgb,
        "Q.LDA": train_hqml_qlda,
        "Q.Perceptron": train_hqml_qperceptron,
        "Q.Ridge": train_hqml_qridge,
        "Q.ExtraTrees": train_hqml_qextra
        # Kernel-based models (QSVM, QSVC_Poly, QKNN) are removed
    }

    all_medley_scores = {}
    N_REPEATS_MEDLEY = 3
    FIGURE_COLS = 2 # Adjusted for 10 models (e.g., 2 cols x 5 rows or 3 cols x 4 rows)

    print(f"\nApplying Q-MEDLEY (n_repeats={N_REPEATS_MEDLEY})...")
    for model_display_name, train_func in hqml_model_trainers.items():
        print(f"\n--- Processing HQML Model: {model_display_name} ---")
        try:
            print("  Training HQML model...")
            trained_hqml_model = train_func(X_train, y_train, feature_map_func=sample_quantum_feature_map)

            print("  Applying Q-MEDLEY explainer...")
            q_medley_explainer = MEDLEY(
                pretrained_model=trained_hqml_model,
                model_type=trained_hqml_model.hqml_model_type,
                quantum_feature_map=trained_hqml_model.quantum_feature_map,
                n_repeats=N_REPEATS_MEDLEY
            )
            q_medley_explainer.fit(X_train, y_train)
            importance_scores = q_medley_explainer.interpret()
            all_medley_scores[model_display_name] = importance_scores

            print(f"    Q-MEDLEY Importances: {np.round(importance_scores, 4)}")
            top_indices = np.argsort(importance_scores)[::-1][:3]
            print(f"    Top 3 features by Q-MEDLEY: {[feature_names[i] for i in top_indices if i < len(feature_names)]}")

        except Exception as e:
            print(f"    ERROR processing {model_display_name}: {e}")
            all_medley_scores[model_display_name] = np.zeros(X_train.shape[1])

    if all_medley_scores:
        num_models = len(all_medley_scores)
        if num_models > 0:
            cols = FIGURE_COLS
            rows = (num_models + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
            axes_flat = axes.flatten()
            sorted_model_names = sorted(all_medley_scores.keys())

            for i, model_name in enumerate(sorted_model_names):
                scores = all_medley_scores[model_name]
                ax = axes_flat[i]
                y_pos = np.arange(len(feature_names))
                ax.barh(y_pos, scores, align='center', color='skyblue', edgecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names, fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel('Q-MEDLEY Score', fontsize=9)
                ax.set_title(model_name, fontsize=10)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)

            for i in range(num_models, len(axes_flat)):
                fig.delaxes(axes_flat[i])

            fig.suptitle("Q-MEDLEY for Amplitude-Encoded HQML Models on Noisy Iris", fontsize=14, y=1.02 if rows >1 else 1.05)
            plt.tight_layout(rect=[0, 0, 1, 0.98 if rows > 1 else 0.95])
            plt.savefig("qmedley_amplitude_hqml_iris_noisy.png", dpi=150)
            print("\nQ-MEDLEY comparison plot saved as qmedley_amplitude_hqml_iris_noisy.png")
            plt.show()
    else:
        print("No Q-MEDLEY scores were generated to plot.")

    print("\nExperiment Finished.")
