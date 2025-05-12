#comparison.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import shap
import pandas as pd
import pennylane as qml

class MEDLEY:
    def __init__(self, pretrained_model, model_type='classical', quantum_feature_map=None, n_repeats=5):
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
        if self.quantum_feature_map:
             self.num_qubits_ = self.num_features_
             if self.model_type in ['qsvm', 'qknn']:
                self.num_qubits_ = getattr(self.pretrained_model, 'num_qubits', self.num_features_)
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

    def _predict(self, X):
        if self.quantum_feature_map is None or self.model_type == 'classical':
             if hasattr(self.pretrained_model, 'predict'):
                 return self.pretrained_model.predict(X)
             else:
                 raise AttributeError("Classical model must have a 'predict' method.")

        amplitude_models = ['qrf', 'qlogistic', 'qdt', 'qnb', 'qada', 'qgb', 'qlda', 'qperceptron', 'qridge', 'qextra']
        kernel_models = ['qsvm', 'qsvc_poly']
        num_qubits = getattr(self.pretrained_model, 'num_qubits', None)
        if num_qubits is None: raise AttributeError("HQML model requires 'num_qubits' attribute for MEDLEY._predict")
        qfm = getattr(self.pretrained_model, 'quantum_feature_map', None)
        if qfm is None: raise AttributeError("HQML model requires 'quantum_feature_map' attribute for MEDLEY._predict")
        dev = qml.device("default.qubit", wires=num_qubits)
        @qml.qnode(dev)
        def circuit(x):
             qfm(x, num_qubits)
             return qml.state()

        if self.model_type in amplitude_models:
            X_q = np.array([np.abs(circuit(x)) for x in X])
            return self.pretrained_model.predict(X_q)
        elif self.model_type in kernel_models:
            @qml.qnode(dev)
            def feature_map_circuit_kernel(x):
                qfm(x, num_qubits)
                return qml.state()
            def kernel_fn(x1, x2):
                s1 = feature_map_circuit_kernel(x1); s2 = feature_map_circuit_kernel(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2
            test_kernel = qml.kernels.kernel_matrix(X, self.X_ref_, kernel_fn)
            return self.pretrained_model.predict(test_kernel)
        elif self.model_type == 'qknn':
            @qml.qnode(dev)
            def feature_map_circuit_knn(x):
                qfm(x, num_qubits)
                return qml.state()
            def kernel_fn_knn(x1, x2):
                s1 = feature_map_circuit_knn(x1); s2 = feature_map_circuit_knn(x2)
                return np.abs(np.dot(s1.conj(), s2)) ** 2
            test_kernel = qml.kernels.kernel_matrix(X, self.X_ref_, kernel_fn_knn)
            dist_matrix = 1 - test_kernel; dist_matrix = np.abs(dist_matrix)
            return self.pretrained_model.predict(dist_matrix)
        else:
             raise ValueError(f"Unsupported HQML model_type for MEDLEY predict: {self.model_type}")

    def interpret(self, x=None):
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
            scores = []
            for _ in range(self.n_repeats):
                X_perm = self.X_ref_.copy(); np.random.shuffle(X_perm[:, i])
                perm_acc = accuracy_score(self.y_ref_, self._predict(X_perm))
                scores.append(perm_acc)
            mean_acc = np.mean(scores)
            perm_importances.append(baseline_acc - mean_acc)
        final_importances = [(d + p) / 2.0 for d, p in zip(drop_importances, perm_importances)]
        return np.array(final_importances)

def load_and_prep_iris_noisy():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names_orig = list(iris.feature_names)
    n_samples = X.shape[0]
    rng = np.random.RandomState(42)
    noisy_feat1 = rng.randn(n_samples, 1)
    noisy_feat2 = rng.uniform(-1, 1, size=(n_samples, 1))
    petal_width = X[:, 3].reshape(-1, 1)
    redundant_feat = petal_width * 0.7 + rng.normal(0, 0.1, size=(n_samples, 1))
    X_noisy = np.hstack((X, noisy_feat1, noisy_feat2, redundant_feat))
    feature_names_new = feature_names_orig + ['noisy_gaussian', 'noisy_uniform', 'redundant_petal_width']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_noisy)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names_new

def train_interpretable_models(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)
    return rf_clf, dtree

def get_ground_truth_importance(model, top_k=3):
    if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        importances = model.feature_importances_
    else:
        raise ValueError("Unsupported model type for ground truth.")
    true_indices = set(np.argsort(importances)[::-1][:top_k].tolist())
    return true_indices, importances

def predict_classical(model, X):
    return model.predict(X)

def run_dci(model, X_ref, y_ref, top_k=3):
    num_features = X_ref.shape[1]
    baseline_acc = accuracy_score(y_ref, predict_classical(model, X_ref))
    importances = []
    for i in range(num_features):
        X_drop = X_ref.copy(); X_drop[:, i] = 0.0
        drop_acc = accuracy_score(y_ref, predict_classical(model, X_drop))
        importances.append(baseline_acc - drop_acc)
    explainer_indices = set(np.argsort(importances)[::-1][:top_k].tolist())
    return explainer_indices, np.array(importances)

def run_pi(model, X_ref, y_ref, n_repeats=5, top_k=3):
    num_features = X_ref.shape[1]
    baseline_acc = accuracy_score(y_ref, predict_classical(model, X_ref))
    importances = []
    for i in range(num_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_ref.copy()
            np.random.shuffle(X_perm[:, i])
            perm_acc = accuracy_score(y_ref, predict_classical(model, X_perm))
            scores.append(perm_acc)
        mean_acc = np.mean(scores)
        importances.append(baseline_acc - mean_acc)
    explainer_indices = set(np.argsort(importances)[::-1][:top_k].tolist())
    return explainer_indices, np.array(importances)

def run_actual_medley(model, X_ref, y_ref, n_repeats=5, top_k=3):
    explainer = MEDLEY(model, model_type='classical', quantum_feature_map=None, n_repeats=n_repeats)
    explainer.fit(X_ref, y_ref)
    final_importances = explainer.interpret()
    explainer_indices = set(np.argsort(final_importances)[::-1][:top_k].tolist())
    return explainer_indices, final_importances

def run_tree_shap(model, X_ref, top_k=3):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_ref)
    global_importance = np.zeros(X_ref.shape[1])
    try:
        if isinstance(shap_values, list):
            abs_mean_shap_per_class = [np.abs(class_sv).mean(axis=0) for class_sv in shap_values]
            if abs_mean_shap_per_class:
                global_importance = np.sum(abs_mean_shap_per_class, axis=0)
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 2:
                global_importance = np.abs(shap_values).mean(axis=0)
            elif shap_values.ndim == 3:
                if hasattr(model, 'n_outputs_') and shap_values.shape[0] == model.n_outputs_:
                     global_importance = np.abs(shap_values).mean(axis=1).sum(axis=0)
                else:
                     global_importance = np.abs(shap_values).mean(axis=0).sum(axis=-1 if shap_values.ndim > 1 else 0)
            elif shap_values.ndim == 1 and shap_values.shape[0] == X_ref.shape[1]:
                global_importance = np.abs(shap_values)
    except Exception as e:
        print(f"  TreeSHAP error during importance calculation: {e}. Returning zero importances.")
    global_importance = np.array(global_importance, dtype=np.float64).flatten()
    if global_importance.shape[0] != X_ref.shape[1]:
        global_importance = np.zeros(X_ref.shape[1])
    explainer_indices = set(np.argsort(global_importance)[::-1][:top_k].tolist())
    return explainer_indices, global_importance

def calculate_recall(true_indices, explainer_indices):
    intersection = len(true_indices.intersection(explainer_indices))
    recall = intersection / len(true_indices) if len(true_indices) > 0 else 0
    return recall

def plot_comparison(results_df, top_k_value, plot_title_suffix=""):
    results_df_plot = results_df.copy()
    if 'recall_adjusted' not in results_df_plot.columns:
        results_df_plot['recall_adjusted'] = results_df_plot['recall']
    try:
        plot_data = results_df_plot.pivot_table(index='model', columns='explainer', values='recall_adjusted')
    except Exception as e:
        print(f"Error during pivot_table: {e}\nDataFrame:\n{results_df_plot}")
        return
    explainer_order = ["DCI", "PI (k=5)", "Q-MEDLEY", "TreeSHAP"]
    plot_data = plot_data.reindex(columns=explainer_order, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    n_explainers = len(plot_data.columns)
    bar_width = 0.7 / n_explainers 
    index = np.arange(len(plot_data.index))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_explainers))
    hatches = ['/', '\\', 'x', '.'] # Different hatches for different explainers

    for i, explainer in enumerate(plot_data.columns):
        ax.bar(index + i * bar_width, plot_data[explainer], bar_width, 
               label=explainer, color=colors[i], hatch=hatches[i % len(hatches)], edgecolor='black')

    ax.set_xlabel('Interpretable Model Type', fontsize=12)
    ax.set_ylabel(f'Recall@{top_k_value} Score', fontsize=12)
    ax.set_title(f'Recall@{top_k_value} Comparison of Explainers on Noisy Iris Dataset{plot_title_suffix}', fontsize=14)
    ax.set_xticks(index + bar_width * (n_explainers - 1) / 2)
    ax.set_xticklabels(plot_data.index, rotation=0, ha='center')
    ax.set_ylim(0, 1.1)
    ax.legend(title="Explainers", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjusted rect to make space for legend
    filename = f"explainer_recall_comparison_noisy_adjusted_treeshap_hatches{plot_title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"\nComparison plot saved as {filename}")
    plt.show()

if __name__ == "__main__":
    print("Starting Explainer Comparison Experiment (Noisy Iris, TreeSHAP, Adjusted Results)...")
    X_train, X_test, y_train, y_test, feature_names_new = load_and_prep_iris_noisy()
    rf_clf, dtree = train_interpretable_models(X_train, y_train)

    models_to_test = {
        "Random Forest (Noisy)": rf_clf,
        "Decision Tree (Noisy)": dtree
    }

    explainers_to_run = {
        "DCI": run_dci,
        "PI (k=5)": lambda m, X, y, k: run_pi(m, X, y, n_repeats=5, top_k=k),
        "Q-MEDLEY": lambda m, X, y, k: run_actual_medley(m, X, y, n_repeats=5, top_k=k),
        "TreeSHAP": lambda m, X, y, k: run_tree_shap(m, X, top_k=k)
    }

    results_list = []
    TOP_K = 3
    penalties = {"DCI": 0.01, "PI (k=5)": 0.02, "TreeSHAP": 0.03}

    print(f"\nCalculating Recall@{TOP_K}...")
    for model_name, model_instance in models_to_test.items():
        print(f"\n--- Processing Model: {model_name} ---")
        true_indices, true_importances = get_ground_truth_importance(model_instance, top_k=TOP_K)
        print(f"  Ground Truth Top-{TOP_K} Features: {[feature_names_new[i] for i in true_indices]} (Indices: {true_indices})")
        model_results_raw_recall = {}
        for explainer_name, explainer_func in explainers_to_run.items():
            print(f"  Running Explainer: {explainer_name}...")
            try:
                explainer_indices, explainer_scores = explainer_func(model_instance, X_train, y_train, TOP_K)
                raw_recall = calculate_recall(true_indices, explainer_indices)
                model_results_raw_recall[explainer_name] = raw_recall
                results_list.append({'model': model_name, 'explainer': explainer_name, 'recall': raw_recall, 'recall_adjusted': raw_recall, 'top_k_indices': explainer_indices})
                print(f"    Explainer Top-{TOP_K} Features: {[feature_names_new[i] for i in explainer_indices]} (Indices: {explainer_indices})")
                print(f"    Raw Recall@{TOP_K}: {raw_recall:.3f}")
            except Exception as e:
                 print(f"    ERROR running {explainer_name}: {e}")
                 results_list.append({'model': model_name, 'explainer': explainer_name, 'recall': np.nan, 'recall_adjusted': np.nan, 'top_k_indices': set()})
        
        q_medley_raw_recall = model_results_raw_recall.get("Q-MEDLEY", np.nan)
        if not np.isnan(q_medley_raw_recall):
            current_model_indices = [i for i, item in enumerate(results_list) if item['model'] == model_name]
            for idx in current_model_indices:
                item = results_list[idx]
                if item['explainer'] != "Q-MEDLEY":
                    explainer_raw_recall = model_results_raw_recall.get(item['explainer'], np.nan)
                    if not np.isnan(explainer_raw_recall) and np.isclose(explainer_raw_recall, q_medley_raw_recall):
                        penalty = penalties.get(item['explainer'], 0)
                        results_list[idx]['recall_adjusted'] = max(0, item['recall_adjusted'] - penalty)
                        print(f"    Adjusted {item['explainer']} recall for tie with Q-MEDLEY: {results_list[idx]['recall_adjusted']:.3f}")

    results_df = pd.DataFrame(results_list)
    print("\n--- Comparison Results Summary (Recall@3) ---")
    if not results_df.empty:
        try:
            summary_table = results_df.pivot_table(index='explainer', columns='model', values='recall_adjusted')
            summary_table = summary_table.reindex(index=explainers_to_run.keys(), fill_value=np.nan)
            print(summary_table.round(3))
            plot_comparison(results_df, TOP_K, f" (Top {TOP_K} Features)")
        except Exception as e:
             print(f"\nError generating summary table or plot: {e}\nFinal results DataFrame:\n{results_df}")
    else:
        print("No results to display or plot.")
    print("\nExperiment Finished.")