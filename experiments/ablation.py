# src/explainers/q_medley_ablation_5_tabular_datasets_v2.py

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Only for potential debugging
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    fetch_covtype, load_diabetes # Replaced KDD with Diabetes
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer # For binning diabetes target
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pennylane as qml

# --- MEDLEYAblation Class Definition (as provided previously) ---
class MEDLEYAblation:
    def __init__(self, pretrained_model, model_type='classical', quantum_feature_map=None, n_repeats=5):
        self.pretrained_model = pretrained_model
        self.model_type = model_type
        self.quantum_feature_map = quantum_feature_map
        self.n_repeats = n_repeats
        self.X_ref_ = None
        self.y_ref_ = None
        self.num_features_ = None
        self.epsilon = 1e-6

    def fit(self, X, y):
        self.X_ref_ = X
        self.y_ref_ = y
        self.num_features_ = X.shape[1]
        if self.num_features_ > 1:
            self.correlations_ = np.abs(np.corrcoef(self.X_ref_.T))
            np.fill_diagonal(self.correlations_, 0)
        else:
            self.correlations_ = np.zeros((1,1))
        return self

    def _predict_classical(self, X):
        if hasattr(self.pretrained_model, 'predict'):
            return self.pretrained_model.predict(X)
        else:
            raise AttributeError("Model must have a 'predict' method.")

    def _calculate_dci_scores(self, baseline_acc):
        drop_importances = []
        for i in range(self.num_features_):
            X_drop = self.X_ref_.copy()
            X_drop[:, i] = 0.0
            drop_acc = accuracy_score(self.y_ref_, self._predict_classical(X_drop))
            drop_importances.append(baseline_acc - drop_acc)
        return np.array(drop_importances)

    def _calculate_pi_scores_standard(self, baseline_acc):
        perm_importances = []
        for i in range(self.num_features_):
            scores = []
            for _ in range(self.n_repeats):
                X_perm = self.X_ref_.copy()
                np.random.shuffle(X_perm[:, i])
                perm_acc = accuracy_score(self.y_ref_, self._predict_classical(X_perm))
                scores.append(perm_acc)
            mean_acc = np.mean(scores)
            perm_importances.append(baseline_acc - mean_acc)
        return np.array(perm_importances)

    def _calculate_pi_scores_interaction_aware(self, baseline_acc):
        perm_importances_interaction = []
        for i in range(self.num_features_):
            scores = []
            if self.num_features_ > 1:
                correlated_feature_idx = np.argmax(self.correlations_[i, :])
                if correlated_feature_idx == i:
                    other_indices = [idx for idx in range(self.num_features_) if idx !=i]
                    correlated_feature_idx = np.random.choice(other_indices) if other_indices else -1
            else:
                correlated_feature_idx = -1
            for _ in range(self.n_repeats):
                X_perm = self.X_ref_.copy()
                np.random.shuffle(X_perm[:, i])
                if correlated_feature_idx != -1 and correlated_feature_idx != i :
                    noise = np.random.normal(0, 0.01 * np.std(X_perm[:, correlated_feature_idx]), X_perm.shape[0])
                    X_perm[:, correlated_feature_idx] = X_perm[:, correlated_feature_idx] + noise
                perm_acc = accuracy_score(self.y_ref_, self._predict_classical(X_perm))
                scores.append(perm_acc)
            mean_acc = np.mean(scores)
            perm_importances_interaction.append(baseline_acc - mean_acc)
        return np.array(perm_importances_interaction)

    def interpret(self, ablation_config="Base (DCI+PI Avg)"):
        if self.X_ref_ is None or self.y_ref_ is None:
            raise ValueError("MEDLEYAblation explainer must be fitted first.")
        baseline_acc = accuracy_score(self.y_ref_, self._predict_classical(self.X_ref_))
        dci_scores = self._calculate_dci_scores(baseline_acc)
        if "Interaction-Aware PI" in ablation_config or "InteractionPI" in ablation_config:
            pi_scores = self._calculate_pi_scores_interaction_aware(baseline_acc)
        else:
            pi_scores = self._calculate_pi_scores_standard(baseline_acc)

        if ablation_config == "Base (DCI only)":
            final_importances = dci_scores
        elif ablation_config == "Base (PI only)":
            final_importances = pi_scores
        elif ablation_config == "Q-MEDLEY (DCI+PI Avg)":
            final_importances = (dci_scores + pi_scores) / 2.0
        elif ablation_config == "Q-MEDLEY + AdaptiveWeighting":
            weighted_importances = np.zeros_like(dci_scores)
            for j in range(self.num_features_):
                dci_val = abs(dci_scores[j]); pi_val = abs(pi_scores[j])
                if dci_val > pi_val * 1.2: w_dci, w_pi = 0.7, 0.3
                elif pi_val > dci_val * 1.2: w_dci, w_pi = 0.3, 0.7
                else: w_dci, w_pi = 0.5, 0.5
                weighted_importances[j] = w_dci * dci_scores[j] + w_pi * pi_scores[j]
            final_importances = weighted_importances
        elif ablation_config == "Q-MEDLEY with InteractionPI":
            final_importances = (dci_scores + pi_scores) / 2.0
        elif ablation_config == "Q-MEDLEY + AdaptiveWeighting + InteractionPI":
            weighted_importances = np.zeros_like(dci_scores)
            for j in range(self.num_features_):
                dci_val = abs(dci_scores[j]); pi_val = abs(pi_scores[j])
                if dci_val > pi_val * 1.2: w_dci, w_pi = 0.7, 0.3
                elif pi_val > dci_val * 1.2: w_dci, w_pi = 0.3, 0.7
                else: w_dci, w_pi = 0.5, 0.5
                weighted_importances[j] = w_dci * dci_scores[j] + w_pi * pi_scores[j]
            final_importances = weighted_importances
        else:
            raise ValueError(f"Unknown ablation configuration: {ablation_config}")
        return final_importances

# --- Data Loading and Preprocessing Functions ---
def _add_noise_and_redundancy(X_orig, feature_names_orig, random_state=42):
    n_samples = X_orig.shape[0]
    rng = np.random.RandomState(random_state)
    
    if X_orig.shape[1] == 0:
        noisy_feat1 = rng.randn(n_samples, 1)
        noisy_feat2 = rng.uniform(-1, 1, size=(n_samples, 1))
        redundant_feat = rng.randn(n_samples, 1)
        X_processed = np.hstack((noisy_feat1, noisy_feat2, redundant_feat))
        feature_names_processed = ['noisy_gaussian', 'noisy_uniform', 'redundant_generic_0']
    else:
        std_mean = np.std(X_orig, axis=0).mean() if X_orig.shape[0] > 1 else 1.0
        noisy_feat1 = rng.randn(n_samples, 1) * std_mean * 0.5
        noisy_feat2 = rng.uniform(-1, 1, size=(n_samples, 1)) * std_mean * 0.5
        
        primary_feat_idx = np.argmax(np.var(X_orig, axis=0)) if X_orig.shape[0] > 1 else 0
        primary_feat_for_redundancy = X_orig[:, primary_feat_idx].reshape(-1, 1)
        std_primary = np.std(primary_feat_for_redundancy) if X_orig.shape[0] > 1 else 1.0
        redundant_feat = primary_feat_for_redundancy * 0.7 + rng.normal(0, 0.05 * std_primary, size=(n_samples, 1))
        redundant_name = f'redundant_{feature_names_orig[primary_feat_idx]}'
        
        X_processed = np.hstack((X_orig, noisy_feat1, noisy_feat2, redundant_feat))
        feature_names_processed = feature_names_orig + ['noisy_gaussian', 'noisy_uniform', redundant_name]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    return X_scaled, feature_names_processed

def load_and_prep_iris_noisy():
    data = load_iris()
    X_scaled, feature_names = _add_noise_and_redundancy(data.data, list(data.feature_names))
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names, "Iris"

def load_and_prep_wine_noisy():
    data = load_wine()
    X_scaled, feature_names = _add_noise_and_redundancy(data.data, list(data.feature_names))
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names, "Wine"

def load_and_prep_breast_cancer_noisy():
    data = load_breast_cancer()
    X_scaled, feature_names = _add_noise_and_redundancy(data.data, list(data.feature_names))
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names, "BreastCancer"

def load_and_prep_covtype_noisy(n_samples_subset=2000, n_features_subset=20):
    print(f"Fetching Covertype dataset (subset)...")
    covtype = fetch_covtype(shuffle=True, random_state=42)
    X_orig = covtype.data
    y_orig = covtype.target - 1
    if n_samples_subset is not None and n_samples_subset < X_orig.shape[0]:
        X_orig, _, y_orig, _ = train_test_split(X_orig, y_orig, train_size=n_samples_subset, random_state=42, stratify=y_orig)
    if n_features_subset is not None and n_features_subset < X_orig.shape[1]:
        X_orig = X_orig[:, :n_features_subset]
    feature_names_orig = [f"cov_feat_{i}" for i in range(X_orig.shape[1])]
    X_scaled, feature_names = _add_noise_and_redundancy(X_orig, feature_names_orig)
    y = y_orig
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names, "Covtype"

def load_and_prep_diabetes_noisy(n_bins=3):
    data = load_diabetes()
    X_orig = data.data
    y_orig = data.target # This is a continuous target
    feature_names_orig = list(data.feature_names)

    # Bin the continuous target into discrete classes
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None, random_state=42)
    y_binned = discretizer.fit_transform(y_orig.reshape(-1, 1)).ravel().astype(int)
    
    X_scaled, feature_names = _add_noise_and_redundancy(X_orig, feature_names_orig)
    y = y_binned
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names, "Diabetes"


# --- Model Training, Ground Truth, and Recall Calculation ---
def train_interpretable_models(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_clf.fit(X_train, y_train)
    dtree = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    dtree.fit(X_train, y_train)
    return rf_clf, dtree

def get_ground_truth_importance(model, top_k=3):
    if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        importances = model.feature_importances_
    else:
        raise ValueError("Unsupported model type for ground truth.")
    true_indices = set(np.argsort(importances)[::-1][:top_k].tolist())
    return true_indices

def calculate_recall(true_indices, explainer_scores, top_k=3):
    if explainer_scores is None or len(explainer_scores) == 0 or np.all(np.isnan(explainer_scores)): return 0.0
    if np.all(explainer_scores == 0) or np.all(np.isnan(explainer_scores)):
         return 0.0
    sorted_indices = np.argsort(explainer_scores)[::-1]
    explainer_indices = set(sorted_indices[:top_k].tolist())
    intersection = len(true_indices.intersection(explainer_indices))
    recall = intersection / len(true_indices) if len(true_indices) > 0 else 0
    return recall

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Q-MEDLEY Functional Component Ablation Study (5 Tabular Datasets)...")

    datasets_to_process = {
        "Iris": load_and_prep_iris_noisy,
        "Wine": load_and_prep_wine_noisy,
        "BreastCancer": load_and_prep_breast_cancer_noisy,
        "Covtype_subset": load_and_prep_covtype_noisy,
        "Diabetes_binned": load_and_prep_diabetes_noisy # New dataset
    }

    ablation_configs_for_qmedley = [
        "Base (DCI only)",
        "Base (PI only)",
        "Q-MEDLEY (DCI+PI Avg)",
        "Q-MEDLEY + AdaptiveWeighting",
        "Q-MEDLEY with InteractionPI",
        "Q-MEDLEY + AdaptiveWeighting + InteractionPI"
    ]

    all_results_list = []
    TOP_K = 3
    N_REPEATS_PI = 3

    print(f"\nCalculating Recall@{TOP_K} for Q-MEDLEY Component Ablations...")

    for dataset_key_name, load_func in datasets_to_process.items(): # Use a different key for the dict
        print(f"\n\n===== Processing Dataset: {dataset_key_name} =====")
        X_train, X_test, y_train, y_test, feature_names, dataset_display_name = load_func()
        
        rf_clf, dtree = train_interpretable_models(X_train, y_train)
        models_to_test_on_current_dataset = {
            "Random Forest": rf_clf,
            "Decision Tree": dtree
        }

        for model_name_short, model_instance in models_to_test_on_current_dataset.items():
            print(f"\n--- Model: {model_name_short} on {dataset_display_name} ---")
            
            true_indices_gt = get_ground_truth_importance(model_instance, top_k=TOP_K)
            print(f"  Ground Truth Top-{TOP_K} Feature Indices: {true_indices_gt}")
            gt_feature_names_display = [feature_names[i] for i in true_indices_gt if i < len(feature_names)]
            print(f"  Corresponding Feature Names: {gt_feature_names_display}")

            explainer_instance = MEDLEYAblation(model_instance, model_type='classical', n_repeats=N_REPEATS_PI)
            explainer_instance.fit(X_train, y_train)

            for config_name in ablation_configs_for_qmedley:
                print(f"  Running Q-MEDLEY Config: {config_name}...")
                try:
                    importance_scores = explainer_instance.interpret(ablation_config=config_name)
                    recall = calculate_recall(true_indices_gt, importance_scores, top_k=TOP_K)
                    
                    all_results_list.append({
                        'Dataset': dataset_display_name,
                        'Model': model_name_short,
                        'Q-MEDLEY Configuration': config_name,
                        f'Recall@{TOP_K}': recall
                    })
                    top_exp_indices = set(np.argsort(importance_scores)[::-1][:TOP_K].tolist())
                    exp_feature_names_display = [feature_names[i] for i in top_exp_indices if i < len(feature_names)]
                    print(f"    Explainer Top-{TOP_K} Indices: {top_exp_indices} ({exp_feature_names_display})")
                    print(f"    Recall@{TOP_K}: {recall:.3f}")
                except Exception as e:
                     print(f"    ERROR running {config_name} for {model_name_short} on {dataset_display_name}: {e}")
                     all_results_list.append({
                        'Dataset': dataset_display_name,
                        'Model': model_name_short,
                        'Q-MEDLEY Configuration': config_name,
                        f'Recall@{TOP_K}': np.nan
                    })

    results_df = pd.DataFrame(all_results_list)

    print("\n\n--- Q-MEDLEY Functional Component Ablation Study Results Table (All Datasets) ---")
    if not results_df.empty:
        try:
            summary_table = results_df.pivot_table(
                index=['Dataset', 'Model'], 
                columns='Q-MEDLEY Configuration', 
                values=f'Recall@{TOP_K}'
            )
            summary_table = summary_table.reindex(columns=ablation_configs_for_qmedley)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            print(summary_table.round(3))
        except Exception as e:
            print(f"Error generating summary table: {e}")
            print("Raw results list (first 10 entries):")
            for item in all_results_list[:10]:
                print(item)
    else:
        print("No results generated for the ablation study.")

    print("\nAblation Study Finished.")