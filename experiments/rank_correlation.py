import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr 
import pandas as pd

class MEDLEY:
    def __init__(self, pretrained_model, model_type='classical', quantum_feature_map=None, n_repeats=5):
        self.pretrained_model = pretrained_model
        self.model_type = model_type
        self.quantum_feature_map = quantum_feature_map 
        self.n_repeats = n_repeats
        self.X_ref_ = None
        self.y_ref_ = None
        self.num_features_ = None

    def fit(self, X, y):
        self.X_ref_ = X
        self.y_ref_ = y
        self.num_features_ = X.shape[1]
        return self

    def _predict(self, X):
        if self.model_type == 'classical':
             if hasattr(self.pretrained_model, 'predict'):
                 return self.pretrained_model.predict(X)
             else:
                 raise AttributeError("Classical model must have a 'predict' method.")
        else:
            if hasattr(self.pretrained_model, 'predict'):
                 return self.pretrained_model.predict(X) 
            else:
                 raise ValueError(f"Unsupported model_type '{self.model_type}' or model lacks 'predict' for _predict.")

    def interpret(self, x=None): 
        if self.X_ref_ is None or self.y_ref_ is None:
            raise ValueError("MEDLEY explainer must be fitted first with reference data.")
        if self.X_ref_.shape[0] == 0:
            return np.zeros(self.num_features_)
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
            mean_acc = np.mean(scores) if scores else baseline_acc
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
    std_dev_feat0 = np.std(X[:,0], ddof=1); noisy_feat1 = rng.randn(n_samples, 1) * (std_dev_feat0 if std_dev_feat0 > 1e-6 else 1.0) * 0.5
    std_dev_feat1 = np.std(X[:,1], ddof=1); noisy_feat2 = rng.uniform(-1, 1, size=(n_samples, 1)) * (std_dev_feat1 if std_dev_feat1 > 1e-6 else 1.0) * 0.5
    petal_width_idx = feature_names_orig.index('petal width (cm)'); petal_width_col = X[:, petal_width_idx].reshape(-1, 1)
    std_dev_petal = np.std(petal_width_col, ddof=1); redundant_feat = petal_width_col * 0.7 + rng.normal(0, 0.1 * (std_dev_petal if std_dev_petal > 1e-6 else 0.1), size=(n_samples, 1))
    X_noisy = np.hstack((X, noisy_feat1, noisy_feat2, redundant_feat))
    feature_names_new = feature_names_orig + ['noisy_gaussian', 'noisy_uniform', 'redundant_petal_width']
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_noisy)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_names_new


def train_interpretable_models(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42); rf_clf.fit(X_train, y_train)
    dtree = DecisionTreeClassifier(random_state=42); dtree.fit(X_train, y_train)
    return rf_clf, dtree

def get_ground_truth_importance(model, X_train_shape_fallback):
    if hasattr(model, 'feature_importances_'): importances_raw = model.feature_importances_
    elif hasattr(model, 'coef_'):
        coef_raw = model.coef_
        if coef_raw.ndim > 1: importances_raw = np.mean(np.abs(coef_raw), axis=0)
        else: importances_raw = np.abs(coef_raw)
    else:
        return np.zeros(X_train_shape_fallback[1] if X_train_shape_fallback else 0)
    importances = np.nan_to_num(importances_raw, nan=0.0) 
    if len(importances) == 0: return np.array([])
    return importances

def run_dci(model_to_explain, X_ref, y_ref, **kwargs):
    medley_temp = MEDLEY(model_to_explain, model_type='classical'); medley_temp.fit(X_ref, y_ref)
    baseline_acc = accuracy_score(y_ref, medley_temp._predict(X_ref))
    importances = []
    for i in range(medley_temp.num_features_):
        X_drop = X_ref.copy(); X_drop[:, i] = 0.0
        drop_acc = accuracy_score(y_ref, medley_temp._predict(X_drop))
        importances.append(baseline_acc - drop_acc)
    scores = np.array(importances); return scores

def run_pi(model_to_explain, X_ref, y_ref, n_repeats=5, **kwargs):
    medley_temp = MEDLEY(model_to_explain, model_type='classical', n_repeats=n_repeats); medley_temp.fit(X_ref, y_ref)
    baseline_acc = accuracy_score(y_ref, medley_temp._predict(X_ref))
    importances = []
    for i in range(medley_temp.num_features_):
        perm_scores_list = []
        for _ in range(n_repeats):
            X_perm = X_ref.copy(); np.random.shuffle(X_perm[:, i])
            perm_acc = accuracy_score(y_ref, medley_temp._predict(X_perm))
            perm_scores_list.append(perm_acc)
        importances.append(baseline_acc - np.mean(perm_scores_list))
    scores = np.array(importances); return scores

def run_actual_medley(model_to_explain, X_ref, y_ref, n_repeats=5, **kwargs):
    explainer = MEDLEY(model_to_explain, model_type='classical', n_repeats=n_repeats); explainer.fit(X_ref, y_ref)
    scores = explainer.interpret(); return scores

def run_logreg_l1_coeffs(model_to_explain_dummy, X_ref, y_ref, **kwargs):
    num_features = X_ref.shape[1]
    importances = np.zeros(num_features)
    log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, C=0.1, max_iter=200)
    try:
        log_reg_l1.fit(X_ref, y_ref)
        if hasattr(log_reg_l1, 'coef_'):
            raw_coeffs = log_reg_l1.coef_
            if raw_coeffs.ndim > 1: 
                importances = np.mean(np.abs(raw_coeffs), axis=0)
            else: 
                importances = np.abs(raw_coeffs)
            if importances.shape[0] != num_features:
                importances = np.zeros(num_features)
        else:
            importances = np.zeros(num_features)
    except Exception:
        importances = np.zeros(num_features)
    importances = np.nan_to_num(importances, nan=0.0)
    return importances

def calculate_spearman_rank_correlation(true_importances, explainer_scores):
    if true_importances is None or explainer_scores is None or len(true_importances)==0 or len(explainer_scores)==0: return np.nan
    if len(true_importances) != len(explainer_scores): return np.nan
    true_importances=np.asarray(true_importances,dtype=float); explainer_scores=np.asarray(explainer_scores,dtype=float)
    true_importances=np.nan_to_num(true_importances,nan=0.0); explainer_scores=np.nan_to_num(explainer_scores,nan=0.0)
    if np.all(explainer_scores==explainer_scores[0]) and np.all(true_importances==true_importances[0]): return 1.0
    if np.all(explainer_scores==explainer_scores[0]) or np.all(true_importances==true_importances[0]): return 0.0
    try: correlation,p_value=spearmanr(true_importances,explainer_scores); return correlation if not np.isnan(correlation) else 0.0
    except Exception: return np.nan

def plot_rank_correlation_heatmap(results_df, plot_title_suffix=""):
    try: heatmap_data = results_df.pivot_table(index='model', columns='explainer', values='spearman_corr')
    except Exception: print(f"Error creating heatmap pivot_table."); return
    explainer_order = ["DCI", "PI (k=5)", "Q-MEDLEY", "LogRegL1"] 
    for col in explainer_order:
        if col not in heatmap_data.columns: heatmap_data[col] = np.nan
    heatmap_data = heatmap_data.reindex(columns=explainer_order)
    plt.figure(figsize=(10, 6)); sns.heatmap(heatmap_data.astype(float), annot=True, fmt=".2f", cmap="viridis_r", linewidths=.5, cbar_kws={'label': 'Spearman Rank Correlation'}, vmin=-1, vmax=1)
    plt.title(f'Spearman Rank Correlation of Explainers vs. Ground Truth{plot_title_suffix}\n(Noisy Iris Dataset)', fontsize=14)
    plt.xlabel('Explainer Type', fontsize=12); plt.ylabel('Interpretable Model Type', fontsize=12)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0); plt.tight_layout()
    filename = f"explainer_rank_corr_heatmap_noisy{plot_title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight'); print(f"\nRank correlation heatmap saved as {filename}"); plt.show(block=False)

if __name__ == "__main__":
    print("Starting Explainer Comparison Experiment (Noisy Iris - Rank Correlation Only)...")
    X_train, X_test, y_train, y_test, feature_names_new = load_and_prep_iris_noisy()
    
    if X_train.shape[0] == 0 or X_train.shape[1] == 0: exit("Error: X_train empty or has no features.")

    rf_clf, dtree = train_interpretable_models(X_train, y_train)

    models_to_test = { "Random Forest": rf_clf, "Decision Tree": dtree }

    explainers_to_run = {
        "DCI": run_dci,
        "PI (k=5)": run_pi,
        "Q-MEDLEY": run_actual_medley,
        "LogRegL1": run_logreg_l1_coeffs
    }

    results_list = []
    
    print(f"\nCalculating Spearman Rank Correlation...")
    for model_name_display, model_instance_to_explain in models_to_test.items():
        print(f"\n--- Explaining Model: {model_name_display} ---")
        true_full_importances = get_ground_truth_importance(model_instance_to_explain, X_train.shape)
        
        if len(true_full_importances) == 0 or X_train.shape[1] != len(true_full_importances):
             print(f"  Skipping model {model_name_display}: Invalid ground truth importances."); continue

        print(f"  Ground Truth Importances (for {model_name_display}) obtained.")
        
        for explainer_name, explainer_func in explainers_to_run.items():
            print(f"  Applying Explainer: {explainer_name}...")
            try:
                explainer_full_scores = explainer_func(
                    model_instance_to_explain, X_train, y_train, n_repeats=5
                )
                
                spearman_corr = calculate_spearman_rank_correlation(true_full_importances, explainer_full_scores)
                
                results_list.append({
                    'model': model_name_display, 
                    'explainer': explainer_name, 
                    'spearman_corr': spearman_corr, 
                })
                spearman_print_val = f"{spearman_corr:.3f}" if not np.isnan(spearman_corr) else "NaN"
                print(f"    Spearman Correlation: {spearman_print_val}")

            except Exception as e:
                 print(f"    ERROR running {explainer_name} for {model_name_display}: {e}")
                 import traceback; traceback.print_exc()
                 results_list.append({'model': model_name_display, 'explainer': explainer_name, 'spearman_corr': np.nan})
        
    results_df = pd.DataFrame(results_list)
    results_df['spearman_corr'] = pd.to_numeric(results_df['spearman_corr'], errors='coerce').fillna(0.0) 

    print("\n\n--- Comparison Results Summary (Spearman Rank Correlation) ---")
    if not results_df.empty and 'spearman_corr' in results_df.columns:
        try:
            spearman_summary_table = results_df.pivot_table(index='explainer', columns='model', values='spearman_corr', dropna=False)
            explainer_order_spearman = ["DCI", "PI (k=5)", "Q-MEDLEY", "LogRegL1"]
            spearman_summary_table = spearman_summary_table.reindex(index=explainer_order_spearman, fill_value=np.nan)
            print(spearman_summary_table.round(3))
            heatmap_df_for_plot = results_df.copy(); heatmap_df_for_plot['spearman_corr'] = heatmap_df_for_plot['spearman_corr'].fillna(0.0)
            plot_rank_correlation_heatmap(heatmap_df_for_plot, f"")
        except Exception as e: print(f"\nError Spearman summary: {e}")
    else: print("No Spearman results to display or plot.")

    plt.show() 
    print("\nExperiment Finished.")
