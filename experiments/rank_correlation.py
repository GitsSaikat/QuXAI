# experiments/rank_correlation.py

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

from explainers.medley_explainer import MEDLEY
from data_processing import _add_noise_and_redundancy


def load_and_prep_iris_noisy():
    iris = load_iris()
    X_scaled, feature_names = _add_noise_and_redundancy(
        iris.data, list(iris.feature_names), random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )
    return X_train, X_test, y_train, y_test, feature_names


def train_interpretable_models(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, y_train)
    return rf_clf, dtree


def get_ground_truth_importance(model, X_train_shape_fallback):
    if hasattr(model, 'feature_importances_'):
        importances_raw = model.feature_importances_
    elif hasattr(model, 'coef_'):
        coef_raw = model.coef_
        importances_raw = np.mean(np.abs(coef_raw), axis=0) if coef_raw.ndim > 1 else np.abs(coef_raw)
    else:
        return np.zeros(X_train_shape_fallback[1])
    return np.nan_to_num(importances_raw, nan=0.0)


def run_dci(model_to_explain, X_ref, y_ref, **kwargs):
    explainer = MEDLEY(model_to_explain, model_type='classical')
    explainer.fit(X_ref, y_ref)
    baseline_acc = accuracy_score(y_ref, explainer._predict(X_ref))
    importances = []
    for i in range(explainer.num_features_):
        X_drop = X_ref.copy()
        X_drop[:, i] = 0.0
        importances.append(baseline_acc - accuracy_score(y_ref, explainer._predict(X_drop)))
    return np.array(importances)


def run_pi(model_to_explain, X_ref, y_ref, n_repeats=5, **kwargs):
    explainer = MEDLEY(model_to_explain, model_type='classical', n_repeats=n_repeats)
    explainer.fit(X_ref, y_ref)
    baseline_acc = accuracy_score(y_ref, explainer._predict(X_ref))
    importances = []
    for i in range(explainer.num_features_):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_ref.copy()
            np.random.shuffle(X_perm[:, i])
            scores.append(accuracy_score(y_ref, explainer._predict(X_perm)))
        importances.append(baseline_acc - np.mean(scores))
    return np.array(importances)


def run_actual_medley(model_to_explain, X_ref, y_ref, n_repeats=5, **kwargs):
    explainer = MEDLEY(model_to_explain, model_type='classical', n_repeats=n_repeats)
    explainer.fit(X_ref, y_ref)
    return explainer.interpret()


def run_logreg_l1_coeffs(model_to_explain_dummy, X_ref, y_ref, **kwargs):
    num_features = X_ref.shape[1]
    log_reg = LogisticRegression(
        penalty='l1', solver='liblinear', random_state=42, C=0.1, max_iter=200
    )
    try:
        log_reg.fit(X_ref, y_ref)
        coef = log_reg.coef_
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        if importances.shape[0] != num_features:
            importances = np.zeros(num_features)
    except Exception:
        importances = np.zeros(num_features)
    return np.nan_to_num(importances, nan=0.0)


def calculate_spearman_rank_correlation(true_importances, explainer_scores):
    if len(true_importances) != len(explainer_scores):
        return np.nan
    a = np.nan_to_num(np.asarray(true_importances, dtype=float), nan=0.0)
    b = np.nan_to_num(np.asarray(explainer_scores, dtype=float), nan=0.0)
    if np.all(b == b[0]) and np.all(a == a[0]):
        return 1.0
    if np.all(b == b[0]) or np.all(a == a[0]):
        return 0.0
    try:
        corr, _ = spearmanr(a, b)
        return corr if not np.isnan(corr) else 0.0
    except Exception:
        return np.nan


def plot_rank_correlation_heatmap(results_df, plot_title_suffix=""):
    try:
        heatmap_data = results_df.pivot_table(
            index='model', columns='explainer', values='spearman_corr'
        )
    except Exception:
        print("Error creating heatmap pivot_table.")
        return
    explainer_order = ["DCI", "PI (k=5)", "Q-MEDLEY", "LogRegL1"]
    for col in explainer_order:
        if col not in heatmap_data.columns:
            heatmap_data[col] = np.nan
    heatmap_data = heatmap_data.reindex(columns=explainer_order)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data.astype(float), annot=True, fmt=".2f",
        cmap="viridis_r", linewidths=0.5,
        cbar_kws={'label': 'Spearman Rank Correlation'}, vmin=-1, vmax=1,
    )
    plt.title(
        f'Spearman Rank Correlation of Explainers vs. Ground Truth{plot_title_suffix}\n'
        '(Noisy Iris Dataset)', fontsize=14,
    )
    plt.xlabel('Explainer Type', fontsize=12)
    plt.ylabel('Interpretable Model Type', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = f"explainer_rank_corr_heatmap_noisy{plot_title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"\nRank correlation heatmap saved as {filename}")
    plt.show(block=False)


if __name__ == "__main__":
    print("Starting Explainer Comparison Experiment (Noisy Iris - Rank Correlation)...")
    X_train, X_test, y_train, y_test, feature_names_new = load_and_prep_iris_noisy()

    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        exit("Error: X_train empty or has no features.")

    rf_clf, dtree = train_interpretable_models(X_train, y_train)
    models_to_test = {"Random Forest": rf_clf, "Decision Tree": dtree}
    explainers_to_run = {
        "DCI": run_dci,
        "PI (k=5)": run_pi,
        "Q-MEDLEY": run_actual_medley,
        "LogRegL1": run_logreg_l1_coeffs,
    }

    results_list = []
    print("\nCalculating Spearman Rank Correlation...")
    for model_name, model_instance in models_to_test.items():
        print(f"\n--- Explaining Model: {model_name} ---")
        true_importances = get_ground_truth_importance(model_instance, X_train.shape)
        if len(true_importances) == 0 or X_train.shape[1] != len(true_importances):
            print(f"  Skipping {model_name}: invalid ground truth importances.")
            continue

        for explainer_name, explainer_func in explainers_to_run.items():
            print(f"  Applying Explainer: {explainer_name}...")
            try:
                scores = explainer_func(model_instance, X_train, y_train, n_repeats=5)
                corr = calculate_spearman_rank_correlation(true_importances, scores)
                results_list.append({
                    'model': model_name, 'explainer': explainer_name, 'spearman_corr': corr,
                })
                print(f"    Spearman Correlation: {corr:.3f}" if not np.isnan(corr) else "    Spearman Correlation: NaN")
            except Exception as e:
                import traceback
                print(f"    ERROR running {explainer_name}: {e}")
                traceback.print_exc()
                results_list.append({'model': model_name, 'explainer': explainer_name, 'spearman_corr': np.nan})

    results_df = pd.DataFrame(results_list)
    results_df['spearman_corr'] = pd.to_numeric(results_df['spearman_corr'], errors='coerce').fillna(0.0)

    print("\n\n--- Comparison Results (Spearman Rank Correlation) ---")
    if not results_df.empty:
        try:
            summary = results_df.pivot_table(
                index='explainer', columns='model', values='spearman_corr', dropna=False
            )
            summary = summary.reindex(index=["DCI", "PI (k=5)", "Q-MEDLEY", "LogRegL1"], fill_value=np.nan)
            print(summary.round(3))
            plot_rank_correlation_heatmap(results_df)
        except Exception as e:
            print(f"Error generating summary: {e}")
    else:
        print("No results to display.")

    plt.show()
    print("\nExperiment Finished.")
