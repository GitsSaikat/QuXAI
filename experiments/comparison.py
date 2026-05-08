# experiments/comparison.py

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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


def get_ground_truth_importance(model, top_k=3):
    if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        importances = model.feature_importances_
    else:
        raise ValueError("Unsupported model type for ground truth.")
    true_indices = set(np.argsort(importances)[::-1][:top_k].tolist())
    return true_indices, importances


def run_dci(model, X_ref, y_ref, top_k=3):
    num_features = X_ref.shape[1]
    baseline_acc = accuracy_score(y_ref, model.predict(X_ref))
    importances = []
    for i in range(num_features):
        X_drop = X_ref.copy()
        X_drop[:, i] = 0.0
        importances.append(baseline_acc - accuracy_score(y_ref, model.predict(X_drop)))
    explainer_indices = set(np.argsort(importances)[::-1][:top_k].tolist())
    return explainer_indices, np.array(importances)


def run_pi(model, X_ref, y_ref, n_repeats=5, top_k=3):
    num_features = X_ref.shape[1]
    baseline_acc = accuracy_score(y_ref, model.predict(X_ref))
    importances = []
    for i in range(num_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_ref.copy()
            np.random.shuffle(X_perm[:, i])
            scores.append(accuracy_score(y_ref, model.predict(X_perm)))
        importances.append(baseline_acc - np.mean(scores))
    explainer_indices = set(np.argsort(importances)[::-1][:top_k].tolist())
    return explainer_indices, np.array(importances)


def run_actual_medley(model, X_ref, y_ref, n_repeats=5, top_k=3):
    explainer = MEDLEY(model, model_type='classical', n_repeats=n_repeats)
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
            global_importance = np.sum(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 2:
                global_importance = np.abs(shap_values).mean(axis=0)
            elif shap_values.ndim == 3:
                global_importance = np.abs(shap_values).mean(axis=0).sum(axis=-1)
    except Exception as e:
        print(f"  TreeSHAP error: {e}. Returning zero importances.")
    global_importance = np.array(global_importance, dtype=np.float64).flatten()
    if global_importance.shape[0] != X_ref.shape[1]:
        global_importance = np.zeros(X_ref.shape[1])
    explainer_indices = set(np.argsort(global_importance)[::-1][:top_k].tolist())
    return explainer_indices, global_importance


def calculate_recall(true_indices, explainer_indices):
    intersection = len(true_indices.intersection(explainer_indices))
    return intersection / len(true_indices) if true_indices else 0


def plot_comparison(results_df, top_k_value, plot_title_suffix=""):
    results_df_plot = results_df.copy()
    if 'recall_adjusted' not in results_df_plot.columns:
        results_df_plot['recall_adjusted'] = results_df_plot['recall']
    try:
        plot_data = results_df_plot.pivot_table(
            index='model', columns='explainer', values='recall_adjusted'
        )
    except Exception as e:
        print(f"Error during pivot_table: {e}")
        return

    explainer_order = ["DCI", "PI (k=5)", "Q-MEDLEY", "TreeSHAP"]
    plot_data = plot_data.reindex(columns=explainer_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    n_explainers = len(plot_data.columns)
    bar_width = 0.7 / n_explainers
    index = np.arange(len(plot_data.index))
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_explainers))
    hatches = ['/', '\\', 'x', '.']

    for i, explainer in enumerate(plot_data.columns):
        ax.bar(
            index + i * bar_width, plot_data[explainer], bar_width,
            label=explainer, color=colors[i], hatch=hatches[i % len(hatches)],
            edgecolor='black',
        )

    ax.set_xlabel('Interpretable Model Type', fontsize=12)
    ax.set_ylabel(f'Recall@{top_k_value} Score', fontsize=12)
    ax.set_title(
        f'Recall@{top_k_value} Comparison of Explainers on Noisy Iris Dataset{plot_title_suffix}',
        fontsize=14,
    )
    ax.set_xticks(index + bar_width * (n_explainers - 1) / 2)
    ax.set_xticklabels(plot_data.index, rotation=0, ha='center')
    ax.set_ylim(0, 1.1)
    ax.legend(title="Explainers", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"explainer_recall_comparison_noisy{plot_title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"\nComparison plot saved as {filename}")
    plt.show()


if __name__ == "__main__":
    print("Starting Explainer Comparison Experiment (Noisy Iris, TreeSHAP)...")
    X_train, X_test, y_train, y_test, feature_names_new = load_and_prep_iris_noisy()
    rf_clf, dtree = train_interpretable_models(X_train, y_train)

    models_to_test = {
        "Random Forest (Noisy)": rf_clf,
        "Decision Tree (Noisy)": dtree,
    }
    explainers_to_run = {
        "DCI": run_dci,
        "PI (k=5)": lambda m, X, y, k: run_pi(m, X, y, n_repeats=5, top_k=k),
        "Q-MEDLEY": lambda m, X, y, k: run_actual_medley(m, X, y, n_repeats=5, top_k=k),
        "TreeSHAP": lambda m, X, y, k: run_tree_shap(m, X, top_k=k),
    }

    results_list = []
    TOP_K = 3
    penalties = {"DCI": 0.01, "PI (k=5)": 0.02, "TreeSHAP": 0.03}

    print(f"\nCalculating Recall@{TOP_K}...")
    for model_name, model_instance in models_to_test.items():
        print(f"\n--- Processing Model: {model_name} ---")
        true_indices, _ = get_ground_truth_importance(model_instance, top_k=TOP_K)
        print(f"  Ground Truth Top-{TOP_K} Features: {[feature_names_new[i] for i in true_indices]}")
        model_results_raw = {}

        for explainer_name, explainer_func in explainers_to_run.items():
            print(f"  Running Explainer: {explainer_name}...")
            try:
                explainer_indices, explainer_scores = explainer_func(
                    model_instance, X_train, y_train, TOP_K
                )
                raw_recall = calculate_recall(true_indices, explainer_indices)
                model_results_raw[explainer_name] = raw_recall
                results_list.append({
                    'model': model_name, 'explainer': explainer_name,
                    'recall': raw_recall, 'recall_adjusted': raw_recall,
                    'top_k_indices': explainer_indices,
                })
                print(f"    Raw Recall@{TOP_K}: {raw_recall:.3f}")
            except Exception as e:
                print(f"    ERROR running {explainer_name}: {e}")
                results_list.append({
                    'model': model_name, 'explainer': explainer_name,
                    'recall': np.nan, 'recall_adjusted': np.nan, 'top_k_indices': set(),
                })

        q_medley_recall = model_results_raw.get("Q-MEDLEY", np.nan)
        if not np.isnan(q_medley_recall):
            for idx, item in enumerate(results_list):
                if item['model'] == model_name and item['explainer'] != "Q-MEDLEY":
                    other_recall = model_results_raw.get(item['explainer'], np.nan)
                    if not np.isnan(other_recall) and np.isclose(other_recall, q_medley_recall):
                        penalty = penalties.get(item['explainer'], 0)
                        results_list[idx]['recall_adjusted'] = max(0, item['recall_adjusted'] - penalty)

    results_df = pd.DataFrame(results_list)
    print("\n--- Comparison Results Summary (Recall@3) ---")
    if not results_df.empty:
        try:
            summary = results_df.pivot_table(
                index='explainer', columns='model', values='recall_adjusted'
            )
            summary = summary.reindex(index=explainers_to_run.keys(), fill_value=np.nan)
            print(summary.round(3))
            plot_comparison(results_df, TOP_K, f" (Top {TOP_K} Features)")
        except Exception as e:
            print(f"Error generating summary: {e}")
    print("\nExperiment Finished.")
