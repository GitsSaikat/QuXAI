# experiments/quxai_eval.py

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pennylane as qml

from models.quantum_models import (
    sample_quantum_feature_map, CLASSICAL_PARAMS, MAX_QUBITS,
    train_qrf, train_qlogistic, train_qdt, train_qnb, train_qada,
    train_qgb, train_qlda, train_qperceptron, train_qridge, train_qextra,
)
from data_processing import _add_noise_and_redundancy


HQML_CONFIG = {
    "QRF":        {"train_func": train_qrf,        "classical_equiv": RandomForestClassifier},
    "QLogistic":  {"train_func": train_qlogistic,  "classical_equiv": LogisticRegression},
    "QDT":        {"train_func": train_qdt,         "classical_equiv": DecisionTreeClassifier},
    "QNB":        {"train_func": train_qnb,         "classical_equiv": GaussianNB},
    "QAda":       {"train_func": train_qada,        "classical_equiv": AdaBoostClassifier},
    "QGB":        {"train_func": train_qgb,         "classical_equiv": GradientBoostingClassifier},
    "QLDA":       {"train_func": train_qlda,        "classical_equiv": LinearDiscriminantAnalysis},
    "QPerceptron":{"train_func": train_qperceptron, "classical_equiv": Perceptron},
    "QRidge":     {"train_func": train_qridge,      "classical_equiv": RidgeClassifier},
    "QExtra":     {"train_func": train_qextra,      "classical_equiv": ExtraTreesClassifier},
}


def evaluate_predictions(y_true, y_pred):
    return {
        'Accuracy':        accuracy_score(y_true, y_pred),
        'F1-Macro':        f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision-Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall-Macro':    recall_score(y_true, y_pred, average='macro', zero_division=0),
    }


def get_hqml_model_predictions(hqml_model, X_test_eval):
    model_type = getattr(hqml_model, 'hqml_model_type', None)
    if model_type is None:
        raise ValueError("HQML model requires 'hqml_model_type' attribute.")

    num_qubits = hqml_model.num_qubits
    qfm = hqml_model.quantum_feature_map
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def q_node_state(x):
        qfm(x, num_qubits)
        return qml.state()

    amplitude_models = {
        'qrf', 'qlogistic', 'qdt', 'qnb', 'qada',
        'qgb', 'qlda', 'qperceptron', 'qridge', 'qextra',
    }
    if model_type in amplitude_models:
        X_sub = X_test_eval[:, :num_qubits]
        X_q = np.array([np.abs(q_node_state(x)) ** 2 for x in X_sub])
        return hqml_model.predict(X_q)
    raise ValueError(f"Unknown HQML model type for prediction: {model_type}")


def load_and_prep_data(dataset_loader_func, dataset_name, random_state=42, sample_frac=1.0):
    data_full = dataset_loader_func()

    if sample_frac < 1.0:
        df_temp = pd.DataFrame(data_full.data, columns=data_full.feature_names)
        df_temp['target'] = data_full.target
        try:
            df_sampled = (
                df_temp.groupby('target', group_keys=False)
                .apply(lambda x: x.sample(frac=sample_frac, random_state=random_state))
                .reset_index(drop=True)
            )
        except ValueError:
            df_sampled = df_temp.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        X_orig = df_sampled.drop('target', axis=1).values
        y = df_sampled['target'].values
        feature_names_orig = list(df_sampled.drop('target', axis=1).columns)
    else:
        X_orig = data_full.data
        y = data_full.target
        feature_names_orig = list(data_full.feature_names)

    X_noisy, feature_names_noisy = _add_noise_and_redundancy(X_orig, feature_names_orig, random_state)

    unique_y, counts_y = np.unique(y, return_counts=True)
    stratify_y = y if len(unique_y) > 1 and all(c >= 2 for c in counts_y) else None

    if len(X_noisy) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y, test_size=0.3, random_state=random_state, stratify=stratify_y
    )
    return X_train, X_test, y_train, y_test, feature_names_noisy


if __name__ == "__main__":
    print("Starting QuXAI Framework Evaluation Experiment (All 10 Amplitude Models)...")

    dataset_loaders = {
        "Iris-Noisy":        lambda: load_and_prep_data(load_iris, "Iris", sample_frac=1.0),
        "Wine-Noisy (5%)":   lambda: load_and_prep_data(load_wine, "Wine", sample_frac=0.05),
        "BreastCancer-Noisy":lambda: load_and_prep_data(load_breast_cancer, "BCancer", sample_frac=1.0),
    }

    all_results_list = []

    for dataset_name, loader_func in dataset_loaders.items():
        print(f"\n--- Processing Dataset: {dataset_name} ---")
        X_train, X_test, y_train, y_test, feature_names = loader_func()

        if X_train.shape[0] < 2 or X_test.shape[0] < 1:
            print(f"Skipping {dataset_name}: insufficient samples.")
            continue
        if len(np.unique(y_train)) < 2:
            print(f"Skipping {dataset_name}: insufficient classes in train set.")
            continue

        for hqml_name, config in HQML_CONFIG.items():
            print(f"  -- Model Type: {hqml_name} --")
            ClassicalModelClass = config["classical_equiv"]
            params = CLASSICAL_PARAMS.get(ClassicalModelClass.__name__, {})

            print(f"    Training Classical Baseline: {ClassicalModelClass.__name__}...")
            try:
                classical_model = ClassicalModelClass(**params)
                classical_model.fit(X_train, y_train)
                perf_classical = evaluate_predictions(y_test, classical_model.predict(X_test))
                all_results_list.append({
                    'Dataset': dataset_name, 'Model': hqml_name,
                    'Framework_Type': 'Classical', **perf_classical,
                })
                print(f"      Accuracy: {perf_classical['Accuracy']:.3f}")
            except Exception as e:
                print(f"      ERROR Classical {ClassicalModelClass.__name__}: {e}")
                all_results_list.append({
                    'Dataset': dataset_name, 'Model': hqml_name, 'Framework_Type': 'Classical',
                    'Accuracy': np.nan, 'F1-Macro': np.nan,
                    'Precision-Macro': np.nan, 'Recall-Macro': np.nan,
                })

            print(f"    Training HQML Model: {hqml_name}...")
            try:
                trained_hqml = config["train_func"](X_train, y_train)
                y_pred_hqml = get_hqml_model_predictions(trained_hqml, X_test)
                perf_hqml = evaluate_predictions(y_test, y_pred_hqml)
                all_results_list.append({
                    'Dataset': dataset_name, 'Model': hqml_name,
                    'Framework_Type': 'QuXAI_HQML', **perf_hqml,
                })
                print(f"      Accuracy: {perf_hqml['Accuracy']:.3f}")
            except Exception as e:
                import traceback
                print(f"      ERROR HQML {hqml_name}: {e}")
                traceback.print_exc()
                all_results_list.append({
                    'Dataset': dataset_name, 'Model': hqml_name, 'Framework_Type': 'QuXAI_HQML',
                    'Accuracy': np.nan, 'F1-Macro': np.nan,
                    'Precision-Macro': np.nan, 'Recall-Macro': np.nan,
                })

    print("\n\n--- QuXAI Framework Evaluation: Quantitative Performance Table ---")
    if not all_results_list:
        print("No results were generated.")
    else:
        results_df = pd.DataFrame(all_results_list)
        try:
            pivot = results_df.pivot_table(
                index=['Dataset', 'Model'],
                columns='Framework_Type',
                values=['Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro'],
            )
            pivot = pivot.reindex(columns=['Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro'], level=0)
            pivot = pivot.reindex(columns=['Classical', 'QuXAI_HQML'], level=1)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            print(pivot.round(3))
        except Exception as e:
            print(f"Could not create pivot table: {e}")
            cols = ['Dataset', 'Model', 'Framework_Type', 'Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro']
            print(results_df[[c for c in cols if c in results_df.columns]].sort_values(
                by=['Dataset', 'Model', 'Framework_Type']
            ).reset_index(drop=True).round(3))

    print("\nExperiment Finished.")
