import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pennylane as qml

MAX_QUBITS_FOR_QMAP = 8

def sample_quantum_feature_map(x, num_qubits):
    for i in range(num_qubits):
        qml.RX(float(x[i]), wires=i)

def _train_amplitude_model_generic(X_train, y_train, classical_model_class, model_params, hqml_model_type_str, feature_map_func):
    input_num_classical_features = X_train.shape[1]
    num_qubits_for_qmap = min(input_num_classical_features, MAX_QUBITS_FOR_QMAP)

    if input_num_classical_features > MAX_QUBITS_FOR_QMAP:
        print(f"      Info for {hqml_model_type_str}: Input classical features ({input_num_classical_features}) > MAX_QUBITS_FOR_QMAP ({MAX_QUBITS_FOR_QMAP}). Quantum mapping will use first {num_qubits_for_qmap} features.")

    X_train_subset_for_qfm = X_train[:, :num_qubits_for_qmap]

    dev = qml.device("default.qubit", wires=num_qubits_for_qmap)

    @qml.qnode(dev)
    def circuit(x_qfm_input):
        feature_map_func(x_qfm_input, num_qubits_for_qmap)
        return qml.state()

    X_train_q_features = np.array([np.abs(circuit(x_row)) for x_row in X_train_subset_for_qfm])

    model = classical_model_class(**model_params)
    model.fit(X_train_q_features, y_train)

    model.num_qubits = num_qubits_for_qmap
    model.quantum_feature_map = feature_map_func
    model.hqml_model_type = hqml_model_type_str
    return model

def train_qrf(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("RandomForestClassifier")
    return _train_amplitude_model_generic(X_train, y_train, RandomForestClassifier, params, 'qrf', feature_map)

def train_qlogistic(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("LogisticRegression")
    return _train_amplitude_model_generic(X_train, y_train, LogisticRegression, params, 'qlogistic', feature_map)

def train_qdt(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("DecisionTreeClassifier")
    return _train_amplitude_model_generic(X_train, y_train, DecisionTreeClassifier, params, 'qdt', feature_map)

def train_qnb(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("GaussianNB")
    return _train_amplitude_model_generic(X_train, y_train, GaussianNB, params, 'qnb', feature_map)

def train_qada(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("AdaBoostClassifier")
    return _train_amplitude_model_generic(X_train, y_train, AdaBoostClassifier, params, 'qada', feature_map)

def train_qgb(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("GradientBoostingClassifier")
    return _train_amplitude_model_generic(X_train, y_train, GradientBoostingClassifier, params, 'qgb', feature_map)

def train_qlda(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("LinearDiscriminantAnalysis")
    return _train_amplitude_model_generic(X_train, y_train, LinearDiscriminantAnalysis, params, 'qlda', feature_map)

def train_qperceptron(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("Perceptron")
    return _train_amplitude_model_generic(X_train, y_train, Perceptron, params, 'qperceptron', feature_map)

def train_qridge(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("RidgeClassifier")
    return _train_amplitude_model_generic(X_train, y_train, RidgeClassifier, params, 'qridge', feature_map)

def train_qextra(X_train, y_train, feature_map=sample_quantum_feature_map):
    params = CLASSICAL_PARAMS.get("ExtraTreesClassifier")
    return _train_amplitude_model_generic(X_train, y_train, ExtraTreesClassifier, params, 'qextra', feature_map)


def _add_noise_and_redundancy(X_orig, feature_names_orig, dataset_name="dataset", random_state=42):
    n_samples = X_orig.shape[0]
    rng = np.random.RandomState(random_state)
    X_processed = X_orig.copy()
    feature_names_processed = list(feature_names_orig)

    if X_orig.shape[1] == 0:
        return X_orig, feature_names_orig

    std_mean_orig = np.std(X_orig, axis=0).mean() if X_orig.shape[0] > 1 else 1.0
    noisy_feat1 = rng.randn(n_samples, 1) * std_mean_orig * 0.3
    noisy_feat2 = rng.uniform(-1, 1, size=(n_samples, 1)) * std_mean_orig * 0.3
    X_processed = np.hstack((X_processed, noisy_feat1, noisy_feat2))
    feature_names_processed.extend([f'{dataset_name}_noisy_g', f'{dataset_name}_noisy_u'])

    if X_orig.shape[1] > 0:
        variances = np.var(X_orig, axis=0)
        if len(variances) > 0 :
            primary_feat_idx = np.argmax(variances)
            primary_feat_for_redundancy = X_orig[:, primary_feat_idx].reshape(-1, 1)
            std_primary = np.std(primary_feat_for_redundancy, ddof=1) if primary_feat_for_redundancy.shape[0] > 1 else 0.1
            redundant_feat_vals = primary_feat_for_redundancy * 0.7 + \
                                rng.normal(0, 0.05 * (std_primary if std_primary > 1e-6 else 0.1), size=(n_samples, 1))
            X_processed = np.hstack((X_processed, redundant_feat_vals))
            original_feat_name_short = feature_names_orig[primary_feat_idx]
            if len(original_feat_name_short) > 10: original_feat_name_short = original_feat_name_short[:7] + "..."
            feature_names_processed.append(f'{dataset_name}_redundant_{original_feat_name_short}')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    return X_scaled, feature_names_processed

def load_and_prep_data(dataset_loader_func, dataset_name, random_state=42, sample_frac=1.0):
    data_full = dataset_loader_func()

    if sample_frac < 1.0:
        df_temp = pd.DataFrame(data_full.data, columns=data_full.feature_names)
        df_temp['target'] = data_full.target

        unique_classes_initial, counts_initial = np.unique(df_temp['target'], return_counts=True)
        min_samples_needed_per_class_for_stratify_sample = 2

        can_stratify_sample = len(unique_classes_initial) > 1 and all(c >= min_samples_needed_per_class_for_stratify_sample for c in counts_initial)

        if can_stratify_sample:
            try:
                df_sampled = df_temp.groupby('target', group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=random_state, replace=False) if len(x) * sample_frac >=1 else x.sample(n=1, random_state=random_state,replace=False) ).reset_index(drop=True)
            except ValueError:
                df_sampled = df_temp.sample(frac=sample_frac, random_state=random_state, replace=False).reset_index(drop=True)
        else:
            df_sampled = df_temp.sample(frac=sample_frac, random_state=random_state, replace=False).reset_index(drop=True)

        X_orig = df_sampled.drop('target', axis=1).values
        y = df_sampled['target'].values
        feature_names_orig = list(df_sampled.drop('target', axis=1).columns)
    else:
        X_orig = data_full.data
        y = data_full.target
        feature_names_orig = list(data_full.feature_names)

    X_noisy, feature_names_noisy = _add_noise_and_redundancy(X_orig, feature_names_orig, dataset_name, random_state)

    stratify_y_split = None
    if len(y) > 0: # Ensure y is not empty
        unique_y_for_split_check, counts_y_for_split_check = np.unique(y, return_counts=True)
        if len(unique_y_for_split_check) > 1 and all(c >= 2 for c in counts_y_for_split_check):
            stratify_y_split = y

    if len(X_noisy) == 0: # If X_noisy became empty after sampling (very small frac)
        return np.array([]), np.array([]), np.array([]), np.array([]), []


    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y, test_size=0.3, random_state=random_state, stratify=stratify_y_split
    )
    return X_train, X_test, y_train, y_test, feature_names_noisy

HQML_CONFIG = {
    "QRF": {"train_func": train_qrf, "classical_equiv": RandomForestClassifier},
    "QLogistic": {"train_func": train_qlogistic, "classical_equiv": LogisticRegression},
    "QDT": {"train_func": train_qdt, "classical_equiv": DecisionTreeClassifier},
    "QNB": {"train_func": train_qnb, "classical_equiv": GaussianNB},
    "QAda": {"train_func": train_qada, "classical_equiv": AdaBoostClassifier},
    "QGB": {"train_func": train_qgb, "classical_equiv": GradientBoostingClassifier},
    "QLDA": {"train_func": train_qlda, "classical_equiv": LinearDiscriminantAnalysis},
    "QPerceptron": {"train_func": train_qperceptron, "classical_equiv": Perceptron},
    "QRidge": {"train_func": train_qridge, "classical_equiv": RidgeClassifier},
    "QExtra": {"train_func": train_qextra, "classical_equiv": ExtraTreesClassifier},
}

CLASSICAL_PARAMS = {
    "RandomForestClassifier": {'n_estimators': 50, 'random_state': 42, 'max_depth': 5},
    "LogisticRegression": {'max_iter': 200, 'random_state': 42, 'solver': 'liblinear'},
    "DecisionTreeClassifier": {'random_state': 42, 'max_depth': 5},
    "GaussianNB": {},
    "AdaBoostClassifier": {'n_estimators': 30, 'random_state': 42},
    "GradientBoostingClassifier": {'n_estimators': 30, 'random_state': 42, 'max_depth': 3},
    "LinearDiscriminantAnalysis": {},
    "Perceptron": {'max_iter': 1000, 'tol': 1e-3, 'random_state': 42},
    "RidgeClassifier": {'random_state': 42},
    "ExtraTreesClassifier": {'n_estimators': 50, 'random_state': 42, 'max_depth': 5},
}

def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'Accuracy': accuracy,
        'F1-Macro': f1,
        'Precision-Macro': precision,
        'Recall-Macro': recall
    }

def get_hqml_model_predictions(hqml_model, X_test_eval):
    model_specific_type = getattr(hqml_model, 'hqml_model_type', None)
    if model_specific_type is None:
        raise ValueError("HQML model requires 'hqml_model_type' attribute.")

    num_qubits_used_in_model = getattr(hqml_model, 'num_qubits')
    qfm = getattr(hqml_model, 'quantum_feature_map')
    dev = qml.device("default.qubit", wires=num_qubits_used_in_model)

    @qml.qnode(dev)
    def q_node_state(x_input_for_qfm):
        qfm(x_input_for_qfm, num_qubits_used_in_model)
        return qml.state()

    amplitude_models_list = ['qrf', 'qlogistic', 'qdt', 'qnb', 'qada', 'qgb', 'qlda', 'qperceptron', 'qridge', 'qextra']

    if model_specific_type in amplitude_models_list:
        X_test_subset_for_qfm = X_test_eval[:, :num_qubits_used_in_model]
        X_test_q = np.array([np.abs(q_node_state(x_row)) for x_row in X_test_subset_for_qfm])
        y_pred = hqml_model.predict(X_test_q)
    else:
        raise ValueError(f"Unknown HQML model type for prediction: {model_specific_type}")
    return y_pred

if __name__ == "__main__":
    print("Starting QuXAI Framework Evaluation Experiment (All 10 Amplitude Models)...")

    dataset_loaders = {
        "Iris-Noisy": lambda: load_and_prep_data(load_iris, "Iris", sample_frac=1.0),
        "Wine-Noisy (5%)": lambda: load_and_prep_data(load_wine, "Wine", sample_frac=0.05),
        "BreastCancer-Noisy": lambda: load_and_prep_data(load_breast_cancer, "BCancer", sample_frac=1.0),
    }

    all_results_list = []

    for dataset_name, loader_func in dataset_loaders.items():
        print(f"\n--- Processing Dataset: {dataset_name} ---")
        X_train, X_test, y_train, y_test, feature_names = loader_func()

        if X_train.shape[0] < 2 or X_test.shape[0] < 1 or y_train.shape[0] < 2 or y_test.shape[0] < 1:
            print(f"Skipping {dataset_name} due to insufficient samples after processing or sampling.")
            continue
        if len(np.unique(y_train)) < 2:
             print(f"Skipping {dataset_name} due to insufficient classes in train set ({len(np.unique(y_train))}) for reliable model training/evaluation.")
             continue
        if len(np.unique(y_test)) < 1: # y_test can have 1 class if y_train has more for prediction
             print(f"Warning for {dataset_name}: Test set has only ({len(np.unique(y_test))}) class(es). Metrics might be limited.")


        for hqml_name, config in HQML_CONFIG.items():
            print(f"  -- Model Type: {hqml_name} --")

            ClassicalModelClass = config["classical_equiv"]
            classical_model_name = ClassicalModelClass.__name__
            params = CLASSICAL_PARAMS.get(classical_model_name, {})

            print(f"    Training Classical Baseline: {classical_model_name}...")
            try:
                classical_model = ClassicalModelClass(**params)
                classical_model.fit(X_train, y_train)
                y_pred_classical = classical_model.predict(X_test)
                perf_classical = evaluate_predictions(y_test, y_pred_classical)
                result_classical = {'Dataset': dataset_name, 'Model': hqml_name, 'Framework_Type': 'Classical', **perf_classical}
                all_results_list.append(result_classical)
                print(f"      Classical Baseline ({classical_model_name} for {hqml_name}) Accuracy: {perf_classical['Accuracy']:.3f}")
            except Exception as e:
                print(f"      ERROR Classical Baseline {classical_model_name}: {e}")
                result_classical = {'Dataset': dataset_name, 'Model': hqml_name, 'Framework_Type': 'Classical',
                                    'Accuracy': np.nan, 'F1-Macro': np.nan, 'Precision-Macro': np.nan, 'Recall-Macro': np.nan}
                all_results_list.append(result_classical)

            print(f"    Training HQML Model: {hqml_name}...")
            try:
                train_hqml_function = config["train_func"]
                trained_hqml_model = train_hqml_function(X_train, y_train, feature_map=sample_quantum_feature_map)

                if not hasattr(trained_hqml_model, 'hqml_model_type'):
                     trained_hqml_model.hqml_model_type = hqml_name.lower()
                if not hasattr(trained_hqml_model, 'num_qubits'):
                     trained_hqml_model.num_qubits = min(X_train.shape[1], MAX_QUBITS_FOR_QMAP)
                if not hasattr(trained_hqml_model, 'quantum_feature_map'):
                     trained_hqml_model.quantum_feature_map = sample_quantum_feature_map

                y_pred_hqml = get_hqml_model_predictions(trained_hqml_model, X_test)
                perf_hqml = evaluate_predictions(y_test, y_pred_hqml)
                result_hqml = {'Dataset': dataset_name, 'Model': hqml_name, 'Framework_Type': 'QuXAI_HQML', **perf_hqml}
                all_results_list.append(result_hqml)
                print(f"      HQML Model ({hqml_name}) Accuracy: {perf_hqml['Accuracy']:.3f}")
            except Exception as e:
                print(f"      ERROR HQML Model {hqml_name}: {e}")
                import traceback
                traceback.print_exc()
                result_hqml = {'Dataset': dataset_name, 'Model': hqml_name, 'Framework_Type': 'QuXAI_HQML',
                               'Accuracy': np.nan, 'F1-Macro': np.nan, 'Precision-Macro': np.nan, 'Recall-Macro': np.nan}
                all_results_list.append(result_hqml)

    print("\n\n--- QuXAI Framework Evaluation: Quantitative Performance Table ---")
    if not all_results_list:
        print("No results were generated.")
    else:
        results_df = pd.DataFrame(all_results_list)

        try:
            pivot_results = results_df.pivot_table(
                index=['Dataset', 'Model'],
                columns='Framework_Type',
                values=['Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro']
            )

            pivot_results = pivot_results.reindex(columns=['Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro'], level=0)
            pivot_results = pivot_results.reindex(columns=['Classical', 'QuXAI_HQML'], level=1)

            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            print(pivot_results.round(3))
        except Exception as e:
            print(f"Could not create pivot table: {e}")
            print("Displaying flat table instead:")
            cols_order = ['Dataset', 'Model', 'Framework_Type', 'Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro']
            cols_existing = [col for col in cols_order if col in results_df.columns]
            results_df_styled = results_df[cols_existing].sort_values(by=['Dataset', 'Model', 'Framework_Type']).reset_index(drop=True)
            print(results_df_styled.round(3))

    print("\nExperiment Finished.")
