# experiments/qmedley_res_wine.py

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from explainers.medley_explainer import MEDLEY
from models.quantum_models import (
    sample_quantum_feature_map,
    train_qrf, train_qlogistic, train_qdt, train_qnb, train_qada,
    train_qgb, train_qlda, train_qperceptron, train_qridge, train_qextra,
)
from data_processing import load_wine_noisy

if __name__ == "__main__":
    print("Starting Q-MEDLEY application on Amplitude HQML models (Noisy Wine)...")
    X_train, X_test, y_train, y_test, feature_names = load_wine_noisy()

    hqml_model_trainers = {
        "Q.RandomForest":  train_qrf,
        "Q.LogRegression": train_qlogistic,
        "Q.DecisionTree":  train_qdt,
        "Q.NaiveBayes":    train_qnb,
        "Q.AdaBoost":      train_qada,
        "Q.GradBoost":     train_qgb,
        "Q.LDA":           train_qlda,
        "Q.Perceptron":    train_qperceptron,
        "Q.Ridge":         train_qridge,
        "Q.ExtraTrees":    train_qextra,
    }

    all_medley_scores = {}
    N_REPEATS_MEDLEY = 3
    FIGURE_COLS = 2

    print(f"\nApplying Q-MEDLEY (n_repeats={N_REPEATS_MEDLEY})...")
    for model_display_name, train_func in hqml_model_trainers.items():
        print(f"\n--- Processing HQML Model: {model_display_name} ---")
        try:
            print("  Training HQML model...")
            trained_model = train_func(X_train, y_train)

            print("  Applying Q-MEDLEY explainer...")
            explainer = MEDLEY(
                pretrained_model=trained_model,
                model_type=trained_model.hqml_model_type,
                quantum_feature_map=trained_model.quantum_feature_map,
                n_repeats=N_REPEATS_MEDLEY,
            )
            explainer.fit(X_train, y_train)
            importance_scores = explainer.interpret()
            all_medley_scores[model_display_name] = importance_scores

            print(f"    Q-MEDLEY Importances: {np.round(importance_scores, 4)}")
            top_indices = np.argsort(importance_scores)[::-1][:3]
            print(f"    Top 3 features: {[feature_names[i] for i in top_indices if i < len(feature_names)]}")

        except Exception as e:
            print(f"    ERROR processing {model_display_name}: {e}")
            all_medley_scores[model_display_name] = np.zeros(X_train.shape[1])

    if all_medley_scores:
        num_models = len(all_medley_scores)
        cols = FIGURE_COLS
        rows = (num_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
        axes_flat = axes.flatten()
        sorted_names = sorted(all_medley_scores.keys())

        for i, model_name in enumerate(sorted_names):
            scores = all_medley_scores[model_name]
            ax = axes_flat[i]
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, scores, align='center', color='mediumseagreen', edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Q-MEDLEY Score', fontsize=9)
            ax.set_title(model_name, fontsize=10)
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        for i in range(num_models, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        fig.suptitle(
            "Q-MEDLEY for Amplitude-Encoded HQML Models on Noisy Wine Data",
            fontsize=14, y=1.02 if rows > 1 else 1.05,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98 if rows > 1 else 0.95])
        plt.savefig("qmedley_amplitude_hqml_wine_noisy.png", dpi=150)
        print("\nQ-MEDLEY comparison plot saved as qmedley_amplitude_hqml_wine_noisy.png")
        plt.show()
    else:
        print("No Q-MEDLEY scores were generated to plot.")

    print("\nExperiment Finished.")
