# src/__init__.py
"""
QuXAI: A Quantum eXplainable AI package.

Public API:
  - MEDLEY: quantum-aware feature importance explainer
  - train_q*: hybrid quantum-classical model trainers
  - plot_medley_scores: bar-chart visualisation of MEDLEY scores
"""

from .explainers.medley_explainer import MEDLEY
from .models.quantum_models import (
    sample_quantum_feature_map,
    train_qrf, train_qsvm, train_qknn,
    train_qlogistic, train_qdt, train_qnb, train_qada, train_qgb,
    train_qlda, train_qperceptron, train_qridge, train_qsvc_poly, train_qextra,
)
from .plotting.visualizations import plot_medley_scores

__all__ = [
    "MEDLEY",
    "sample_quantum_feature_map",
    "train_qrf", "train_qsvm", "train_qknn",
    "train_qlogistic", "train_qdt", "train_qnb", "train_qada", "train_qgb",
    "train_qlda", "train_qperceptron", "train_qridge", "train_qsvc_poly", "train_qextra",
    "plot_medley_scores",
]
