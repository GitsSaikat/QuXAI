# QuXAI — Repository Summary

> **Paper:** Barua et al., *QuXAI: Explainers for Hybrid Quantum Machine Learning Models*, arXiv:2505.10167 (2025)
> **License:** Apache 2.0 | **Python:** 3.8+

---

## Purpose

QuXAI is a research framework that brings explainability (XAI) to **hybrid quantum-classical machine learning (HQML)** models. Classical XAI tools (SHAP, LIME, permutation importance) assume a purely classical feature space; QuXAI addresses the gap by routing perturbations through the same quantum feature map the model used during training, ensuring that importance scores reflect what the quantum model actually sees.

---

## Architecture Overview

```
QuXAI/
├── src/                          # Core library
│   ├── config.py                 # Global constants (SEED, TRAIN_FRACTION, TEST_SIZE)
│   ├── data_processing.py        # CSV → sample → scale → split pipeline
│   ├── main.py                   # Demo entry point
│   ├── models/
│   │   ├── quantum_models.py     # 13 HQML model trainers
│   │   └── utils.py              # accuracy_score / classification_report helpers
│   ├── explainers/
│   │   ├── medley_explainer.py   # MEDLEY explainer class
│   │   └── utils.py              # Placeholder (empty)
│   └── plotting/
│       ├── visualizations.py     # Horizontal bar chart for MEDLEY scores
│       └── utils.py              # Seaborn style helper
└── experiments/
    ├── ablation.py               # 6-configuration DCI/PI ablation on 5 datasets
    ├── comparison.py             # Q-MEDLEY vs DCI, PI, TreeSHAP (Recall@K)
    ├── qmedley_res_iris.py       # Q-MEDLEY on 10 HQML models, Iris dataset
    ├── qmedley_res_wine.py       # Q-MEDLEY on 10 HQML models, Wine dataset
    ├── quxai_eval.py             # HQML vs Classical baselines (Accuracy/F1/Precision/Recall)
    └── rank_correlation.py       # Spearman ρ of explainers vs ground truth
```

---

## Quantum Feature Map

All models share a single **angle-encoding** feature map:

```python
def sample_quantum_feature_map(x, num_qubits):
    for i in range(num_qubits):
        qml.RX(x[i], wires=i)   # one RX gate per feature
```

This is a **product (non-entangled) encoding** — each feature maps independently to one qubit's rotation angle. The resulting quantum state is then measured as an amplitude vector (|ψ⟩) or used to compute a fidelity kernel K(x₁, x₂) = |⟨ψ₁|ψ₂⟩|².

---

## Supported HQML Models

| Short name | Classical classifier | Encoding |
|---|---|---|
| `qrf` | RandomForest | Amplitude |
| `qsvm` | SVC (precomputed) | Fidelity kernel |
| `qknn` | KNeighbors (precomputed) | Fidelity distance (1 − K) |
| `qlogistic` | LogisticRegression | Amplitude |
| `qdt` | DecisionTree | Amplitude |
| `qnb` | GaussianNB | Amplitude |
| `qada` | AdaBoost | Amplitude |
| `qgb` | GradientBoosting | Amplitude |
| `qlda` | LinearDiscriminantAnalysis | Amplitude |
| `qperceptron` | Perceptron | Amplitude |
| `qridge` | RidgeClassifier | Amplitude |
| `qsvc_poly` | SVC (precomputed, degree-3 poly) | Polynomial fidelity kernel |
| `qextra` | ExtraTrees | Amplitude |

**Amplitude models** convert each sample `x` to |ψ(x)⟩ via the feature map, then feed `|amplitude|` (or `|amplitude|²` in some experiments) to the classical classifier as a `2^n`-dimensional feature vector.

**Kernel models** compute an N×N fidelity matrix over training samples and pass it to `SVC(kernel='precomputed')` or `KNeighborsClassifier(metric='precomputed')`.

---

## MEDLEY Explainer

**MEDLEY** (Mixed Explainability through Drop-column and Label-permutation, Encoded via Quantum) lives in `src/explainers/medley_explainer.py`. For each feature `i` it computes:

1. **Drop-Column Importance (DCI):** set feature `i` = 0 on the full reference set, re-run prediction through the quantum encoding, measure accuracy drop vs baseline.
2. **Permutation Importance (PI):** shuffle feature `i` randomly (repeated `n_repeats` times), same measurement.
3. **Final score:** `(DCI_i + PI_i) / 2`

Because perturbations are passed through `_predict()` — which re-applies the quantum feature map — the scores reflect true quantum-encoded sensitivity rather than classical-space sensitivity.

**Ablation variants** (in `experiments/ablation.py`):

| Config | Description |
|---|---|
| Base (DCI only) | Uses only drop-column scores |
| Base (PI only) | Uses only permutation scores |
| Q-MEDLEY (DCI+PI Avg) | Equal-weight average (default) |
| Q-MEDLEY + AdaptiveWeighting | Dynamic 0.3/0.7 weights based on relative magnitudes |
| Q-MEDLEY with InteractionPI | Permutation that also perturbs the most-correlated feature |
| Q-MEDLEY + AdaptiveWeighting + InteractionPI | Combination of both extensions |

---

## Experiments

| Script | What it measures | Datasets |
|---|---|---|
| `ablation.py` | Recall@3 across 6 MEDLEY configurations | Iris, Wine, BreastCancer, Covtype, Diabetes |
| `comparison.py` | Recall@3 — Q-MEDLEY vs DCI, PI, TreeSHAP | Noisy Iris |
| `rank_correlation.py` | Spearman ρ — Q-MEDLEY vs DCI, PI, LogReg-L1 | Noisy Iris |
| `quxai_eval.py` | Accuracy / F1 / Precision / Recall — HQML vs Classical | Iris, Wine, BreastCancer |
| `qmedley_res_iris.py` | Q-MEDLEY bar charts for 10 amplitude HQML models | Noisy Iris |
| `qmedley_res_wine.py` | Q-MEDLEY bar charts for 10 amplitude HQML models | Noisy Wine |

Experiments use **noisy augmentation**: two random noise columns and one redundant column (linear combination of the highest-variance feature) are appended to each dataset to stress-test explainer faithfulness.

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `pennylane` | 0.41.1 | Quantum circuits and kernel utilities |
| `scikit-learn` | 1.6.1 | Classical classifiers, metrics, preprocessing |
| `numpy` | 2.2.5 | Numerical operations |
| `pandas` | 2.2.3 | Tabular data handling |
| `matplotlib` | 3.10.3 | Visualization |
| `shap` | *(unlisted)* | TreeSHAP baseline in `comparison.py` |
| `scipy` | *(unlisted)* | Spearman correlation in `rank_correlation.py` |
| `seaborn` | *(unlisted)* | Heatmap in `rank_correlation.py` |

---

## Known Issues

| # | Issue | Location |
|---|---|---|
| 1 | `shap`, `scipy`, `seaborn` used but absent from `requirements.txt` | `experiments/` |
| 2 | Stray module-level `_predict` function (dead code / leftover class method) | `src/main.py:149–172` |
| 3 | `plt.style.use('seaborn')` deprecated in Matplotlib ≥ 3.6 | `src/plotting/utils.py` |
| 4 | MEDLEY class duplicated in 4 experiment files with subtle inconsistencies | `experiments/*.py` |
| 5 | Amplitude encoding uses `np.abs(state)` in core but `np.abs(state)**2` in some experiments | mixed |
| 6 | Relative imports (`from config import …`) require running from `src/` directory | `src/data_processing.py`, `src/main.py` |
| 7 | No upper bound on `num_qubits` in `quantum_models.py`; high-dim datasets cause exponential memory use | `src/models/quantum_models.py` |
| 8 | `src/main.py` hardcodes `iris.csv` which no longer exists in the repo | `src/main.py:24` |
| 9 | `src/explainers/utils.py` is an empty stub | `src/explainers/utils.py` |
| 10 | No test suite anywhere in the project | — |
| 11 | No packaging configuration (`setup.py`, `pyproject.toml`) | — |
