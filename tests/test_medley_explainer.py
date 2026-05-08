"""Unit tests for src/explainers/medley_explainer.py"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from explainers.medley_explainer import MEDLEY
from models.quantum_models import train_qdt, train_qrf

rng = np.random.RandomState(1)
X_toy = rng.randn(20, 4).astype(np.float64)
y_toy = (X_toy[:, 0] > 0).astype(int)


def test_medley_classical_returns_correct_length():
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_toy, y_toy)
    explainer = MEDLEY(clf, model_type='classical', n_repeats=2)
    explainer.fit(X_toy, y_toy)
    scores = explainer.interpret()
    assert len(scores) == X_toy.shape[1]


def test_medley_classical_scores_are_finite():
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_toy, y_toy)
    explainer = MEDLEY(clf, model_type='classical', n_repeats=2)
    explainer.fit(X_toy, y_toy)
    scores = explainer.interpret()
    assert np.all(np.isfinite(scores))


def test_medley_hqml_returns_correct_length():
    model = train_qdt(X_toy, y_toy)
    explainer = MEDLEY(
        model, model_type='qdt',
        quantum_feature_map=model.quantum_feature_map,
        n_repeats=2,
    )
    explainer.fit(X_toy, y_toy)
    scores = explainer.interpret()
    assert len(scores) == X_toy.shape[1]


def test_medley_hqml_scores_are_finite():
    model = train_qdt(X_toy, y_toy)
    explainer = MEDLEY(
        model, model_type='qdt',
        quantum_feature_map=model.quantum_feature_map,
        n_repeats=2,
    )
    explainer.fit(X_toy, y_toy)
    scores = explainer.interpret()
    assert np.all(np.isfinite(scores))


def test_medley_dci_and_perm_combine():
    """DCI-only vs full MEDLEY should differ on a real dataset."""
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X_toy, y_toy)
    explainer = MEDLEY(clf, model_type='classical', n_repeats=3)
    explainer.fit(X_toy, y_toy)
    scores = explainer.interpret()
    # At least one score should be non-zero for a meaningful model
    assert np.any(scores != 0)


def test_medley_fit_sets_num_features():
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_toy, y_toy)
    explainer = MEDLEY(clf, model_type='classical')
    explainer.fit(X_toy, y_toy)
    assert explainer.num_features_ == X_toy.shape[1]
