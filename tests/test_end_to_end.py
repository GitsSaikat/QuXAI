"""End-to-end smoke test: train → explain → check scores."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from models.quantum_models import train_qdt
from explainers.medley_explainer import MEDLEY

# Minimal synthetic dataset: 4 features, 20 samples, 2 classes
rng = np.random.RandomState(99)
X = rng.randn(20, 4).astype(np.float64)
y = (X[:, 0] > 0).astype(int)


def test_end_to_end_train_and_explain():
    model = train_qdt(X, y)
    explainer = MEDLEY(
        pretrained_model=model,
        model_type='qdt',
        quantum_feature_map=model.quantum_feature_map,
        n_repeats=2,
    )
    explainer.fit(X, y)
    scores = explainer.interpret()

    assert scores is not None
    assert len(scores) == X.shape[1], "Score vector length must equal number of features"
    assert np.all(np.isfinite(scores)), "All importance scores must be finite"


def test_end_to_end_classical_model():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    explainer = MEDLEY(clf, model_type='classical', n_repeats=2)
    explainer.fit(X, y)
    scores = explainer.interpret()

    assert len(scores) == X.shape[1]
    assert np.all(np.isfinite(scores))
