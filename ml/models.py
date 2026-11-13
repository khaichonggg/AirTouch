"""The three candidate classifiers compared in the project (shared by training
and the research studies so both use identical settings)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("RandomForest", "SVM", "MLP")


def make_models(seed: int = 0) -> dict:
    """Fresh, unfitted estimators. All support predict_proba."""
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale",
                                                   probability=True, random_state=seed)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64),
                                                             max_iter=600, early_stopping=True,
                                                             random_state=seed)),
    }
