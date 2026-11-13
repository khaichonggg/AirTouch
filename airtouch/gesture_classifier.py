"""ML gesture recogniser: loads the model trained by ``ml/train_model.py``.

The saved bundle (joblib) is a dict:
    model          fitted scikit-learn estimator/pipeline with predict_proba
    name           "RandomForest" | "SVM" | "MLP"
    feature_flags  dict passed to features.feature_vector (translate/rotate/scale)
    aspect         camera aspect the features assume
    labels         list of class names
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import GESTURE_MODEL_PATH, Config
from .features import feature_vector
from .gestures_rule import RuleRecognizer


def _single_thread(model) -> None:
    """Parallel prediction (n_jobs=-1) costs ~45 ms per single sample because
    workers spin up on every call; one frame at a time is fastest on 1 thread."""
    parts = [model]
    if hasattr(model, "steps"):
        parts += [step for _, step in model.steps]
    for est in parts:
        if hasattr(est, "n_jobs"):
            try:
                est.n_jobs = 1
            except Exception:
                pass


class GestureClassifier:
    name = "ml"

    def __init__(self, path: str | Path = GESTURE_MODEL_PATH, threshold: float = 0.55,
                 aspect: float | None = None):
        import joblib

        bundle = joblib.load(Path(path))
        self.model = bundle["model"]
        self.model_name = bundle.get("name", type(self.model).__name__)
        self.flags = dict(bundle.get("feature_flags", {}))
        self.aspect = aspect if aspect is not None else bundle.get("aspect", 4 / 3)
        self.labels = [str(c) for c in self.model.classes_]
        self.threshold = threshold
        _single_thread(self.model)

    def predict(self, lm: np.ndarray) -> tuple[str, float]:
        x = feature_vector(lm, self.aspect, **self.flags)[None, :]
        proba = self.model.predict_proba(x)[0]      # predict_proba only: one pass, not two
        i = int(np.argmax(proba))
        conf = float(proba[i])
        if conf < self.threshold:
            return "none", conf
        return self.labels[i], conf


def load_recognizer(cfg: Config, path: str | Path = GESTURE_MODEL_PATH):
    """ML recogniser if requested and a trained model exists, else the rule engine.

    Returns (recogniser, message)."""
    if cfg.engine == "ml":
        try:
            if Path(path).exists():
                clf = GestureClassifier(path, cfg.ml_confidence, cfg.aspect)
                return clf, f"ML engine ({clf.model_name})"
        except Exception as e:  # unavailable, corrupt or incompatible optional model
            return RuleRecognizer(cfg.aspect, cfg.pinch_on), f"ML model failed to load ({e}); using rules"
        return RuleRecognizer(cfg.aspect, cfg.pinch_on), "no trained model yet; using rules"
    return RuleRecognizer(cfg.aspect, cfg.pinch_on), "rule engine"
