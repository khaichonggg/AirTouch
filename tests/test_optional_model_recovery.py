"""An unavailable optional ML model must not disable rule-based control."""
from pathlib import Path

import pytest

from airtouch import gesture_classifier
from airtouch.config import Config
from airtouch.gestures_rule import RuleRecognizer


@pytest.mark.parametrize("failure", [PermissionError("denied"), OSError("device unavailable")])
def test_optional_model_stat_failure_falls_back_and_next_attempt_recovers(tmp_path, monkeypatch, failure):
    path = tmp_path / "gesture_model.joblib"
    path.write_bytes(b"placeholder: never passed to joblib")
    original_exists = Path.exists
    unavailable = True

    def exists(self):
        if self == path and unavailable:
            raise failure
        return original_exists(self)

    restored = type("RestoredClassifier", (), {"name": "ml", "model_name": "restored"})()
    constructed = []

    def classifier(model_path, threshold, aspect):
        constructed.append((model_path, threshold, aspect))
        return restored

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(gesture_classifier, "GestureClassifier", classifier)
    cfg = Config(engine="ml", frame_width=800, frame_height=600, ml_confidence=0.7)
    recognizer, message = gesture_classifier.load_recognizer(cfg, path)
    assert isinstance(recognizer, RuleRecognizer)
    assert "using rules" in message
    assert constructed == []
    assert cfg.engine == "ml"  # the next attempt must still request ML

    unavailable = False
    recognizer, message = gesture_classifier.load_recognizer(cfg, path)
    assert recognizer is restored
    assert "restored" in message
    assert constructed == [(path, cfg.ml_confidence, cfg.aspect)]
