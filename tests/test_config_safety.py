"""Malformed JSON flags must not enable dwell clicks or native overlays."""
import json

import numpy as np
import pytest

from airtouch.config import Config
from airtouch.controller import FakeBackend
from airtouch.engine import GestureEngine
from ml.synthetic_data import HandPose, make_hand


@pytest.mark.parametrize("name, expected", [
    ("mirror", True), ("show_hud", True),
    ("dwell_enabled", False), ("cursor_halo", False),
])
@pytest.mark.parametrize("invalid", [[], [0], {}, {"enabled": False},
                                    float("nan"), float("inf"), float("-inf"), 2, -1, .5],
                         ids=["empty-list", "nonempty-list", "empty-object", "object", "nan", "inf", "negative-inf",
                              "two", "negative-one", "fraction"])
def test_malformed_boolean_flags_use_safe_field_defaults(tmp_path, name, expected, invalid):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({name: invalid}), encoding="utf-8")
    assert getattr(Config.load(path), name) is expected


@pytest.mark.parametrize("invalid", [[0], {"enabled": False}, float("nan"), float("inf"), 2, -1, .5],
                         ids=["list", "object", "nan", "inf", "two", "negative-one", "fraction"])
def test_malformed_dwell_flag_cannot_click_a_stationary_target(tmp_path, invalid):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"dwell_enabled": invalid, "cursor_mode": "relative"}), encoding="utf-8")

    class PointRecognizer:
        def predict(self, _hand):
            return "point", 1.0

    backend = FakeBackend()
    engine = GestureEngine(Config.load(path), backend, PointRecognizer())
    hand = make_hand("point", np.random.default_rng(7), pose=HandPose(), noise=0.0)
    for frame in range(40):
        engine.process(hand, frame / 30)

    assert backend.count("down") == 0, backend.events
    assert backend.down == set()


@pytest.mark.parametrize("name", ["mirror", "show_hud", "dwell_enabled", "cursor_halo"])
@pytest.mark.parametrize("value, expected", [(0.0, False), (1.0, True)])
def test_canonical_numeric_boolean_flags_remain_supported(tmp_path, name, value, expected):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({name: value}), encoding="utf-8")
    assert getattr(Config.load(path), name) is expected


@pytest.mark.parametrize("opening, closing", [("[", "]"), ('{"nested":', "}")],
                         ids=["arrays", "objects"])
def test_excessive_json_nesting_recovers_without_preventing_startup(tmp_path, opening, closing):
    path = tmp_path / "config.json"
    path.write_text('{"unknown":' + opening * 5000 + '0' + closing * 5000 + '}', encoding="utf-8")
    assert Config.load(path) == Config()
