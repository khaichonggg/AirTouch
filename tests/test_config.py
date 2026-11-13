"""Config: a bad edit of config.json must never break the app.

Every value ends up inside its safe range, and loading can only ever produce a
usable Config - unknown keys, corrupt files and missing files included.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields

import pytest

import airtouch.config as config_module
from airtouch.config import _RANGES, Config

INT_FIELDS = sorted(f.name for f in fields(Config) if f.name in _RANGES and type(f.default) is int)
FLOAT_FIELDS = sorted(name for name in _RANGES if name not in INT_FIELDS)
BOOL_FIELDS = sorted(f.name for f in fields(Config) if type(f.default) is bool)

UNUSABLE = [float("nan"), float("inf"), float("-inf"), "abc", "", None, [], {}, object()]
UNUSABLE_IDS = ["nan", "inf", "-inf", "text", "empty-text", "none", "list", "dict", "object"]


def write(path, content):
    """Write ``content`` (text or raw bytes) and return the path."""
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


# ------------------------------------------------------------------- defaults
def test_defaults_match_the_documented_settings():
    cfg = Config()
    assert (cfg.camera_index, cfg.frame_width, cfg.frame_height, cfg.camera_fps) == (0, 640, 480, 30)
    assert cfg.mirror is True and cfg.show_hud is True and cfg.dwell_enabled is False
    assert cfg.max_hands == 1
    assert (cfg.min_detection_confidence, cfg.min_tracking_confidence) == (0.6, 0.5)
    assert (cfg.frame_margin, cfg.smoothing, cfg.beta, cfg.d_cutoff) == (0.15, 1.2, 1.0, 1.0)
    assert (cfg.pinch_on, cfg.pinch_off, cfg.debounce_frames) == (0.20, 0.32, 3)
    assert (cfg.scroll_gain, cfg.scroll_deadzone, cfg.scroll_smoothing) == (40, 0.010, 0.4)
    assert (cfg.zoom_gain, cfg.zoom_deadzone) == (6, 0.08)
    assert (cfg.dwell_ms, cfg.dwell_radius_px) == (900, 18)
    assert (cfg.engine, cfg.ml_confidence) == ("rule", 0.55)


@pytest.mark.parametrize("name", sorted(_RANGES))
def test_every_default_is_inside_its_own_range(name):
    lo, hi = _RANGES[name]
    assert lo <= getattr(Config(), name) <= hi


# ------------------------------------------------------------------- clamping
@pytest.mark.parametrize("name", sorted(_RANGES))
def test_far_too_large_values_are_clamped_to_the_maximum(name):
    assert getattr(Config(**{name: 1e9}), name) == _RANGES[name][1]


@pytest.mark.parametrize("name", sorted(_RANGES))
def test_far_too_small_values_end_up_in_range(name):
    lo, hi = _RANGES[name]
    value = getattr(Config(**{name: -1e9}), name)
    assert lo <= value <= hi
    if name != "pinch_off":      # pinch_off is then pushed above pinch_on (tested below)
        assert value == lo


@pytest.mark.parametrize("name", sorted(_RANGES))
def test_values_already_in_range_are_kept(name):
    lo, hi = _RANGES[name]
    mid = (lo + hi) / 2
    value = getattr(Config(**{name: mid}), name)
    if name in INT_FIELDS:
        mid = round(mid)
    if name == "pinch_on":       # pinch_off is pushed up, pinch_on itself stays
        assert value == pytest.approx(mid)
    else:
        assert value == pytest.approx(mid)


@pytest.mark.parametrize("bad", UNUSABLE, ids=UNUSABLE_IDS)
def test_unusable_values_fall_back_to_the_default(bad):
    default = Config()
    for name in sorted(_RANGES):
        assert getattr(Config(**{name: bad}), name) == getattr(default, name), name


def test_numeric_text_is_accepted():
    assert Config(frame_width="800").frame_width == 800
    assert Config(smoothing="2.5").smoothing == 2.5
    assert Config(pinch_on=" 0.25 ").pinch_on == 0.25


def test_relative_cursor_settings_round_trip(tmp_path):
    cfg = Config(cursor_mode="relative", cursor_gain=2.0, cursor_halo=True)
    path = tmp_path / "relative.json"
    cfg.save(path)
    restored = Config.load(path)
    assert (restored.cursor_mode, restored.cursor_gain, restored.cursor_halo) == ("relative", 2.0, True)


@pytest.mark.parametrize("mode", [None, "invalid", [], 4])
def test_invalid_cursor_mode_keeps_absolute_compatibility(mode):
    assert Config(cursor_mode=mode).cursor_mode == "absolute"


@pytest.mark.parametrize("name", INT_FIELDS)
def test_int_fields_stay_int_whatever_they_are_given(name):
    lo, hi = _RANGES[name]
    for raw in (lo + 0.6, float(hi), str(hi), 1e9, -1e9, float("nan"), "abc", None, True):
        value = getattr(Config(**{name: raw}), name)
        assert type(value) is int, (raw, value)
        assert lo <= value <= hi


def test_int_fields_round_to_the_nearest_whole_number():
    assert Config(frame_width=640.7).frame_width == 641
    assert Config(debounce_frames=2.2).debounce_frames == 2
    assert Config(dwell_ms="900.6").dwell_ms == 901


@pytest.mark.parametrize("name", FLOAT_FIELDS)
def test_float_fields_stay_float(name):
    lo, hi = _RANGES[name]
    for raw in (int(hi), str(hi), 1e9, -1e9, float("nan"), None):
        assert type(getattr(Config(**{name: raw}), name)) is float, raw


# ---------------------------------------------------------------- other fields
@pytest.mark.parametrize("name", BOOL_FIELDS)
def test_bool_fields_become_real_bools(name):
    for raw, expected in ((True, True), (False, False), (1, True), (0, False), (None, False), ("", False)):
        assert getattr(Config(**{name: raw}), name) is expected, raw


@pytest.mark.parametrize("engine", ["rule", "ml"])
def test_valid_engines_are_kept(engine):
    assert Config(engine=engine).engine == engine


@pytest.mark.parametrize("engine", ["", "RULE", "tensorflow", None, 5, ["ml"], {"ml": 1}])
def test_invalid_engine_falls_back_to_rules(engine):
    assert Config(engine=engine).engine == "rule"


# ----------------------------------------------------------------- pinch order
@pytest.mark.parametrize("on, off", [(0.3, 0.3), (0.3, 0.1), (0.6, 0.06), (0.6, 0.6), (0.05, 0.06 - 1)])
def test_release_threshold_is_always_wider_than_the_press_threshold(on, off):
    cfg = Config(pinch_on=on, pinch_off=off)
    assert cfg.pinch_off > cfg.pinch_on
    lo, hi = _RANGES["pinch_off"]
    assert lo <= cfg.pinch_off <= hi


def test_inverted_pinch_thresholds_are_repaired_with_the_standard_margin():
    cfg = Config(pinch_on=0.3, pinch_off=0.2)
    assert cfg.pinch_on == 0.3
    assert cfg.pinch_off == pytest.approx(0.38)


def test_valid_pinch_thresholds_are_left_alone():
    cfg = Config(pinch_on=0.25, pinch_off=0.5)
    assert (cfg.pinch_on, cfg.pinch_off) == (0.25, 0.5)


def test_unusable_pinch_off_is_still_repaired_against_pinch_on():
    cfg = Config(pinch_on=0.5, pinch_off=float("nan"))     # default 0.32 would be below 0.5
    assert cfg.pinch_off > cfg.pinch_on


# ---------------------------------------------------------------------- aspect
@pytest.mark.parametrize(
    "width, height, expected",
    [(640, 480, 4 / 3), (1280, 720, 16 / 9), (1920, 1080, 16 / 9), (480, 640, 3 / 4)],
)
def test_aspect_is_width_over_height(width, height, expected):
    assert Config(frame_width=width, frame_height=height).aspect == pytest.approx(expected)


def test_aspect_can_never_divide_by_zero():
    assert Config(frame_width=0, frame_height=0).aspect == pytest.approx(160 / 120)


# --------------------------------------------------------------------- loading
def test_load_of_a_missing_file_gives_defaults(tmp_path):
    assert Config.load(tmp_path / "does_not_exist.json") == Config()


@pytest.mark.parametrize(
    "content",
    ["", "   ", "{not json", '{"frame_width": 800', "{'frame_width': 800}", b"\xff\xfe\x00{", b"\x00\x01\x02"],
    ids=["empty", "blank", "garbage", "truncated", "single-quotes", "bad-utf8", "binary"],
)
def test_load_of_a_corrupt_file_gives_defaults(tmp_path, content):
    assert Config.load(write(tmp_path / "config.json", content)) == Config()


def test_load_of_a_folder_gives_defaults(tmp_path):
    assert Config.load(tmp_path) == Config()


def test_load_ignores_unknown_keys(tmp_path):
    path = write(tmp_path / "config.json", json.dumps({"frame_width": 800, "wat": 1, "__class__": "x"}))
    cfg = Config.load(path)
    assert cfg.frame_width == 800
    assert not hasattr(cfg, "wat")


def test_load_keeps_defaults_for_keys_not_in_the_file(tmp_path):
    cfg = Config.load(write(tmp_path / "config.json", json.dumps({"beta": 2.0})))
    assert cfg.beta == 2.0
    assert cfg.smoothing == Config().smoothing


def test_load_makes_bad_values_safe(tmp_path):
    bad = {
        "frame_width": -5,
        "frame_height": 99999,
        "smoothing": "fast",
        "beta": None,
        "engine": 7,
        "mirror": 0,
        "pinch_on": 0.3,
        "pinch_off": 0.1,
    }
    cfg = Config.load(write(tmp_path / "config.json", json.dumps(bad)))
    assert cfg.frame_width == 160
    assert cfg.frame_height == 1080
    assert cfg.smoothing == Config().smoothing
    assert cfg.beta == Config().beta
    assert cfg.engine == "rule"
    assert cfg.mirror is False
    assert cfg.pinch_off > cfg.pinch_on


def test_load_makes_json_nan_and_infinity_safe(tmp_path):
    # Python's json module accepts these non-standard literals
    text = '{"smoothing": NaN, "beta": Infinity, "d_cutoff": -Infinity}'
    cfg = Config.load(write(tmp_path / "config.json", text))
    default = Config()
    assert (cfg.smoothing, cfg.beta, cfg.d_cutoff) == (default.smoothing, default.beta, default.d_cutoff)


def test_load_accepts_text_and_path_objects(tmp_path):
    path = write(tmp_path / "config.json", '{"frame_width": 800}')
    assert Config.load(str(path)).frame_width == 800
    assert Config.load(path).frame_width == 800


@pytest.mark.parametrize("text", ["[1, 2, 3]", "null", "42", '"abc"', "true"])
def test_load_of_json_that_is_not_an_object_gives_defaults(tmp_path, text):
    assert Config.load(write(tmp_path / "config.json", text)) == Config()


def test_absurdly_large_integer_is_clamped_or_defaulted():
    assert Config(frame_width=10 ** 400).frame_width in (Config().frame_width, _RANGES["frame_width"][1])


def test_load_survives_an_absurdly_large_integer(tmp_path):
    cfg = Config.load(write(tmp_path / "config.json", '{"frame_width": 1' + "0" * 400 + "}"))
    assert _RANGES["frame_width"][0] <= cfg.frame_width <= _RANGES["frame_width"][1]


def test_load_does_not_crash_on_a_byte_order_mark(tmp_path):
    cfg = Config.load(write(tmp_path / "config.json", b"\xef\xbb\xbf" + b'{"frame_width": 800}'))
    assert isinstance(cfg, Config)


def test_load_honours_settings_in_a_file_with_a_byte_order_mark(tmp_path):
    cfg = Config.load(write(tmp_path / "config.json", b"\xef\xbb\xbf" + b'{"frame_width": 800}'))
    assert cfg.frame_width == 800


@pytest.mark.parametrize("name", BOOL_FIELDS)
@pytest.mark.parametrize("text", ["false", " FALSE ", "0", "off", "no"])
def test_false_boolean_text_does_not_enable_a_setting(name, text):
    assert getattr(Config(**{name: text}), name) is False


@pytest.mark.parametrize("name", BOOL_FIELDS)
def test_unrecognized_boolean_text_uses_the_field_default(name):
    assert getattr(Config(**{name: "invalid"}), name) is getattr(Config(), name)


# ---------------------------------------------------------------------- saving
def test_save_then_load_round_trip(tmp_path):
    original = Config(
        camera_index=2,
        frame_width=1280,
        frame_height=720,
        camera_fps=60,
        mirror=False,
        max_hands=2,
        frame_margin=0.2,
        smoothing=0.7,
        beta=3.25,
        pinch_on=0.17,
        pinch_off=0.41,
        debounce_frames=5,
        scroll_gain=75.5,
        dwell_enabled=True,
        dwell_ms=1200,
        engine="ml",
        ml_confidence=0.8,
        show_hud=False,
    )
    path = tmp_path / "config.json"
    original.save(path)
    assert Config.load(path) == original


def test_save_writes_every_field_as_readable_json(tmp_path):
    path = tmp_path / "config.json"
    Config().save(path)
    assert json.loads(path.read_text(encoding="utf-8")) == asdict(Config())


def test_save_and_load_work_in_a_non_ascii_folder(tmp_path):
    folder = tmp_path / "Bachelor in Computer Science 【 Final Year Project 】"
    folder.mkdir()
    Config(frame_width=800).save(folder / "config.json")
    assert Config.load(folder / "config.json").frame_width == 800


def test_load_and_save_use_the_default_path_when_none_is_given(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", path)
    assert Config.load() == Config()
    Config(frame_width=800).save()
    assert path.exists()
    assert Config.load().frame_width == 800
