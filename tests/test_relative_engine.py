"""Relative-mode clutch safety through the real engine/controller pipeline.

Only recognition is scripted so debounce lag is deterministic; landmarks,
OneEuro smoothing, live pinch checks and all mouse events use the real code
with FakeBackend. No camera, UI or operating-system input is used.
"""

from __future__ import annotations

import numpy as np
import pytest

from airtouch.config import Config
from airtouch.controller import FakeBackend
from airtouch.engine import GestureEngine
from airtouch.landmarks import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, hand_size, to_metric
from ml.synthetic_data import HandPose, make_hand


class Recognizer:
    label = "point"

    def predict(self, _landmarks):
        return self.label, 1.0


def hand(label="point", dx=0.0, touching=True):
    curls = {
        "point": (0.05, 0.9, 0.9, 0.9),
        "fist": (0.9, 0.9, 0.9, 0.9),
        "open_palm": (0.05, 0.05, 0.05, 0.05),
        "pinch_index": (0.4, 0.7, 0.7, 0.7),
        "pinch_middle": (0.05, 0.45, 0.7, 0.7),
    }[label]
    landmarks = make_hand(label, np.random.default_rng(1),
                          pose=HandPose(cx=0.5, cy=0.55, size=0.16),
                          curls=dict(zip(("index", "middle", "ring", "pinky"), curls)),
                          thumb_gap=0.0, noise=0.0)
    if not touching:
        # Set a deterministic 2-D release gap; synthetic thumb_gap uses a 3-D
        # random direction, whose projection can still look like a held pinch.
        metric = to_metric(landmarks, 4 / 3)
        tip, other = ((INDEX_TIP, MIDDLE_TIP) if label == "pinch_index"
                      else (MIDDLE_TIP, INDEX_TIP))
        away = metric[tip, :2] - metric[other, :2]
        target = metric[tip, :2] + away / np.linalg.norm(away) * 0.6 * hand_size(metric)
        landmarks[THUMB_TIP, :2] = target[0] / (4 / 3), target[1]
    return landmarks + np.array([dx, 0.0, 0.0])


def primed_engine(pose="point", mode="relative"):
    backend = FakeBackend()
    recognizer = Recognizer()
    recognizer.label = pose
    engine = GestureEngine(Config(cursor_mode=mode), backend, recognizer)
    # Held-pinch tests start early enough to pass the deliberate drag delay;
    # all callers can still use .08/.09 as fresh times after fixture setup.
    times = ((-.30, -.27, -.24, 0.0, .03, .06)
             if mode == "relative" and pose.startswith("pinch_") else (0.0, .03, .06))
    for t in times:
        engine.process(hand(pose), t)
    backend.events.clear()
    return engine, backend, recognizer


def test_unknown_between_pinch_candidates_keeps_the_palm_stroke_moving():
    backend = FakeBackend()
    recognizer = Recognizer()
    engine = GestureEngine(Config(cursor_mode="relative"), backend, recognizer)
    for i in range(12):
        recognizer.label = "pinch_index" if i % 2 == 0 else "none"
        engine.process(hand("pinch_index", dx=.005 * i), i / 30)
    assert backend.pos[0] > 1000
    assert backend.count("move") > 5
    assert backend.count("down") == 0


def test_unknown_continuation_cannot_fire_a_dwell_click():
    backend = FakeBackend()
    recognizer = Recognizer()
    engine = GestureEngine(Config(cursor_mode="relative", dwell_enabled=True, dwell_ms=200),
                           backend, recognizer)
    for i in range(7):
        engine.process(hand(), i * .03)
    recognizer.label = "none"
    result = engine.process(hand(), .21)
    assert result.label == "point"
    assert backend.count("down") == 0
    recognizer.label = "point"
    engine.process(hand(), .24)
    assert backend.count("down", "left") == 1
    assert backend.down == set()


@pytest.mark.parametrize("pose, button", [("point", None), ("open_palm", None), ("pinch_index", "left"),
                                         ("pinch_middle", "right")])
def test_relative_short_hand_loss_reanchors_without_releasing_a_held_button(pose, button):
    engine, backend, _ = primed_engine(pose)
    assert engine.controller.button == button

    lost = engine.process(None, 0.08)
    assert lost.button == button and lost.events == []
    backend.pos = (230, 340)             # actual mouse may move during the gap
    returned = engine.process(hand(pose, dx=0.25), 0.10)

    assert returned.cursor == (230, 340)
    assert returned.button == button
    assert backend.events == []
    engine.process(hand(pose, dx=0.30), 0.13)
    assert backend.pos[0] > 230


@pytest.mark.parametrize("pose, button", [("pinch_index", "left"), ("pinch_middle", "right")])
def test_relative_hand_loss_keeps_the_existing_release_deadline(pose, button):
    engine, backend, _ = primed_engine(pose)
    engine.process(None, 0.08)
    engine.process(None, 0.22)
    assert backend.down == {button}

    released = engine.process(None, 0.24)
    again = engine.process(None, 0.30)

    assert released.events == [f"up:{button}"]
    assert again.events == []
    assert backend.down == set()


def test_relative_open_hand_follows_the_index_tip_without_pressing():
    backend = FakeBackend()
    recognizer = Recognizer()
    recognizer.label = "open_palm"
    engine = GestureEngine(Config(cursor_mode="relative"), backend, recognizer)
    landmarks = hand("open_palm")

    anchored = engine.process(landmarks, 0.0)
    landmarks[INDEX_TIP, 0] += .025   # the palm and all other landmarks stay still
    moved = engine.process(landmarks, 1 / 30)

    assert anchored.label == moved.label == "point"
    assert moved.raw_label == "open_palm"
    assert moved.cursor[0] > anchored.cursor[0]
    assert backend.count("down") == 0
    assert backend.down == set()


def test_real_open_hand_recognition_moves_in_relative_mode():
    backend = FakeBackend()
    engine = GestureEngine(Config(cursor_mode="relative"), backend)
    landmarks = hand("open_palm")

    first = engine.process(landmarks, 0.0)
    landmarks[INDEX_TIP, 0] += .025
    moved = engine.process(landmarks, 1 / 30)

    assert first.raw_label == moved.raw_label == "open_palm"
    assert first.label == moved.label == "point"
    assert moved.cursor[0] > first.cursor[0]
    assert backend.down == set()


def test_relative_point_and_open_hand_share_one_continuous_tip_stroke():
    reference, expected, _ = primed_engine()
    engine, backend, recognizer = primed_engine()

    for frame in range(1, 13):
        landmarks = hand(dx=frame * .005)
        t = .06 + frame / 30
        reference.process(landmarks, t)
        recognizer.label = "open_palm" if frame % 2 else "point"
        result = engine.process(landmarks, t)
        assert result.label == "point"

    assert backend.pos == expected.pos
    assert backend.count("move") == expected.count("move")
    assert backend.count("down") == 0


def test_relative_open_hand_has_the_same_bounded_unknown_continuation_as_point():
    engine, backend, recognizer = primed_engine("open_palm")
    recognizer.label = "none"

    continued = engine.process(hand("open_palm", dx=.02), .10)
    before = backend.pos
    expired = engine.process(hand("open_palm", dx=.10), .181)

    assert continued.label == "point"
    assert continued.cursor[0] > 960
    assert expired.label == "none"
    assert backend.pos == before
    assert backend.count("down") == 0


@pytest.mark.parametrize("raw", ["fist"])
@pytest.mark.parametrize("resume", ["point", "open_palm"])
def test_relative_raw_clutch_stops_before_debounce_and_resumes_without_a_jump(raw, resume):
    engine, backend, recognizer = primed_engine()
    before = backend.pos
    recognizer.label = raw

    paused = engine.process(hand(raw if raw != "none" else "point", dx=0.25), 0.09)

    assert paused.cursor == before
    assert paused.label == raw                  # also drives the halo's active state
    assert backend.events == []
    assert engine.debouncer.current == "point"  # the debouncer itself is unchanged
    recognizer.label = resume
    resumed = engine.process(hand(resume, dx=0.25), 0.12)
    assert resumed.cursor == before
    assert backend.events == []
    engine.process(hand(resume, dx=0.30), 0.15)
    assert backend.pos[0] > before[0]


@pytest.mark.parametrize("raw", ["fist"])
def test_relative_raw_clutch_does_not_start_a_new_pinch(raw):
    engine, backend, recognizer = primed_engine()
    before = backend.pos
    recognizer.label = raw

    result = engine.process(hand("pinch_index", dx=0.25), 0.09)

    assert result.button is None
    assert backend.pos == before
    assert backend.events == []


def test_relative_alternating_unknown_frames_keep_the_same_continuous_motion():
    reference, expected, _ = primed_engine()
    engine, backend, recognizer = primed_engine()

    for frame in range(1, 13):
        landmarks = hand(dx=frame * 0.005)
        t = 0.06 + frame / 30
        reference.process(landmarks, t)
        recognizer.label = "none" if frame % 2 else "point"
        result = engine.process(landmarks, t)
        assert result.label == "point"

    assert backend.pos == expected.pos
    assert backend.pos[0] > 1060
    assert backend.count("down") == 0


def test_relative_unknown_continuity_expires_by_time_and_reanchors():
    engine, backend, recognizer = primed_engine()
    recognizer.label = "none"

    for t, dx in ((0.10, 0.01), (0.14, 0.02)):
        result = engine.process(hand(dx=dx), t)
        assert result.label == "point"
    before = backend.pos
    backend.events.clear()

    expired = engine.process(hand(dx=0.25), 0.181)

    assert expired.label == "none"
    assert backend.pos == before
    assert backend.events == []
    recognizer.label = "point"
    resumed = engine.process(hand(dx=0.25), 0.20)
    assert resumed.cursor == before
    assert backend.events == []
    engine.process(hand(dx=0.27), 0.23)
    assert backend.pos[0] > before[0]


@pytest.mark.parametrize("clutch", ["fist"])
def test_relative_point_resumes_before_the_stale_clutch_debouncer(clutch):
    engine, backend, recognizer = primed_engine(clutch)
    before = backend.pos
    recognizer.label = "point"

    anchored = engine.process(hand(dx=0.25), 0.09)
    assert anchored.label == "point"
    assert backend.events == []
    assert engine.debouncer.current == clutch
    moved = engine.process(hand(dx=0.27), 0.12)

    assert moved.cursor[0] > before[0]
    assert backend.count("move") == 1


@pytest.mark.parametrize("pose, button", [("pinch_index", "left"), ("pinch_middle", "right")])
@pytest.mark.parametrize("idle", ["none", "fist", "open_palm"])
def test_relative_two_matching_pinch_frames_start_pending_then_hold_from_idle(idle, pose, button):
    engine, backend, recognizer = primed_engine(idle if idle != "none" else "fist")
    if idle == "none":
        engine.reset()
    recognizer.label = pose

    candidate = engine.process(hand(pose), 0.09)
    assert candidate.label == pose
    assert candidate.button is None
    assert backend.count("down") == 0
    confirmed = engine.process(hand(pose, dx=0.01), 0.12)

    assert confirmed.button is None
    assert backend.down == set()
    assert backend.count("down") == 0
    assert engine.debouncer.current == idle
    held = engine.process(hand(pose, dx=.01), .38)
    assert held.button == button
    assert backend.down == {button}
    assert backend.count("down") == 1


@pytest.mark.parametrize("pose", ["pinch_index", "pinch_middle"])
def test_relative_single_pinch_candidate_does_not_press(pose):
    engine, backend, recognizer = primed_engine()
    recognizer.label = pose

    candidate = engine.process(hand(pose), 0.09)
    recognizer.label = "point"
    engine.process(hand(), 0.12)

    assert candidate.label == pose
    assert candidate.button is None
    assert backend.count("down") == 0


@pytest.mark.parametrize("interrupt", ["none", "fist", "open_palm", "point", "hand_loss", "reset"])
def test_relative_pinch_confirmation_requires_consecutive_frames(interrupt):
    engine, backend, recognizer = primed_engine()
    recognizer.label = "pinch_index"
    engine.process(hand("pinch_index"), 0.09)
    if interrupt == "hand_loss":
        engine.process(None, 0.12)
    elif interrupt == "reset":
        engine.reset()
    else:
        recognizer.label = interrupt
        engine.process(hand(interrupt if interrupt != "none" else "point"), 0.12)
    recognizer.label = "pinch_index"

    result = engine.process(hand("pinch_index"), 0.15)

    assert result.button is None
    assert backend.count("down") == 0


def test_relative_different_pinch_candidates_do_not_confirm_each_other():
    engine, backend, recognizer = primed_engine()
    recognizer.label = "pinch_index"
    engine.process(hand("pinch_index"), 0.09)
    recognizer.label = "pinch_middle"

    candidate = engine.process(hand("pinch_middle"), 0.12)
    assert candidate.button is None
    assert backend.count("down") == 0
    confirmed = engine.process(hand("pinch_middle"), 0.15)
    assert confirmed.button is None
    assert backend.down == set()
    held = engine.process(hand("pinch_middle"), .41)
    assert held.button == "right"
    assert backend.down == {"right"}


def test_relative_unknown_continuity_never_starts_a_new_pinch():
    engine, backend, recognizer = primed_engine()
    recognizer.label = "none"

    result = engine.process(hand("pinch_index", dx=0.01), 0.09)

    assert result.label == "point"
    assert result.button is None
    assert backend.count("down") == 0


@pytest.mark.parametrize("raw", ["point", "open_palm"])
def test_relative_point_with_contact_geometry_cannot_bypass_pinch_confirmation(raw):
    engine, backend, recognizer = primed_engine()
    recognizer.label = raw

    result = engine.process(hand("pinch_index"), 0.09)

    assert result.button is None
    assert backend.count("down") == 0


def test_absolute_contact_still_presses_without_waiting_for_raw_pinch_confirmation():
    engine, backend, _ = primed_engine(mode="absolute")

    result = engine.process(hand("pinch_index"), 0.09)

    assert result.button == "left"
    assert backend.count("down") == 1


def test_absolute_open_hand_still_pauses_after_its_existing_debounce():
    engine, backend, recognizer = primed_engine(mode="absolute")
    recognizer.label = "open_palm"
    for t, dx in ((.09, .05), (.12, .10), (.15, .15)):
        result = engine.process(hand("open_palm", dx=dx), t)
    before = backend.pos
    backend.events.clear()

    result = engine.process(hand("open_palm", dx=.25), .18)

    assert result.label == "open_palm"
    assert backend.pos == before
    assert backend.events == []


@pytest.mark.parametrize("raw", ["fist", "open_palm", "none"])
def test_absolute_raw_clutch_keeps_its_existing_debounced_movement(raw):
    engine, backend, recognizer = primed_engine(mode="absolute")
    before = backend.pos
    recognizer.label = raw

    result = engine.process(hand(raw if raw != "none" else "point", dx=0.25), 0.09)

    assert result.label == "point"
    assert backend.pos[0] > before[0]


@pytest.mark.parametrize("raw", ["fist", "open_palm", "none"])
@pytest.mark.parametrize("pose, button", [("pinch_index", "left"), ("pinch_middle", "right")])
def test_relative_valid_held_pinch_still_drags_when_the_raw_label_wobbles(raw, pose, button):
    engine, backend, recognizer = primed_engine(pose)
    before = backend.pos
    recognizer.label = raw

    result = engine.process(hand(pose, dx=0.1), 0.09)

    assert result.button == button
    assert backend.down == {button}
    assert backend.pos[0] > before[0]
    assert backend.count("down") == 0
    assert backend.count("up") == 0


@pytest.mark.parametrize("raw", ["fist", "open_palm", "none"])
def test_relative_released_pinch_stops_on_the_same_raw_clutch_frame(raw):
    engine, backend, recognizer = primed_engine("pinch_index")
    before = backend.pos
    recognizer.label = raw

    result = engine.process(hand("pinch_index", dx=0.25, touching=False), 0.09)

    assert result.button is None
    assert backend.down == set()
    assert result.events == ["up:left"]
    assert backend.pos == before
