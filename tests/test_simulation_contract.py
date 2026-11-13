"""Simulator contracts use FakeBackend only: no camera or system mouse."""

from pathlib import Path

import pytest

import simulate
from airtouch.gesture_classifier import GestureClassifier


class TimedPinchRecognizer:
    """Recognition can finish before or after contact at frame 18."""

    def __init__(self, button, start=16, label=None):
        self.expected = label or simulate.PINCH_GESTURE[button]
        self.start = start
        self.frame = 0

    def predict(self, _landmarks):
        frame = self.frame
        self.frame += 1
        return (self.expected if self.start <= frame <= 27 else "point"), 1.0


def click_context(button, recognizer):
    context = simulate.Ctx(f"contract_{button}", recognizer)
    simulate._click_scenario(context, button)
    return context


def failures(context):
    return [(check.what, check.info) for check in context.checks if not check.ok]


@pytest.mark.parametrize("button", ["left", "right"])
@pytest.mark.parametrize("start, expected_down_label", [(16, "pinch"), (18, "point")])
def test_click_accepts_recognition_before_or_after_physical_contact(button, start, expected_down_label):
    context = click_context(button, TimedPinchRecognizer(button, start))
    session = context.sessions[0]
    down, = session.event_frames(f"down:{button}")
    up, = session.event_frames(f"up:{button}")
    expected = simulate.PINCH_GESTURE[button] if expected_down_label == "pinch" else "point"

    assert session.results[down].label == expected
    assert down == session.first_touch(button) == 18
    assert up == 28
    assert session.backend.down == set()
    assert failures(context) == []


@pytest.mark.parametrize("button", ["left", "right"])
def test_real_ml_click_scenario_checks_actions_and_recognition(button):
    if not Path(simulate.GESTURE_MODEL_PATH).is_file():
        pytest.skip("No trained model installed")
    context = click_context(button, GestureClassifier())
    session = context.sessions[0]
    down, = session.event_frames(f"down:{button}")

    assert down == session.first_touch(button)
    assert failures(context) == []


@pytest.mark.parametrize("button", ["left", "right"])
def test_click_scenario_rejects_a_recognizer_that_never_recognizes_the_pinch(button):
    context = click_context(button, TimedPinchRecognizer(button, start=1000))
    session = context.sessions[0]

    assert session.event_frames(f"down:{button}") == [18]
    assert session.event_frames(f"up:{button}") == [28]
    assert failures(context), "Correct mouse geometry alone cannot pass the recognition scenario"


@pytest.mark.parametrize("button", ["left", "right"])
def test_click_scenario_rejects_the_other_pinch_label(button):
    other = "right" if button == "left" else "left"
    context = click_context(button, TimedPinchRecognizer(button, label=simulate.PINCH_GESTURE[other]))

    assert failures(context)


@pytest.mark.parametrize("button", ["left", "right"])
def test_click_scenario_rejects_a_down_before_the_geometric_contact(button, monkeypatch):
    original = simulate.Session.first_touch

    def later_contact(self, pressed_button, start=0):
        actual = original(self, pressed_button, start)
        return actual + 1 if actual is not None else None

    monkeypatch.setattr(simulate.Session, "first_touch", later_contact)
    context = click_context(button, TimedPinchRecognizer(button, start=18))

    assert failures(context), "A negative contact-to-down delay is an early false press"
