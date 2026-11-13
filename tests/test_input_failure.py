"""Permission/unavailable input responses through API stubs; no OS calls."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from airtouch.config import Config
from airtouch.controller import MouseController, Win32Backend
from ml.synthetic_data import HandPose, make_hand


class Function:
    def __init__(self, action):
        self.action = action

    def __call__(self, *args):
        return self.action(*args)


@pytest.fixture
def api_backend(monkeypatch):
    state = SimpleNamespace(send_result=1, send_results=[], move_result=1, calls=[], position=(960, 540))

    def send(count, pointer, size):
        item = pointer._obj
        state.calls.append(("input", item.type,
                            item.u.mi.dwFlags if item.type == 0 else item.u.ki.dwFlags))
        return state.send_results.pop(0) if state.send_results else state.send_result

    def move(x, y):
        state.calls.append(("move", x, y))
        if state.move_result:
            state.position = x, y
        return state.move_result

    def get_position(pointer):
        pointer._obj.x, pointer._obj.y = state.position
        return 1

    api = SimpleNamespace(
        SendInput=Function(send), SetCursorPos=Function(move),
        GetCursorPos=Function(get_position),
        GetSystemMetrics=lambda index: {0: 1920, 1: 1080, 76: 0, 77: 0, 78: 1920, 79: 1080}[index],
        SetProcessDPIAware=lambda: None,
    )
    # Replace both native entry routes before constructing the real adapter.
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: api, raising=False)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        shcore=SimpleNamespace(SetProcessDpiAwareness=lambda *_a: None)), raising=False)
    return Win32Backend(), state


@pytest.mark.parametrize("operation", ["press", "release", "scroll"])
def test_rejected_mouse_input_is_reported_instead_of_claiming_success(api_backend, operation):
    backend, state = api_backend
    state.send_result = 0
    with pytest.raises(OSError):
        getattr(backend, operation)(1 if operation == "scroll" else "left")


def test_rejected_ctrl_down_does_not_turn_zoom_into_an_unmodified_scroll(api_backend):
    backend, state = api_backend
    state.send_result = 0
    with pytest.raises(OSError):
        backend.zoom(1)
    assert state.calls == [("input", 1, 0)], state.calls


def test_rejected_cursor_move_is_reported(api_backend):
    backend, state = api_backend
    state.move_result = 0
    with pytest.raises(OSError):
        backend.move(100, 200)
    assert state.position == (960, 540)


def test_failed_press_does_not_publish_a_held_controller_button(api_backend):
    backend, state = api_backend
    controller = MouseController(Config(), backend)
    state.send_result = 0
    hand = make_hand("pinch_index", np.random.default_rng(7), pose=HandPose(), thumb_gap=0.0, noise=0.0)
    with pytest.raises(OSError):
        controller.update("pinch_index", hand, 0)
    assert controller.button is None


def test_failed_release_retains_state_until_cleanup_can_retry(api_backend):
    backend, state = api_backend
    controller = MouseController(Config(), backend)
    controller.button = "left"
    state.send_result = 0
    with pytest.raises(OSError):
        controller.release_all()
    assert controller.button == "left"
    state.send_result = 1
    controller.halt()
    assert controller.button is None
    assert state.calls == [("input", 0, 4), ("input", 0, 4)]


def test_successful_input_and_cursor_moves_remain_available(api_backend):
    backend, state = api_backend
    backend.move(100, 200)
    backend.press("left")
    backend.release("left")
    backend.zoom(1)
    MouseController(Config(), backend).halt()  # no owned modifier remains to release
    assert state.position == (100, 200)
    assert state.calls == [("move", 100, 200), ("input", 0, 2), ("input", 0, 4),
                           ("input", 1, 0), ("input", 0, 2048), ("input", 1, 2)]


def test_rejected_ctrl_up_is_retried_by_input_cleanup(api_backend):
    backend, state = api_backend
    controller = MouseController(Config(), backend)
    state.send_results = [1, 1, 0]
    with pytest.raises(OSError):
        backend.zoom(1)
    controller.halt()
    assert state.calls == [("input", 1, 0), ("input", 0, 2048),
                           ("input", 1, 2), ("input", 1, 2)]


def test_modifier_cleanup_is_attempted_even_if_mouse_release_is_rejected(api_backend):
    backend, state = api_backend
    controller = MouseController(Config(), backend)
    state.send_results = [1, 1, 0]
    with pytest.raises(OSError):
        backend.zoom(1)
    controller.button = "left"
    state.send_results = [0, 1]
    with pytest.raises(OSError):
        controller.halt()
    assert state.calls[-2:] == [("input", 0, 4), ("input", 1, 2)]
    assert controller.button == "left"
    controller.halt()
    assert controller.button is None
    assert state.calls[-1] == ("input", 0, 4)
