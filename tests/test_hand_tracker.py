"""HandTracker against the real MediaPipe model, on made-up frames (no webcam)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from airtouch.config import HAND_MODEL_PATH, Config
from airtouch.hand_tracker import HandTracker, _quiet_native_stderr

needs_model = pytest.mark.skipif(
    not HAND_MODEL_PATH.is_file(), reason="hand model not downloaded yet (run setup.bat)"
)


def blank(width=640, height=480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def tracker():
    with HandTracker(Config()) as t:
        yield t


# -- real model ------------------------------------------------------------------
@needs_model
def test_blank_frame_has_no_hands(tracker):
    assert tracker.detect(blank()) == []


@needs_model
def test_detect_hands_is_an_alias(tracker):
    assert tracker.detect_hands(blank()) == []


@needs_model
def test_non_increasing_timestamps_do_not_raise(tracker):
    for t_ms in (5000, 5000, 4000, 0, -10, 5001):
        assert tracker.detect(blank(), t_ms) == []
    assert tracker.detect(blank()) == []            # and mixed with the built-in clock
    assert tracker.detect(blank(), 1) == []


@needs_model
def test_accepts_frames_that_are_not_contiguous(tracker):
    # e.g. a flip done with slicing: a view with a negative stride
    assert tracker.detect(blank()[:, ::-1]) == []


@needs_model
def test_noisy_picture_does_not_crash(tracker):
    noise = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    hands = tracker.detect(noise)
    assert isinstance(hands, list)


@needs_model
def test_model_loads_from_a_folder_with_non_ascii_name(tmp_path):
    # The whole reason for model_asset_buffer: MediaPipe's own file loader fails here.
    folder = tmp_path / "Bachelor in Computer Science 【 Final Year Project 】"
    folder.mkdir()
    copy = shutil.copy(HAND_MODEL_PATH, folder / "hand_landmarker.task")
    with HandTracker(model_path=copy) as t:
        assert t.detect(blank()) == []


@needs_model
def test_close_is_idempotent_and_detect_after_close_is_a_clear_error():
    t = HandTracker()
    t.close()
    t.close()
    with pytest.raises(RuntimeError):
        t.detect(blank())


@needs_model
def test_start_up_preserves_stderr_and_hides_chatter_where_supported(request):
    if sys.platform == "win32":
        # capfd itself replaces fd2. Observe native output through a child's
        # stable startup pipe instead of invalidating the parent DLL's HANDLE.
        done = subprocess.run(
            [sys.executable, "-B", "-c", "import os, sys; "
             "from airtouch.hand_tracker import HandTracker; "
             "t=HandTracker(); t.close(); "
             "print('python-level message',file=sys.stderr,flush=True); "
             "os.write(2,b'native-level message\\n')"],
            cwd=HAND_MODEL_PATH.parent.parent, capture_output=True, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        assert done.returncode == 0, done.stderr
        assert b"python-level message" in done.stderr and b"native-level message" in done.stderr
        return
    capfd = request.getfixturevalue("capfd")
    capfd.readouterr()
    with HandTracker() as t:
        t.detect(blank())
    print("python-level message", file=sys.stderr, flush=True)
    os.write(2, b"native-level message\n")
    err = capfd.readouterr().err
    if sys.platform != "win32":
        assert "W0000" not in err and "INFO: Created" not in err
    assert "python-level message" in err and "native-level message" in err


# -- problems with the model file ---------------------------------------------------
def test_missing_model_gives_a_plain_message(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        HandTracker(model_path=tmp_path / "nope.task")
    message = str(err.value)
    assert "setup.bat" in message and "python -m airtouch.download_model" in message
    assert "找不到" in message


@needs_model
def test_damaged_model_gives_a_plain_message(tmp_path):
    bad = tmp_path / "bad.task"
    bad.write_bytes(b"this is not a model" * 100)
    with pytest.raises(RuntimeError) as err:
        HandTracker(model_path=bad)
    assert "打不开" in str(err.value) and "setup.bat" in str(err.value)


# -- our side of the API, with the native part stubbed out ----------------------------
class StubLandmarker:
    def __init__(self):
        self.calls = []
        self.closed = 0
        self.hands = []

    def detect_for_video(self, image, timestamp_ms):
        self.calls.append((image.numpy_view().copy(), timestamp_ms))
        return SimpleNamespace(hand_landmarks=self.hands)

    def close(self):
        self.closed += 1


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """HandTracker with a stub in place of the native landmarker; records its options."""
    from mediapipe.tasks.python import vision

    model = tmp_path / "model.task"
    model.write_bytes(b"model-bytes")
    stub = StubLandmarker()
    seen = {}

    def create(options):
        seen["options"] = options
        return stub

    monkeypatch.setattr(vision.HandLandmarker, "create_from_options", staticmethod(create))
    return SimpleNamespace(model=model, stub=stub, seen=seen)


def test_options_come_from_config_and_model_comes_from_bytes(stubbed):
    from mediapipe.tasks.python import vision

    cfg = Config(max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.4)
    HandTracker(cfg, model_path=stubbed.model)
    options = stubbed.seen["options"]
    assert options.num_hands == 2
    assert options.running_mode == vision.RunningMode.VIDEO
    assert options.min_hand_detection_confidence == pytest.approx(0.7)
    assert options.min_hand_presence_confidence == pytest.approx(0.4)
    assert options.min_tracking_confidence == pytest.approx(0.4)
    assert options.base_options.model_asset_buffer == b"model-bytes"
    assert options.base_options.model_asset_path is None


def test_num_hands_argument_overrides_config(stubbed):
    HandTracker(Config(max_hands=2), model_path=stubbed.model, num_hands=1)
    assert stubbed.seen["options"].num_hands == 1


def test_landmarks_become_float_arrays_in_mediapipe_order(stubbed):
    t = HandTracker(model_path=stubbed.model)
    points = [SimpleNamespace(x=i / 100, y=i / 50, z=-i / 200) for i in range(21)]
    stubbed.stub.hands = [points, points]
    hands = t.detect(blank(64, 48))
    assert len(hands) == 2
    for hand in hands:
        assert hand.shape == (21, 3) and hand.dtype == np.float64
        assert hand[5] == pytest.approx([0.05, 0.1, -0.025])


def test_frame_reaches_mediapipe_as_contiguous_rgb(stubbed):
    t = HandTracker(model_path=stubbed.model)
    frame = blank(4, 2)
    frame[..., 0] = 10      # B
    frame[..., 1] = 20      # G
    frame[..., 2] = 30      # R
    t.detect(frame, 1000)
    image, _ = stubbed.stub.calls[-1]
    assert image.shape == (2, 4, 3)
    assert image[0, 0].tolist() == [30, 20, 10]
    assert frame[0, 0].tolist() == [10, 20, 30]     # the caller's frame is left alone


def test_timestamps_reach_mediapipe_strictly_increasing(stubbed):
    t = HandTracker(model_path=stubbed.model)
    for t_ms in (100, 100, 50, 0, 200, None, 7):
        t.detect(blank(4, 2), t_ms)
    stamps = [ts for _, ts in stubbed.stub.calls]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    assert stamps[1:4] == [100, 101, 102]           # caller's values kept when possible; bumped when not


def test_close_closes_the_native_side_once(stubbed):
    with HandTracker(model_path=stubbed.model) as t:
        pass
    t.close()
    assert stubbed.stub.closed == 1


def test_warmup_failure_closes_the_created_native_tracker(stubbed, monkeypatch):
    def fail_warmup(_image, _timestamp_ms):
        raise RuntimeError("warm-up failed")

    monkeypatch.setattr(stubbed.stub, "detect_for_video", fail_warmup)
    with pytest.raises(RuntimeError, match="warm-up failed"):
        HandTracker(model_path=stubbed.model)
    assert stubbed.stub.closed == 1


def test_quiet_block_restores_stderr_after_an_exception(request):
    if sys.platform == "win32":
        code = (
            "import os\nfrom airtouch.hand_tracker import _quiet_native_stderr\n"
            "try:\n    with _quiet_native_stderr():\n"
            "        os.write(2,b'hidden\\n')\n        raise ValueError('boom')\n"
            "except ValueError:\n    pass\nos.write(2,b'visible\\n')\n"
        )
        done = subprocess.run([sys.executable, "-B", "-c", code],
                              cwd=HAND_MODEL_PATH.parent.parent, capture_output=True,
                              timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        assert done.returncode == 0, done.stderr
        assert done.stderr == b"hidden\nvisible\n"
        return
    capfd = request.getfixturevalue("capfd")
    capfd.readouterr()
    with pytest.raises(ValueError):
        with _quiet_native_stderr():
            os.write(2, b"hidden\n")
            raise ValueError("boom")
    os.write(2, b"visible\n")
    err = capfd.readouterr().err
    if sys.platform != "win32":
        assert "hidden" not in err
    else:
        assert "hidden" in err    # Windows keeps native stderr HANDLEs stable.
    assert "visible" in err
