import json
import logging

import pytest

from airtouch.config import Config


class Camera:
    frame_size = (640, 480)
    measured_fps = 30.0


def test_performance_summary_distinguishes_capture_tracking_and_control(caplog):
    from airtouch.performance import PerformanceMonitor

    caplog.set_level(logging.INFO, logger="airtouch.performance")
    monitor = PerformanceMonitor(Config(), Camera(), "rule", interval=5.0)
    monitor.start(0.0)
    for t, seq, hands, raw, track in [
        (1.0, 1, 1, "point", 10.0), (2.0, 2, 1, "point", 20.0),
        (3.0, 4, 0, "none", 30.0), (5.0, 5, 1, "pinch_index", 40.0),
    ]:
        monitor.frame(t, seq, hands, raw, "point", .9, track, 1.0, 2.0)
    records = [json.loads(r.message) for r in caplog.records if r.name == "airtouch.performance"]
    assert records[0]["event"] == "session_start"
    sample = records[-1]
    assert sample["event"] == "performance"
    assert sample["processing_fps"] == pytest.approx(.8)
    assert sample["capture_fps"] == 30.0
    assert sample["tracking_ms"] == {"mean": 25.0, "p95": 40.0, "max": 40.0}
    assert sample["control_ms"]["mean"] == 1.0
    assert sample["confidence"]["mean"] == .9
    assert sample["skipped_frames"] == 1
    assert sample["hand_frames"] == 3 and sample["hand_loss_events"] == 1
    assert sample["raw_gestures"] == {"point": 2, "none": 1, "pinch_index": 1}
    assert "landmarks" not in sample and "image" not in sample


def test_short_session_flushes_metrics_and_does_not_log_each_frame(caplog):
    from airtouch.performance import PerformanceMonitor

    caplog.set_level(logging.INFO, logger="airtouch.performance")
    monitor = PerformanceMonitor(Config(), Camera(), "rule")
    monitor.start(0)
    for seq in range(1, 11):
        monitor.frame(seq / 30, seq, 0, "none", "none", 0, 14, .1, None)
    assert len(caplog.records) == 1
    monitor.finish(.4)
    sample = json.loads(caplog.records[-1].message)
    assert sample["event"] == "session_end" and sample["frames"] == 10
    assert sample["frame_age_ms"] is None


def test_performance_logger_rotates_and_creates_valid_json_lines(tmp_path):
    from airtouch.performance import configure_performance_log

    handler = configure_performance_log(tmp_path, max_bytes=140, backups=1)
    logger = logging.getLogger("airtouch.performance")
    try:
        for i in range(10):
            logger.info(json.dumps({"event": "test", "number": i, "value": "x" * 40}))
        handler.flush()
        paths = sorted(tmp_path.glob("performance.jsonl*"))
        assert 1 <= len(paths) <= 2
        assert paths[0].stat().st_size < 200
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                assert json.loads(line)["event"] == "test"
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.propagate = True


def test_a_broken_performance_sink_does_not_stop_gesture_control():
    from airtouch.performance import PerformanceMonitor

    class Broken:
        def info(self, *_args):
            raise OSError("disk full")

    monitor = PerformanceMonitor(Config(), Camera(), "rule", logger=Broken())
    monitor.start(0)
    monitor.frame(5, 1, 1, "point", "point", 1.0, 10, 1, 0)
    monitor.finish(6)


def test_action_metrics_separate_mouse_events_from_recognition_labels(caplog):
    from airtouch.performance import PerformanceMonitor

    caplog.set_level(logging.INFO, logger="airtouch.performance")
    monitor = PerformanceMonitor(Config(), Camera(), "rule")
    monitor.start(0)
    monitor.frame(1, 1, 1, "point", "point", 1, 10, .2, 1,
                  cursor=(100, 100), events=())
    monitor.frame(2, 2, 1, "none", "point", 1, 10, .2, 1,
                  cursor=(103, 104), events=("down:left",))
    monitor.frame(5, 3, 1, "fist", "fist", 1, 10, .2, 1,
                  cursor=(103, 104), events=("up:left",))
    sample = json.loads(caplog.records[-1].message)
    assert sample["mouse_actions"] == {"down:left": 1, "up:left": 1}
    assert sample["cursor_changed_frames"] == 1
    assert sample["cursor_path_px"] == 5.0
    assert sample["unknown_continuation_frames"] == 1


def test_geometry_metrics_help_explain_pinch_thresholds_without_storing_landmarks(caplog):
    import numpy as np
    from airtouch.performance import PerformanceMonitor

    caplog.set_level(logging.INFO, logger="airtouch.performance")
    hand = np.zeros((21, 3))
    hand[9, 1] = .2
    hand[5, 1], hand[6, 1], hand[8, 1] = .2, .3, .4
    hand[10, 1], hand[12, 1], hand[4, 1] = .4, .5, .4
    hand[14, 1], hand[16, 1] = .3, .45
    hand[18, 1], hand[20, 1] = .2, .3
    monitor = PerformanceMonitor(Config(), Camera(), "rule")
    monitor.start(0)
    for seq in (1, 2, 3):
        monitor.frame(5 if seq == 3 else seq, seq, 1, "point", "point", 1, 10, .2, 1,
                      landmarks=hand)
    sample = json.loads(caplog.records[-1].message)
    assert sample["hand_geometry"]["left_gap"]["min"] == 0.0
    assert sample["hand_geometry"]["right_gap"]["mean"] == .5
    assert sample["hand_geometry"]["index_extension"]["mean"] == 1.333
    assert sample["hand_geometry"]["ring_extension"]["mean"] == 1.5
    assert sample["hand_geometry"]["pinky_extension"]["mean"] == 1.5
    assert "landmarks" not in sample and "image" not in sample


def test_tracker_metrics_separate_missing_hands_from_rejected_landmarks(caplog):
    from airtouch.performance import PerformanceMonitor

    caplog.set_level(logging.INFO, logger="airtouch.performance")
    monitor = PerformanceMonitor(Config(), Camera(), "rule")
    monitor.start(0)
    for seq, raw_hands, valid_hands in [(1, 0, 0), (2, 1, 0), (3, 2, 1), (4, 1, 1)]:
        monitor.frame(seq, seq, valid_hands, "none", "none", 1, 10, .2, 1,
                      tracker_hands=raw_hands)
    monitor.frame(5, 5, 1, "point", "point", 1, 10, .2, 1)
    sample = json.loads(caplog.records[-1].message)
    assert sample["hand_frames"] == 3
    assert sample["tracker_observed_frames"] == 4
    assert sample["tracker_hand_frames"] == 3
    assert sample["invalid_hand_frames"] == 2
    monitor.frame(10, 6, 0, "none", "none", 0, 10, .2, 1, tracker_hands=0)
    following = json.loads(caplog.records[-1].message)
    assert following["tracker_observed_frames"] == 1
    assert following["tracker_hand_frames"] == following["invalid_hand_frames"] == 0


def test_old_monitor_calls_leave_unknown_tracker_counts_unavailable(caplog):
    from airtouch.performance import PerformanceMonitor

    caplog.set_level(logging.INFO, logger="airtouch.performance")
    monitor = PerformanceMonitor(Config(), Camera(), "rule")
    monitor.start(0)
    monitor.frame(5, 1, 1, "point", "point", 1, 10, .2, 1)
    sample = json.loads(caplog.records[-1].message)
    assert sample["tracker_observed_frames"] == 0
    assert sample["tracker_hand_frames"] is None
    assert sample["invalid_hand_frames"] is None


def test_frame_quality_samples_existing_frames_at_most_twice_per_second(caplog):
    import numpy as np
    from airtouch.performance import PerformanceMonitor

    caplog.set_level(logging.INFO, logger="airtouch.performance")
    monitor = PerformanceMonitor(Config(), Camera(), "rule")
    monitor.start(0)
    dark = np.zeros((64, 96, 3), dtype=np.uint8)
    bright = np.full_like(dark, 255)
    for seq, (now, frame) in enumerate([(0, dark), (.1, bright), (.49, bright),
                                      (.5, bright), (.6, dark)], 1):
        monitor.frame(now, seq, 0, "none", "none", 0, 10, .2, 1, frame=frame)
    monitor.finish(.7)
    sample = json.loads(caplog.records[-1].message)
    quality = sample["frame_quality"]
    assert quality["samples"] == 2
    assert quality["luma_mean"] == {"mean": 127.5, "p95": 255.0, "max": 255.0, "min": 0.0}
    assert quality["luma_std"]["max"] == 0
    assert quality["dark_fraction"]["mean"] == .5
    assert set(quality) == {"samples", "luma_mean", "luma_std", "dark_fraction"}
    assert "image" not in sample and "frame" not in sample


def test_frame_quality_reports_contrast_on_sparse_luminance_samples(caplog):
    import numpy as np
    from airtouch.performance import PerformanceMonitor

    caplog.set_level(logging.INFO, logger="airtouch.performance")
    frame = np.zeros((64, 96), dtype=np.uint8)
    frame[32:] = 255
    monitor = PerformanceMonitor(Config(), Camera(), "rule")
    monitor.start(0)
    monitor.frame(5, 1, 0, "none", "none", 0, 10, .2, 1, frame=frame)
    sample = json.loads(caplog.records[-1].message)
    assert sample["frame_quality"]["luma_mean"]["mean"] == 127.5
    assert sample["frame_quality"]["luma_std"]["mean"] == 127.5
    assert sample["frame_quality"]["dark_fraction"]["mean"] == .5


@pytest.mark.parametrize("bad_frame", [None, [], "not a frame", "empty", "one_dim", "channels", "float"])
def test_frame_quality_ignores_missing_or_malformed_frames(bad_frame, caplog):
    import numpy as np
    from airtouch.performance import PerformanceMonitor

    malformed = {"empty": np.empty((0, 10, 3), dtype=np.uint8),
                 "one_dim": np.zeros(10, dtype=np.uint8),
                 "channels": np.zeros((2, 2, 2), dtype=np.uint8),
                 "float": np.full((2, 2, 3), np.nan)}
    frame = malformed.get(bad_frame, bad_frame) if isinstance(bad_frame, str) else bad_frame
    caplog.set_level(logging.INFO, logger="airtouch.performance")
    monitor = PerformanceMonitor(Config(), Camera(), "rule")
    monitor.start(0)
    monitor.frame(5, 1, 0, "none", "none", 0, 10, .2, 1, frame=frame)
    sample = json.loads(caplog.records[-1].message)
    assert sample["frames"] == 1
    assert sample["frame_quality"] == {"samples": 0}


def test_headless_loop_passes_original_tracker_count_and_existing_frame_once():
    import threading
    from types import SimpleNamespace
    import numpy as np
    from airtouch.session import _capture_loop

    stop = threading.Event()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    raw_hands = [np.full((21, 3), np.nan)]
    calls = []

    class Tracker:
        def detect(self, received, _timestamp):
            assert received is frame
            calls.append("detect")
            return raw_hands

    class Engine:
        def process_hands(self, received, _timestamp):
            assert received is raw_hands
            return SimpleNamespace(hands=0, raw_label="none", label="none",
                                   confidence=0, cursor=None, events=())

    class Monitor:
        def start(self, _now):
            pass

        def frame(self, *_args, **kwargs):
            calls.append(kwargs)
            stop.set()

    camera = SimpleNamespace(read_if_new=lambda _seq: (1, frame))
    _capture_loop(stop, Config(mirror=False), camera, Tracker(), Engine(), Monitor())
    assert calls[0] == "detect" and len(calls) == 2
    assert calls[1]["tracker_hands"] == 1
    assert calls[1]["frame"] is frame
