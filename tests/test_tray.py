"""The tray switch: icon drawing, menu, state changes, single-instance guard.

The real icon loop is never started and the registry is always faked. The Windows
parts that can run without showing anything (menu and icon handles, the named
mutex, the pythonw entry point up to its single-instance exit) are run for real.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from airtouch import autostart, tray
from airtouch.config import PROJECT_ROOT
from airtouch.session import ERROR_PREFIX, CaptureWorker
from airtouch.tray import SingleInstance, TrayApp, make_icon

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


@pytest.fixture(autouse=True)
def fake_registry(monkeypatch):
    class Registry:
        def __init__(self):
            self.values, self.fail_writes = {}, False

        def get(self, name):
            return self.values.get(name)

        def set(self, name, value):
            if self.fail_writes:
                raise PermissionError("denied")
            self.values[name] = value

        def delete(self, name):
            if self.fail_writes:
                raise PermissionError("denied")
            self.values.pop(name, None)

    registry = Registry()
    monkeypatch.setattr(autostart, "backend", registry)
    return registry


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


# --------------------------------------------------------------------- the icon
def opaque(img: Image.Image) -> np.ndarray:
    return np.asarray(img)[:, :, 3] > 128


def runs(line: np.ndarray) -> int:
    """How many separate stretches of True in a row of pixels."""
    return int(np.sum(np.diff(np.concatenate(([0], line.astype(int)))) == 1))


def digits(img: Image.Image):
    """Bounding boxes (top, bottom, left, right) of the separate shapes above the palm."""
    mask = opaque(img).copy()
    mask[round(img.height * 0.50):] = False
    labels, count = ndimage.label(mask)
    boxes = [(s[0].start, s[0].stop, s[1].start, s[1].stop) for s in ndimage.find_objects(labels)]
    return sorted(boxes, key=lambda box: box[2])    # left to right: thumb, then index ... little finger


@pytest.mark.parametrize("size", [16, 32, 64, 256])
@pytest.mark.parametrize("on", [True, False])
def test_icon_size_and_mode(on, size):
    img = make_icon(on, size)
    assert img.mode == "RGBA" and img.size == (size, size)


def test_default_icon_size_is_64():
    assert make_icon(True).size == (64, 64)


def test_icon_has_a_transparent_background():
    alpha = np.asarray(make_icon(True, 64))[:, :, 3]
    assert (alpha == 0).mean() > 0.3
    assert alpha[0, 0] == alpha[0, -1] == alpha[-1, 0] == alpha[-1, -1] == 0


def test_on_is_teal_and_off_is_grey():
    def mean_color(on):
        px = np.asarray(make_icon(on, 64)).astype(float)
        return px[px[:, :, 3] > 250][:, :3].mean(axis=0)

    r, g, b = mean_color(True)
    assert g > r + 60 and b > r + 60 and abs(g - b) < 30    # teal: green+blue, no red
    r, g, b = mean_color(False)
    assert max(r, g, b) - min(r, g, b) < 25                 # grey: no strong colour
    assert np.abs(mean_color(True) - mean_color(False)).sum() > 60


def test_on_and_off_have_the_same_shape():
    assert np.array_equal(opaque(make_icon(True, 64)), opaque(make_icon(False, 64)))


@pytest.mark.parametrize("size", [64, 128, 256, 512])
def test_a_line_across_the_hand_crosses_exactly_five_fingers(size):
    mask = opaque(make_icon(True, size))
    assert runs(mask[round(size * 0.47)]) == 5


@pytest.mark.parametrize("size", [64, 256])
def test_five_separate_digits_stick_out_of_the_palm(size):
    assert len(digits(make_icon(True, size))) == 5


def test_the_four_long_fingers_are_similar_so_none_stands_out_alone():
    """Guards against the original icon's accident: it looked like a raised middle finger."""
    size = 256
    thumb, *fingers = digits(make_icon(True, size))
    assert len(fingers) == 4
    heights = [bottom - top for top, bottom, _left, _right in fingers]
    assert min(heights) >= 0.70 * max(heights)
    tips = [top for top, _bottom, _left, _right in fingers]
    assert sorted(tips)[1] - min(tips) < 0.06 * size        # the tallest barely beats the next one
    # ... and the middle finger, if tallest at all, is only a neighbour's height away
    middle_tip, neighbours = tips[1], (tips[0], tips[2])
    assert min(neighbours) - middle_tip < 0.06 * size


def test_the_thumb_is_shorter_and_angled_out_to_the_side():
    size = 256
    thumb, *fingers = digits(make_icon(True, size))
    top, bottom, left, right = thumb
    assert bottom - top < min(f[1] - f[0] for f in fingers)
    assert left < fingers[0][2] and right < fingers[0][2]   # completely to the side of the fingers
    assert top > min(f[0] for f in fingers) + 0.2 * size    # its tip is lower than the fingertips

    # angled: in the full picture it runs diagonally, so its width is comparable to its height
    full = opaque(make_icon(True, size))
    thumb_only = full[:, : fingers[0][2]]
    rows, cols = np.nonzero(thumb_only[: round(size * 0.70)])
    assert (cols.max() - cols.min()) > 0.5 * (rows.max() - rows.min())


@pytest.mark.parametrize("size", [16, 24, 32])
def test_fingers_stay_visible_at_real_tray_sizes(size):
    """Windows gets these from pystray's .ico: LANCZOS thumbnails of the 64 px icon."""
    small = make_icon(True, 64).copy()
    small.thumbnail((size, size), Image.Resampling.LANCZOS, reducing_gap=None)
    alpha = np.asarray(small)[:, :, 3]
    # fingertips: four bumps in the alpha along a row through the upper fingers
    assert runs(alpha[round(size * 0.27)] > 100) == 4


def test_published_previews_match_the_current_drawing():
    for name, on in (("tray_on.png", True), ("tray_off.png", False)):
        with Image.open(PROJECT_ROOT / "docs" / "images" / name) as saved:
            assert saved.size == (256, 256)
            assert np.array_equal(np.asarray(saved.convert("RGBA")), np.asarray(make_icon(on, 256)))


def test_save_previews_writes_both_files(tmp_path):
    tray.save_previews(tmp_path / "pics", size=32)
    assert Image.open(tmp_path / "pics" / "tray_on.png").size == (32, 32)
    assert Image.open(tmp_path / "pics" / "tray_off.png").size == (32, 32)


# -------------------------------------------------------------------- test doubles
class FakeWorker:
    def __init__(self):
        self.on_status = None
        self.is_running = False
        self.calls: list = []

    def start(self):
        self.calls.append("start")
        self.is_running = True

    def stop(self, timeout=3):
        self.calls.append(("stop", timeout))
        self.is_running = False
        return True


class FakeIcon:
    HAS_NOTIFICATION = True

    def __init__(self):
        self.icon, self.title = None, None
        self.notifications: list[tuple] = []
        self.menu_updates = 0
        self.stopped = False

    def update_menu(self):
        self.menu_updates += 1

    def notify(self, message, title=None):
        self.notifications.append((message, title))

    def stop(self):
        self.stopped = True


@pytest.fixture
def app():
    tray_app = TrayApp(worker=FakeWorker())
    tray_app.icon = FakeIcon()
    return tray_app


def menu_items(app):
    return [item for item in app.build_menu().items]


# -------------------------------------------------------------- switching on / off
def test_starts_off_without_touching_the_camera(app):
    assert app.on is False
    assert app.worker.calls == []
    assert app.title == tray.TITLE_OFF


def test_turning_on_starts_the_worker_and_shows_it(app):
    app.toggle()
    assert app.on is True
    assert app.worker.calls == ["start"]
    assert app.icon.icon is app._images[True]
    assert app.icon.title == tray.TITLE_ON
    assert app.icon.menu_updates == 1


def test_turning_off_stops_the_worker_without_blocking_the_menu(app):
    app.toggle()
    app.toggle()
    assert app.on is False
    assert app.worker.calls == ["start", ("stop", 0)]
    assert app.icon.icon is app._images[False]
    assert app.icon.title == tray.TITLE_OFF


def test_setting_the_same_state_twice_does_nothing(app):
    app.set_on(False)
    assert app.worker.calls == []
    app.set_on(True)
    app.set_on(True)
    assert app.worker.calls == ["start"]


def test_on_and_off_icons_really_differ(app):
    assert app._images[True].tobytes() != app._images[False].tobytes()


# ------------------------------------------------------------------- worker errors
def test_worker_error_switches_off_and_notifies(app):
    app.toggle()
    app.worker.is_running = False
    app.worker.on_status(ERROR_PREFIX + "打不开摄像头")
    assert app.on is False
    assert app.icon.icon is app._images[False]
    assert app.icon.title == tray.TITLE_OFF
    assert app.icon.notifications == [("打不开摄像头", "AirTouch")]


def test_very_long_error_fits_in_a_windows_notification(app):
    app.toggle()
    app.worker.on_status(ERROR_PREFIX + "错" * 1000)
    assert len(app.icon.notifications[0][0]) <= 255


def test_ordinary_status_changes_nothing_while_running(app):
    app.toggle()
    app.worker.on_status("正在准备")
    assert app.on is True
    assert app.icon.notifications == []


def test_loop_ending_by_itself_switches_off_quietly(app):
    app.toggle()
    app.worker.is_running = False
    app.worker.on_status("手势控制已经结束。")
    assert app.on is False
    assert app.icon.icon is app._images[False]
    assert app.icon.notifications == []


def test_status_arriving_when_already_off_is_harmless(app):
    updates = app.icon.menu_updates
    app.worker.on_status("已关闭，摄像头也关掉了。")
    assert app.on is False and app.icon.menu_updates == updates


def test_icon_without_notification_support_is_fine(app):
    app.icon.HAS_NOTIFICATION = False
    app.toggle()
    app.worker.on_status(ERROR_PREFIX + "打不开摄像头")
    assert app.on is False and app.icon.notifications == []


def test_a_failing_icon_update_cannot_kill_the_worker_callback(app):
    def broken():
        raise OSError("shell is restarting")

    app.icon.update_menu = broken
    app.icon.notify = lambda *_a: broken()
    app.toggle()
    app.worker.on_status(ERROR_PREFIX + "打不开摄像头")     # must not raise
    assert app.on is False


def test_error_before_the_icon_exists_is_harmless():
    tray_app = TrayApp(worker=FakeWorker())
    tray_app.worker.on_status(ERROR_PREFIX + "打不开摄像头")
    assert tray_app.on is False


def test_with_the_real_worker_error_goes_all_the_way_back_to_off():
    def failing_loop(_stop_event, _on_status):
        raise RuntimeError("打不开摄像头")

    tray_app = TrayApp(worker=CaptureWorker(loop_fn=failing_loop))
    tray_app.icon = FakeIcon()
    tray_app.toggle()
    assert wait_until(lambda: not tray_app.on)
    assert wait_until(lambda: bool(tray_app.icon.notifications))
    assert tray_app.icon.notifications == [("打不开摄像头", "AirTouch")]
    assert tray_app.icon.title == tray.TITLE_OFF
    assert not tray_app.worker.is_running


def test_with_the_real_worker_off_ends_the_loop():
    stopped = threading.Event()

    def loop(stop_event, _on_status):
        stop_event.wait(10)
        stopped.set()

    tray_app = TrayApp(worker=CaptureWorker(loop_fn=loop))
    tray_app.icon = FakeIcon()
    tray_app.toggle()
    assert tray_app.worker.is_running
    tray_app.toggle()
    assert stopped.wait(5)
    assert wait_until(lambda: not tray_app.worker.is_running)
    assert tray_app.on is False


def test_with_the_real_worker_rapid_clicking_settles_on_the_last_state():
    active, most_active = [0], [0]
    lock = threading.Lock()

    def loop(stop_event, _on_status):
        with lock:
            active[0] += 1
            most_active[0] = max(most_active[0], active[0])
        stop_event.wait(10)
        time.sleep(0.03)
        with lock:
            active[0] -= 1

    tray_app = TrayApp(worker=CaptureWorker(loop_fn=loop))
    tray_app.icon = FakeIcon()
    for _ in range(7):                  # OFF -> ON ... ends ON
        tray_app.toggle()
    assert tray_app.on is True
    assert wait_until(lambda: active[0] == 1)
    assert tray_app.worker.stop() is True
    assert most_active[0] == 1 and active[0] == 0


# -------------------------------------------------------------------------- menu
def test_menu_has_switch_autostart_and_quit(app):
    items = menu_items(app)
    assert [i.text for i in items if i is not tray_pystray().Menu.SEPARATOR] == [
        tray.TEXT_TURN_ON, tray.TEXT_AUTOSTART, tray.TEXT_QUIT]


def tray_pystray():
    import pystray

    return pystray


def test_menu_text_is_simple_chinese_with_english_after_a_slash():
    for text in (tray.TEXT_TURN_ON, tray.TEXT_TURN_OFF, tray.TEXT_AUTOSTART, tray.TEXT_QUIT):
        chinese, english = text.split(" / ")
        assert all("一" <= ch <= "鿿" for ch in chinese)
        assert english.isascii()


def test_first_item_is_the_default_so_left_click_toggles(app):
    menu = app.build_menu()
    assert [i.default for i in menu.items][:2] == [True, False]
    menu(app.icon)              # what pystray does on a left-click
    assert app.on is True
    menu(app.icon)
    assert app.on is False


def test_switch_text_follows_the_state(app):
    switch = menu_items(app)[0]
    assert switch.text == tray.TEXT_TURN_ON
    switch(app.icon)
    assert switch.text == tray.TEXT_TURN_OFF
    switch(app.icon)
    assert switch.text == tray.TEXT_TURN_ON


def test_autostart_checkbox_reads_and_writes_the_registry(app, fake_registry):
    box = menu_items(app)[1]
    assert box.checked is False
    box(app.icon)
    assert box.checked is True
    assert fake_registry.values == {"AirTouch": autostart.build_command()}
    box(app.icon)
    assert box.checked is False
    assert fake_registry.values == {}
    assert app.icon.notifications == []


def test_autostart_failure_tells_the_user(app, fake_registry):
    fake_registry.fail_writes = True
    box = menu_items(app)[1]
    box(app.icon)
    assert box.checked is False
    assert app.icon.notifications == [(tray.MSG_AUTOSTART_FAILED, "AirTouch")]


def test_quit_stops_the_worker_first_and_then_the_icon(app):
    order = []
    app.worker.stop = lambda timeout=3: order.append("worker")
    app.icon.stop = lambda: order.append("icon")
    app.toggle()
    menu_items(app)[-1](app.icon)
    assert order == ["worker", "icon"]
    assert app.on is False


def test_quit_waits_for_the_camera_to_close_really():
    closed = threading.Event()

    def loop(stop_event, _on_status):
        stop_event.wait(10)
        time.sleep(0.1)
        closed.set()

    tray_app = TrayApp(worker=CaptureWorker(loop_fn=loop))
    tray_app.icon = FakeIcon()
    tray_app.toggle()
    tray_app.quit()
    assert closed.is_set()
    assert tray_app.icon.stopped


def test_tray_gives_the_worker_its_status_callback():
    worker = FakeWorker()
    tray_app = TrayApp(worker=worker)
    assert worker.on_status == tray_app._on_status


def test_default_worker_is_a_capture_worker():
    assert isinstance(TrayApp().worker, CaptureWorker)


# ------------------------------------------------------- real Win32, nothing shown
@windows_only
def test_real_windows_menu_and_icon_handles_can_be_built(app):
    """No window, no icon on screen: only native objects the running icon would use."""
    import pystray
    from pystray._util import win32

    icon = pystray.Icon("AirTouchTest" + uuid.uuid4().hex, app._images[False], tray.TITLE_OFF, app.build_menu())
    icon.update_menu()
    hmenu, callbacks = icon._menu_handle
    assert hmenu and len(callbacks) == 4
    win32.DestroyMenu(hmenu)
    icon._menu_handle = None

    for on in (False, True):            # an RGBA PIL image -> .ico -> a real icon handle
        icon._icon = app._images[on]
        icon._assert_icon_handle()
        assert icon._icon_handle
        icon._release_icon()


@windows_only
def test_tooltip_and_notification_text_fit_the_windows_limits():
    import ctypes

    from pystray._util import win32

    for title in (tray.TITLE_ON, tray.TITLE_OFF):
        win32.NOTIFYICONDATAW(cbSize=ctypes.sizeof(win32.NOTIFYICONDATAW), szTip=title)
    win32.NOTIFYICONDATAW(szInfo="错" * tray.MAX_NOTIFICATION_CHARS, szInfoTitle=tray.APP_NAME)


# --------------------------------------------------------------- single instance
class FakeMutexes:
    """Stands in for CreateMutexW: the second request for a name finds it taken."""

    def __init__(self):
        self.taken: set[str] = set()
        self.closed: list = []

    def create(self, name):
        already = name in self.taken
        self.taken.add(name)
        return f"handle-{name}", already

    def close(self, handle):
        self.closed.append(handle)
        self.taken.discard(handle.removeprefix("handle-"))


def test_first_instance_wins_and_second_is_refused():
    fake = FakeMutexes()
    assert SingleInstance("x", fake.create, fake.close).acquire() is True
    assert SingleInstance("x", fake.create, fake.close).acquire() is False


def test_other_names_do_not_interfere():
    fake = FakeMutexes()
    assert SingleInstance("x", fake.create, fake.close).acquire() is True
    assert SingleInstance("y", fake.create, fake.close).acquire() is True


def test_release_lets_the_next_instance_in():
    fake = FakeMutexes()
    first = SingleInstance("x", fake.create, fake.close)
    assert first.acquire() is True
    first.release()
    first.release()         # twice is fine
    assert fake.closed == ["handle-x"]
    assert SingleInstance("x", fake.create, fake.close).acquire() is True


def test_acquiring_the_same_guard_twice_keeps_its_original_handle():
    fake = FakeMutexes()
    guard = SingleInstance("x", fake.create, fake.close)
    assert guard.acquire() is True
    assert guard.acquire() is True
    guard.release()
    assert fake.closed == ["handle-x"]


def test_a_failed_mutex_creation_cannot_start_an_unguarded_instance():
    guard = SingleInstance("x", create_mutex=lambda _name: (None, False))
    with pytest.raises(OSError):
        guard.acquire()


def test_main_releases_its_guard_when_the_icon_fails(monkeypatch):
    fake = FakeMutexes()
    guard = SingleInstance("x", fake.create, fake.close)

    class BrokenApp:
        def run(self):
            raise OSError("shell unavailable")

    monkeypatch.setattr(tray, "SingleInstance", lambda: guard)
    monkeypatch.setattr(tray, "_make_dpi_aware", lambda: None)
    monkeypatch.setattr(tray, "TrayApp", BrokenApp)
    with pytest.raises(OSError, match="shell unavailable"):
        tray.main()
    assert fake.closed == ["handle-x"]


def test_guard_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(tray.sys, "platform", "linux")
    guard = SingleInstance("anything")
    assert guard.acquire() is True
    assert guard.acquire() is True
    guard.release()


@windows_only
def test_real_named_mutex():
    name = "Local\\AirTouchTest" + uuid.uuid4().hex
    first, second, third = SingleInstance(name), SingleInstance(name), SingleInstance(name)
    assert first.acquire() is True
    assert second.acquire() is False
    second.release()
    first.release()
    assert third.acquire() is True
    third.release()


# ------------------------------------------------------------------------- main()
def test_main_exits_quietly_when_already_running(monkeypatch):
    monkeypatch.setattr(tray.SingleInstance, "acquire", lambda self: False)

    def must_not_build(*_a, **_k):
        raise AssertionError("second instance must not create a tray icon")

    monkeypatch.setattr(tray, "TrayApp", must_not_build)
    assert tray.main() == 0


def test_main_sets_up_in_order_and_runs_the_icon(monkeypatch):
    order = []

    class App:
        def run(self):
            order.append("run")

    monkeypatch.setattr(tray.SingleInstance, "acquire", lambda self: order.append("guard") or True)
    monkeypatch.setattr(tray, "_make_dpi_aware", lambda: order.append("dpi"))
    monkeypatch.setattr(tray, "TrayApp", lambda: order.append("app") or App())
    assert tray.main() == 0
    assert order == ["guard", "dpi", "app", "run"]


def test_run_stops_the_worker_even_if_the_icon_loop_dies(monkeypatch):
    import pystray

    class DyingIcon:
        def __init__(self, name, image, title, menu):
            self.args = (name, image, title, menu)

        def run(self):
            raise OSError("no tray available")

    monkeypatch.setattr(pystray, "Icon", DyingIcon)
    app = TrayApp(worker=FakeWorker())
    with pytest.raises(OSError):
        app.run()
    assert app.icon.args[0] == "AirTouch"
    assert app.icon.args[1] is app._images[False]      # parked: OFF at start
    assert app.icon.args[2] == tray.TITLE_OFF
    assert app.worker.calls == [("stop", 3)]


@windows_only
def test_entry_point_exits_quietly_when_another_instance_holds_the_mutex():
    """pythonw + airtouch_tray.pyw for real, launched with the exact autostart command.

    The mutex is held here, so it must exit at its guard: no icon, no camera. If the
    guard ever broke, the timeout below fails the test and the process is killed.
    """
    guard = SingleInstance()
    guard.acquire()
    try:
        command = autostart.build_command()
        process = subprocess.Popen(command)
        try:
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.kill()
    finally:
        guard.release()
    assert (PROJECT_ROOT / "logs" / "tray.log").exists()
