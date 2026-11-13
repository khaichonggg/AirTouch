"""Pointer feedback only: renderer doubles never open a window or move a mouse."""

from __future__ import annotations

import importlib
import ctypes
import sys
import threading
import time

import pytest
from PIL import Image


def halo_module():
    return importlib.import_module("airtouch.cursor_halo")


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


class Renderer:
    def __init__(self):
        self.calls = [("create", threading.get_ident())]
        self.closed = threading.Event()

    def show(self, position, pressed):
        self.calls.append(("show", position, pressed, threading.get_ident()))

    def hide(self):
        self.calls.append(("hide", threading.get_ident()))

    def pump(self):
        return True

    def close(self):
        self.calls.append(("close", threading.get_ident()))
        self.closed.set()

    @property
    def shown(self):
        return [call for call in self.calls if call[0] == "show"]


def test_hidden_feedback_does_not_create_a_native_renderer():
    made = []
    halo = halo_module().CursorHalo(renderer_factory=lambda: made.append(Renderer()))
    halo.update(None, active=False)
    halo.close()
    halo.close()
    halo.update((5, 6))
    assert made == []


def test_every_renderer_operation_belongs_to_its_worker_thread():
    made = []

    def create():
        made.append(Renderer())
        return made[-1]

    halo = halo_module().CursorHalo(renderer_factory=create)
    halo.update((-123, 456), pressed=True)
    assert wait_until(lambda: made and made[0].shown)
    halo.update(None, active=False)
    assert wait_until(lambda: any(c[0] == "hide" for c in made[0].calls))
    halo.close()
    halo.close()
    renderer = made[0]
    assert renderer.shown[0][1:3] == ((-123, 456), True)
    owners = {call[-1] for call in renderer.calls}
    assert len(owners) == 1 and threading.get_ident() not in owners
    assert [c[0] for c in renderer.calls].count("close") == 1


def test_updates_are_nonblocking_and_only_latest_survives_slow_startup():
    entered, proceed = threading.Event(), threading.Event()
    made = []

    def create():
        entered.set()
        assert proceed.wait(2)
        made.append(Renderer())
        return made[-1]

    halo = halo_module().CursorHalo(renderer_factory=create)
    try:
        halo.update((1, 2))
        assert entered.wait(2)
        started = time.monotonic()
        for i in range(1000):
            halo.update((-i, i), pressed=bool(i % 2))
        assert time.monotonic() - started < 0.2
        proceed.set()
        assert wait_until(lambda: made and made[0].shown)
        assert [call[1:3] for call in made[0].shown] == [((-999, 999), True)]
    finally:
        proceed.set()
        halo.close()


def test_close_during_initialization_never_shows_pending_feedback():
    entered, proceed = threading.Event(), threading.Event()
    made = []

    def create():
        entered.set()
        assert proceed.wait(2)
        made.append(Renderer())
        return made[-1]

    halo = halo_module().CursorHalo(renderer_factory=create)
    halo.update((100, 200))
    assert entered.wait(2)
    closer = threading.Thread(target=halo.close)
    closer.start()
    assert wait_until(lambda: halo._closed)
    proceed.set()
    closer.join(2)
    assert not closer.is_alive()
    assert made[0].closed.is_set()
    assert made[0].shown == []


def test_stale_feedback_hides_but_a_new_update_can_show_again():
    renderer = Renderer()
    halo = halo_module().CursorHalo(renderer_factory=lambda: renderer, timeout=0.05)
    try:
        halo.update((30, 40))
        assert wait_until(lambda: renderer.shown)
        assert wait_until(lambda: any(c[0] == "hide" for c in renderer.calls))
        halo.update((30, 40), pressed=True)
        assert wait_until(lambda: len(renderer.shown) == 2)
        assert renderer.shown[-1][1:3] == ((30, 40), True)
    finally:
        halo.close()


def test_stationary_updates_refresh_expiry_without_redrawing():
    renderer = Renderer()
    halo = halo_module().CursorHalo(renderer_factory=lambda: renderer, timeout=0.08)
    try:
        for _ in range(15):
            halo.update((10, 20))
            time.sleep(0.01)
        assert renderer.shown
        assert not any(c[0] == "hide" for c in renderer.calls)
        assert len(renderer.shown) == 1
    finally:
        halo.close()


@pytest.mark.parametrize("failure", ["factory", "show", "pump", "close"])
def test_renderer_failures_disable_feedback_and_log_once(caplog, failure):
    renderer = Renderer()

    def broken(*_args):
        raise OSError("renderer unavailable")

    if failure != "factory":
        setattr(renderer, failure, broken)
    halo = halo_module().CursorHalo(renderer_factory=broken if failure == "factory" else lambda: renderer)
    halo.update((4, 5))
    if failure == "close":
        assert wait_until(lambda: renderer.shown)
        halo.close()
    else:
        assert wait_until(lambda: halo._closed)
        halo.close()
    for _ in range(10):
        halo.update((6, 7))
    messages = [r for r in caplog.records if r.name == "airtouch.cursor_halo"]
    assert len(messages) == 1
    if failure not in ("factory", "close"):
        assert renderer.closed.is_set()


def test_draw_and_cleanup_failures_do_not_log_twice(caplog):
    renderer = Renderer()

    def broken(*_args):
        raise OSError("renderer unavailable")

    renderer.show = renderer.close = broken
    halo = halo_module().CursorHalo(renderer_factory=lambda: renderer)
    halo.update((4, 5))
    assert wait_until(lambda: halo._closed)
    halo.close()
    assert len([r for r in caplog.records if r.name == "airtouch.cursor_halo"]) == 1


def test_ring_center_and_corners_stay_transparent_with_a_visible_outline():
    image = halo_module().make_halo_image()
    assert image.mode == "RGBA"
    assert image.size == (40, 40)
    assert image.getpixel((20, 20))[3] == 0
    assert image.getpixel((0, 0))[3] == 0
    assert max(image.getchannel("A").tobytes()) > 150


def test_pressed_ring_changes_color_and_dpi_scales_its_canvas():
    module = halo_module()
    normal = module.make_halo_image()
    pressed = module.make_halo_image(pressed=True)
    assert normal.tobytes() != pressed.tobytes()
    assert module.make_halo_image(dpi=144).size == (60, 60)
    assert module.make_halo_image(dpi=192).size == (80, 80)
    # The strongest cyan/warm pixels must retain different dominant channels.
    normal_colors = [normal.getpixel((x, y)) for x in range(40) for y in range(40)
                     if normal.getpixel((x, y))[3] > 150]
    pressed_colors = [pressed.getpixel((x, y)) for x in range(40) for y in range(40)
                      if pressed.getpixel((x, y))[3] > 150]
    assert any(g > r and b > r for r, g, b, _a in normal_colors)
    assert any(r > g and r > b for r, g, b, _a in pressed_colors)


def test_layered_bitmap_uses_top_down_premultiplied_bgra():
    image = Image.new("RGBA", (2, 1))
    image.putdata([(200, 100, 50, 128), (200, 100, 50, 0)])
    assert halo_module().premultiplied_bgra(image) == bytes((25, 50, 100, 128, 0, 0, 0, 0))


class ForeignFunction:
    """ctypes signature attributes with all OS calls replaced below that boundary."""

    def __init__(self, native, name):
        self.native, self.name = native, name

    def __call__(self, *args):
        self.native.calls.append((self.name, args))
        if self.name in self.native.fail:
            return 0
        callback = getattr(self.native, self.name, None)
        return callback(*args) if callback is not None else 1


class FakeWin32:
    def __init__(self, monkeypatch):
        self.calls = []
        self.fail = set()
        self.dpi = 96
        self.selected = 0x400000004
        self.bitmaps = set()
        self.buffers = []
        self.layer_updates = []
        self.deleted = []
        self.dlls = {}
        names = {
            "kernel32": "GetModuleHandleW",
            "user32": "RegisterClassW UnregisterClassW CreateWindowExW DefWindowProcW DestroyWindow "
                      "ShowWindow SetWindowPos UpdateLayeredWindow PeekMessageW TranslateMessage "
                      "DispatchMessageW SetThreadDpiAwarenessContext GetDpiForWindow",
            "gdi32": "CreateCompatibleDC DeleteDC CreateDIBSection SelectObject DeleteObject",
        }
        import types
        for dll, functions in names.items():
            self.dlls[dll] = types.SimpleNamespace(**{name: ForeignFunction(self, name)
                                                       for name in functions.split()})
        monkeypatch.setattr(ctypes, "WinDLL", lambda name, **_kwargs: self.dlls[name])

    def GetModuleHandleW(self, _name):
        return 0x100000001

    def CreateWindowExW(self, *_args):
        return 0x200000002

    def CreateCompatibleDC(self, _dc):
        return 0x300000003

    def SetThreadDpiAwarenessContext(self, _context):
        return -3

    def GetDpiForWindow(self, _hwnd):
        return self.dpi

    def CreateDIBSection(self, _dc, info, _usage, bits, _section, _offset):
        header = info._obj.bmiHeader
        assert header.biHeight < 0 and header.biBitCount == 32
        buffer = ctypes.create_string_buffer(header.biWidth * -header.biHeight * 4)
        self.buffers.append(buffer)
        bits._obj.value = ctypes.addressof(buffer)
        handle = 0x500000000 + len(self.buffers)
        self.bitmaps.add(handle)
        return handle

    def SelectObject(self, _dc, bitmap):
        previous, self.selected = self.selected, bitmap
        return previous

    def DeleteDC(self, _dc):
        self.selected = None
        return 1

    def DeleteObject(self, bitmap):
        assert self.selected != bitmap, "bitmap was deleted while still selected"
        self.bitmaps.remove(bitmap)
        self.deleted.append(bitmap)
        return 1

    def UpdateLayeredWindow(self, hwnd, _screen_dc, destination, size, _dc, _source,
                            _color_key, blend, flags):
        self.layer_updates.append((hwnd, destination._obj.x, destination._obj.y,
                                   size._obj.cx, size._obj.cy, blend._obj.AlphaFormat, flags))
        return 1

    def PeekMessageW(self, *_args):
        return 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ctypes structure definitions")
def test_native_window_is_click_through_nonactivating_and_preserves_negative_coordinates(monkeypatch):
    native = FakeWin32(monkeypatch)
    renderer = halo_module()._Win32Renderer()
    try:
        renderer.show((-120, 300), False)
        create_args = next(args for name, args in native.calls if name == "CreateWindowExW")
        assert create_args[0] & 0x080800A0 == 0x080800A0
        assert not create_args[3] & 0x10000000       # Created hidden, no WS_VISIBLE.
        assert native.layer_updates[-1] == (0x200000002, -140, 280, 40, 40, 1, 2)
        assert all(args[-1] & 0x0010 for name, args in native.calls if name == "SetWindowPos")
        assert renderer._window_proc(renderer._hwnd, 0x0084, 0, 0) == -1
        assert renderer._window_proc(renderer._hwnd, 0x0021, 0, 0) == 3
        assert native.dlls["user32"].CreateWindowExW.restype is ctypes.wintypes.HWND
        assert native.dlls["user32"].DefWindowProcW.restype is ctypes.c_ssize_t
    finally:
        renderer.close()
    assert not native.bitmaps
    assert [name for name, _args in native.calls].count("DestroyWindow") == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ctypes structure definitions")
def test_native_dpi_change_keeps_the_ring_centered_and_releases_both_bitmaps(monkeypatch):
    native = FakeWin32(monkeypatch)
    renderer = halo_module()._Win32Renderer()
    renderer.show((-120, 300), False)
    native.dpi = 144
    renderer.show((-120, 300), True)
    assert native.layer_updates[-1][1:5] == (-150, 270, 60, 60)
    assert len(native.deleted) == 1
    renderer.close()
    renderer.close()
    assert len(native.deleted) == 2 and not native.bitmaps


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ctypes structure definitions")
def test_native_failed_window_creation_unregisters_class_and_restores_dpi(monkeypatch):
    native = FakeWin32(monkeypatch)
    native.fail.add("CreateWindowExW")
    with pytest.raises(OSError, match="CreateWindowExW"):
        halo_module()._Win32Renderer()
    names = [name for name, _args in native.calls]
    assert "UnregisterClassW" in names
    assert names.count("SetThreadDpiAwarenessContext") == 2
    assert "CreateCompatibleDC" not in names


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ctypes structure definitions")
def test_native_cleanup_continues_after_failure_restoring_the_stock_bitmap(monkeypatch):
    native = FakeWin32(monkeypatch)
    renderer = halo_module()._Win32Renderer()
    renderer.show((100, 200), False)
    native.fail.add("SelectObject")
    with pytest.raises(OSError, match="restore bitmap"):
        renderer.close()
    names = [name for name, _args in native.calls]
    assert "DeleteDC" in names and "UnregisterClassW" in names
    assert not native.bitmaps
