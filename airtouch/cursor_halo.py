"""Small, click-through pointer feedback, without a camera preview or input hooks.

Only the private worker creates or operates native windows. ``update`` replaces
one pending state and wakes that worker, so tracking never waits for rendering.
The renderer factory is injectable: unit tests need neither a desktop nor a mouse.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from PIL import Image, ImageDraw

log = logging.getLogger("airtouch.cursor_halo")


def make_halo_image(dpi: int = 96, pressed: bool = False) -> Image.Image:
    """An antialiased, hollow ring; its canvas is 40 logical pixels wide."""
    size = max(20, round(40 * dpi / 96))
    scale = 4
    canvas = size * scale
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # A quiet white edge keeps the cyan/amber ring legible on dark backgrounds.
    edge = canvas * 0.075
    box = (edge, edge, canvas - edge - 1, canvas - edge - 1)
    draw.ellipse(box, outline=(255, 255, 255, 160), width=max(1, round(canvas * 0.075)))
    inset = canvas * 0.025
    color = (245, 158, 11, 220) if pressed else (6, 182, 212, 210)
    draw.ellipse(tuple(v + inset if i < 2 else v - inset for i, v in enumerate(box)),
                 outline=color, width=max(1, round(canvas * 0.05)))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def premultiplied_bgra(image: Image.Image) -> bytes:
    """Win32 AC_SRC_ALPHA requires BGRA with RGB already multiplied by alpha."""
    pixels = bytearray(image.convert("RGBA").tobytes())
    for i in range(0, len(pixels), 4):
        red, green, blue, alpha = pixels[i:i + 4]
        pixels[i:i + 4] = bytes(((blue * alpha + 127) // 255,
                                 (green * alpha + 127) // 255,
                                 (red * alpha + 127) // 255, alpha))
    return bytes(pixels)


class CursorHalo:
    """Lazy pointer feedback with latest-only updates and stale-state expiry.

    A renderer implements ``show(position, pressed)``, ``hide()``, nonblocking
    ``pump()`` (False ends the worker), and ``close()``. All run on one thread.
    Renderer failures disable this instance and never reach the capture loop.
    """

    def __init__(self, renderer_factory=None, *, timeout: float = 0.3):
        self._factory = renderer_factory or _Win32Renderer
        self._timeout = max(0.01, timeout)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._state = (None, False, 0.0)

    def update(self, position, pressed: bool = False, active: bool = True) -> None:
        """Replace feedback state immediately; None or inactive hides the ring."""
        position = tuple(map(int, position)) if active and position is not None else None
        with self._lock:
            if self._closed:
                return
            self._state = (position, bool(pressed), time.monotonic())
            if self._thread is None and position is not None:
                self._thread = threading.Thread(target=self._run, name="airtouch-cursor-halo", daemon=True)
                try:
                    self._thread.start()
                except Exception:
                    self._closed = True
                    self._thread = None
                    log.exception("cursor halo disabled")
            self._wake.set()

    def close(self) -> None:
        """Stop feedback and release its resources; safe before startup or twice."""
        with self._lock:
            self._closed = True
            thread = self._thread
            self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(1.0)

    def _run(self) -> None:
        renderer = None
        failure = None
        try:
            renderer = self._factory()
            shown = None
            while True:
                self._wake.clear()
                with self._lock:
                    if self._closed:
                        break
                    position, pressed, updated = self._state
                active = position is not None and time.monotonic() - updated < self._timeout
                desired = (position, pressed) if active else None
                if desired != shown:
                    if desired is None:
                        renderer.hide()
                    else:
                        renderer.show(*desired)
                    shown = desired
                if renderer.pump() is False:
                    break
                self._wake.wait(0.02 if shown is not None else 0.1)
        except Exception as exc:
            failure = exc
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception as exc:
                    if failure is None:
                        failure = exc
            with self._lock:
                self._closed = True
            if failure is not None:
                log.error("cursor halo disabled", exc_info=(type(failure), failure, failure.__traceback__))


class _Win32Renderer:
    """Owns one layered HWND and its GDI objects, entirely on its calling thread."""

    # LAYERED | TRANSPARENT | TOOLWINDOW | NOACTIVATE | TOPMOST.
    _EX_STYLE = 0x00080000 | 0x00000020 | 0x00000080 | 0x08000000 | 0x00000008

    def __init__(self):
        if sys.platform != "win32":
            raise OSError("cursor halo requires Windows")
        import ctypes
        from ctypes import wintypes as wt

        self._ct, self._wt = ctypes, wt
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._hwnd = self._dc = self._bitmap = self._old_bitmap = self._atom = None
        self._old_dpi_context = None
        self._visible = False
        self._dirty = False
        self._callback_error = None
        self._position = None
        self._pressed = False
        self._size = 40
        self._image_key = None
        self._images = {}
        self._class_name = f"AirTouchCursorHalo-{id(self):x}"
        self._define_types()
        self._bind_api()
        try:
            set_context = getattr(self._user32, "SetThreadDpiAwarenessContext", None)
            if set_context is not None:
                self._old_dpi_context = set_context(ctypes.c_void_p(-4))
            self._instance = self._kernel32.GetModuleHandleW(None)
            self._wndproc = self._WNDPROC(self._window_proc)
            wc = self._WNDCLASS(0, self._wndproc, 0, 0, self._instance, None, None, None,
                                None, self._class_name)
            self._atom = self._check(self._user32.RegisterClassW(ctypes.byref(wc)), "RegisterClassW")
            self._hwnd = self._check(self._user32.CreateWindowExW(
                self._EX_STYLE, self._class_name, "", 0x80000000, 0, 0, 40, 40,
                None, None, self._instance, None), "CreateWindowExW")
            self._dc = self._check(self._gdi32.CreateCompatibleDC(None), "CreateCompatibleDC")
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _define_types(self) -> None:
        ct, wt = self._ct, self._wt
        self._WNDPROC = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT, ct.c_size_t, ct.c_ssize_t)

        class WNDCLASS(ct.Structure):
            _fields_ = [("style", wt.UINT), ("lpfnWndProc", self._WNDPROC),
                        ("cbClsExtra", ct.c_int), ("cbWndExtra", ct.c_int),
                        ("hInstance", wt.HINSTANCE), ("hIcon", wt.HICON),
                        ("hCursor", wt.HANDLE), ("hbrBackground", wt.HANDLE),
                        ("lpszMenuName", wt.LPCWSTR), ("lpszClassName", wt.LPCWSTR)]

        class BITMAPINFOHEADER(ct.Structure):
            _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                        ("biPlanes", wt.WORD), ("biBitCount", wt.WORD),
                        ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                        ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG),
                        ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]

        class BITMAPINFO(ct.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wt.DWORD * 1)]

        class BLENDFUNCTION(ct.Structure):
            _fields_ = [("BlendOp", ct.c_ubyte), ("BlendFlags", ct.c_ubyte),
                        ("SourceConstantAlpha", ct.c_ubyte), ("AlphaFormat", ct.c_ubyte)]

        self._WNDCLASS = WNDCLASS
        self._BITMAPINFO, self._BITMAPINFOHEADER = BITMAPINFO, BITMAPINFOHEADER
        self._BLENDFUNCTION = BLENDFUNCTION

    def _bind_api(self) -> None:
        ct, wt, u, g, k = self._ct, self._wt, self._user32, self._gdi32, self._kernel32

        def bind(dll, name, arguments, result):
            function = getattr(dll, name)
            function.argtypes, function.restype = arguments, result

        bind(k, "GetModuleHandleW", (wt.LPCWSTR,), wt.HMODULE)
        bind(u, "RegisterClassW", (ct.POINTER(self._WNDCLASS),), wt.ATOM)
        bind(u, "UnregisterClassW", (wt.LPCWSTR, wt.HINSTANCE), wt.BOOL)
        bind(u, "CreateWindowExW", (wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
                                    ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                                    wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID), wt.HWND)
        bind(u, "DefWindowProcW", (wt.HWND, wt.UINT, ct.c_size_t, ct.c_ssize_t), ct.c_ssize_t)
        bind(u, "DestroyWindow", (wt.HWND,), wt.BOOL)
        bind(u, "ShowWindow", (wt.HWND, ct.c_int), wt.BOOL)
        bind(u, "SetWindowPos", (wt.HWND, wt.HWND, ct.c_int, ct.c_int, ct.c_int, ct.c_int, wt.UINT), wt.BOOL)
        bind(u, "UpdateLayeredWindow", (wt.HWND, wt.HDC, ct.POINTER(wt.POINT), ct.POINTER(wt.SIZE),
                                        wt.HDC, ct.POINTER(wt.POINT), wt.DWORD,
                                        ct.POINTER(self._BLENDFUNCTION), wt.DWORD), wt.BOOL)
        bind(u, "PeekMessageW", (ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT), wt.BOOL)
        bind(u, "TranslateMessage", (ct.POINTER(wt.MSG),), wt.BOOL)
        bind(u, "DispatchMessageW", (ct.POINTER(wt.MSG),), ct.c_ssize_t)
        bind(g, "CreateCompatibleDC", (wt.HDC,), wt.HDC)
        bind(g, "DeleteDC", (wt.HDC,), wt.BOOL)
        bind(g, "CreateDIBSection", (wt.HDC, ct.POINTER(self._BITMAPINFO), wt.UINT,
                                     ct.POINTER(wt.LPVOID), wt.HANDLE, wt.DWORD), wt.HBITMAP)
        bind(g, "SelectObject", (wt.HDC, wt.HANDLE), wt.HANDLE)
        bind(g, "DeleteObject", (wt.HANDLE,), wt.BOOL)
        if hasattr(u, "SetThreadDpiAwarenessContext"):
            bind(u, "SetThreadDpiAwarenessContext", (ct.c_void_p,), ct.c_void_p)
        if hasattr(u, "GetDpiForWindow"):
            bind(u, "GetDpiForWindow", (wt.HWND,), wt.UINT)

    def _check(self, result, operation):
        if not result:
            error = self._ct.get_last_error()
            raise OSError(error, f"{operation}: {self._ct.FormatError(error)}")
        return result

    def _window_proc(self, hwnd, message, wparam, lparam):
        try:
            if message == 0x0084:           # WM_NCHITTEST (layered style also passes through).
                return -1
            if message == 0x0021:           # WM_MOUSEACTIVATE: MA_NOACTIVATE.
                return 3
            if message == 0x02E0:           # WM_DPICHANGED: redraw at the cursor hotspot.
                self._dirty = True
                return 0
            if message == 0x0082:           # WM_NCDESTROY.
                self._hwnd = None
            return self._user32.DefWindowProcW(hwnd, message, wparam, lparam)
        except Exception as exc:
            self._callback_error = exc
            return 0

    def _set_image(self, dpi: int, pressed: bool) -> None:
        key = (dpi, pressed)
        if self._image_key == key:
            return
        if key not in self._images:
            image = make_halo_image(dpi, pressed)
            self._images[key] = (image.width, premultiplied_bgra(image))
        size, pixels = self._images[key]
        ct, wt = self._ct, self._wt
        if self._bitmap is None or size != self._size:
            info = self._BITMAPINFO()
            info.bmiHeader = self._BITMAPINFOHEADER(ct.sizeof(self._BITMAPINFOHEADER), size, -size,
                                                    1, 32, 0, size * size * 4, 0, 0, 0, 0)
            bits = wt.LPVOID()
            bitmap = self._check(self._gdi32.CreateDIBSection(
                self._dc, ct.byref(info), 0, ct.byref(bits), None, 0), "CreateDIBSection")
            previous = self._gdi32.SelectObject(self._dc, bitmap)
            if not previous or previous == ct.c_void_p(-1).value:
                self._gdi32.DeleteObject(bitmap)
                self._check(0, "SelectObject")
            if self._old_bitmap is None:
                self._old_bitmap = previous
            if self._bitmap is not None:
                self._gdi32.DeleteObject(self._bitmap)
            self._bitmap, self._bits, self._size = bitmap, bits, size
        ct.memmove(self._bits, pixels, len(pixels))
        self._image_key = key

    def show(self, position, pressed: bool) -> None:
        ct, wt = self._ct, self._wt
        x, y = map(int, position)
        self._position, self._pressed = (x, y), pressed
        # Move while hidden first so GetDpiForWindow reflects the target monitor.
        self._check(self._user32.SetWindowPos(self._hwnd, wt.HWND(-1),
                    x - self._size // 2, y - self._size // 2, 0, 0, 0x0001 | 0x0010), "SetWindowPos")
        get_dpi = getattr(self._user32, "GetDpiForWindow", None)
        dpi = (get_dpi(self._hwnd) if get_dpi else 96) or 96
        self._set_image(dpi, pressed)
        destination = wt.POINT(x - self._size // 2, y - self._size // 2)
        size, source = wt.SIZE(self._size, self._size), wt.POINT(0, 0)
        blend = self._BLENDFUNCTION(0, 0, 255, 1)
        self._check(self._user32.UpdateLayeredWindow(self._hwnd, None, ct.byref(destination),
                    ct.byref(size), self._dc, ct.byref(source), 0, ct.byref(blend), 2), "UpdateLayeredWindow")
        if not self._visible:
            self._check(self._user32.SetWindowPos(self._hwnd, wt.HWND(-1), 0, 0, 0, 0,
                        0x0001 | 0x0002 | 0x0010 | 0x0040), "show cursor halo")
            self._visible = True
        self._dirty = False

    def hide(self) -> None:
        if self._hwnd:
            self._user32.ShowWindow(self._hwnd, 0)
        self._visible = False

    def pump(self) -> bool:
        message = self._wt.MSG()
        for _ in range(100):
            if not self._user32.PeekMessageW(self._ct.byref(message), None, 0, 0, 1):
                break
            if message.message == 0x0012:      # WM_QUIT, for this worker only.
                return False
            self._user32.TranslateMessage(self._ct.byref(message))
            self._user32.DispatchMessageW(self._ct.byref(message))
        if self._callback_error is not None:
            raise self._callback_error
        if self._dirty and self._visible and self._position is not None:
            self.show(self._position, self._pressed)
        return bool(self._hwnd)

    def close(self) -> None:
        # Every cleanup runs even if a previous native step fails.
        failure = None
        steps = []
        if self._hwnd:
            hwnd, self._hwnd = self._hwnd, None
            steps.append((self._user32.DestroyWindow, (hwnd,), "DestroyWindow"))
        if self._dc and self._old_bitmap:
            steps.append((self._gdi32.SelectObject, (self._dc, self._old_bitmap), "restore bitmap"))
        if self._dc:
            steps.append((self._gdi32.DeleteDC, (self._dc,), "DeleteDC"))
        # Deleting the memory DC also deselects its bitmap if restoration failed.
        if self._bitmap:
            steps.append((self._gdi32.DeleteObject, (self._bitmap,), "DeleteObject"))
        if self._atom:
            steps.append((self._user32.UnregisterClassW, (self._class_name, self._instance), "UnregisterClassW"))
        if self._old_dpi_context:
            steps.append((self._user32.SetThreadDpiAwarenessContext, (self._old_dpi_context,), "restore DPI"))
        self._bitmap = self._dc = self._old_bitmap = self._atom = self._old_dpi_context = None
        for function, arguments, operation in steps:
            try:
                self._check(function(*arguments), operation)
            except Exception as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure
