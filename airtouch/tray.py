"""The tray icon: an on/off switch, so gesture control only uses power when wanted.

Left-click the icon to switch. ON (teal) runs the gesture loop; OFF (grey) ends
it and closes the camera. It always starts OFF, so starting with Windows just
parks the icon. Run ``airtouch_tray.pyw`` (no console) or ``python -m airtouch.tray``.

pystray is imported only when the menu or icon is really built, and nothing here
needs OpenCV or MediaPipe until the switch is turned on.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from PIL import Image, ImageDraw

from . import autostart
from .config import PROJECT_ROOT
from .console import use_utf8_console
from .session import ERROR_PREFIX, CaptureWorker, plain_message

log = logging.getLogger("airtouch.tray")

APP_NAME = "AirTouch"
MUTEX_NAME = "Local\\AirTouchTraySingleInstance"
ERROR_ALREADY_EXISTS = 183

TEXT_TURN_ON = "开启 / Turn ON"
TEXT_TURN_OFF = "关闭 / Turn OFF"
TEXT_AUTOSTART = "开机自动启动 / Start with Windows"
TEXT_QUIT = "退出 / Quit"
TITLE_ON = "AirTouch：已开启，可以用手势控制鼠标"
TITLE_OFF = "AirTouch：已关闭，摄像头没有在用"
MSG_AUTOSTART_FAILED = "开机自动启动没有设置成功，请再试一次。"
MAX_NOTIFICATION_CHARS = 250        # Windows limits a notification to 255 characters

COLOR_ON = (13, 148, 136, 255)      # teal, readable on light and dark taskbars
COLOR_OFF = (128, 134, 145, 255)    # grey

# The hand, in fractions of the icon. Each digit is a rounded bar:
# (base x, base y, tip x, tip y, thickness). The four long fingers fan out and
# are nearly the same length, so all five stay separate even at 16 px and no
# single finger ever stands out.
_DIGITS = (
    (0.400, 0.60, 0.305, 0.135, 0.108),     # index
    (0.525, 0.58, 0.505, 0.070, 0.108),     # middle
    (0.645, 0.58, 0.700, 0.105, 0.108),     # ring
    (0.755, 0.62, 0.895, 0.205, 0.100),     # little finger
    (0.355, 0.76, 0.090, 0.475, 0.108),     # thumb: shorter, angled out
)
_PALM = (0.325, 0.52, 0.82, 0.96)
_PALM_RADIUS = 0.13
_SUPERSAMPLE = 8


def make_icon(on: bool, size: int = 64) -> Image.Image:
    """An open hand with five separate fingers: teal when ON, grey when OFF."""
    canvas = size * _SUPERSAMPLE
    color = COLOR_ON if on else COLOR_OFF
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for base_x, base_y, tip_x, tip_y, thickness in _DIGITS:
        ends = ((base_x * canvas, base_y * canvas), (tip_x * canvas, tip_y * canvas))
        radius = thickness * canvas / 2
        draw.line(ends, fill=color, width=round(thickness * canvas))
        for x, y in ends:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.rounded_rectangle(tuple(v * canvas for v in _PALM), radius=_PALM_RADIUS * canvas, fill=color)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def save_previews(folder: str | Path = PROJECT_ROOT / "docs" / "images", size: int = 256) -> None:
    """Write tray_on.png and tray_off.png for the documentation."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    make_icon(True, size).save(folder / "tray_on.png")
    make_icon(False, size).save(folder / "tray_off.png")


# ------------------------------------------------------------- single instance
def _create_mutex_win32(name: str) -> tuple[object, bool]:
    """CreateMutexW: returns (handle, somebody else already holds this name)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    handle = kernel32.CreateMutexW(None, False, name)
    return handle, ctypes.get_last_error() == ERROR_ALREADY_EXISTS


def _close_handle_win32(handle) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle(handle)


class SingleInstance:
    """Only one tray icon, however it was launched (autostart + a manual double-click).

    A named mutex: the first process to create it wins, and Windows frees it
    when that process ends, even after a crash. Always succeeds off Windows.
    """

    def __init__(self, name: str = MUTEX_NAME, create_mutex=None, close_handle=None):
        on_windows = sys.platform == "win32"
        self.name = name
        self._create_mutex = create_mutex or (_create_mutex_win32 if on_windows else None)
        self._close_handle = close_handle or (_close_handle_win32 if on_windows else None)
        self._handle = None
        self._acquired = False

    def acquire(self) -> bool:
        """True if this is the only instance. Keep this object alive while running."""
        if self._create_mutex is None:
            return True
        if self._handle:
            return self._acquired
        self._handle, already_running = self._create_mutex(self.name)
        if not self._handle:
            raise OSError("无法创建 AirTouch 的运行锁，请关闭程序后重试。")
        self._acquired = not already_running
        return self._acquired

    def release(self) -> None:
        if self._handle and self._close_handle is not None:
            self._close_handle(self._handle)
        self._handle = None
        self._acquired = False


def _make_dpi_aware() -> None:
    """The mouse backend does this too, but only at switch-on. Doing it before the
    icon creates its windows means they and the cursor coordinates agree from the
    start on a scaled display, and Windows picks a sharp icon size."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)     # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            log.exception("could not make the process DPI-aware")


# --------------------------------------------------------------------- the app
class TrayApp:
    def __init__(self, worker=None, cfg=None):
        self._generation = 0
        self.worker = worker if worker is not None else CaptureWorker(cfg=cfg)
        self.worker.on_status = self._on_status
        self.on = False
        self.icon = None        # the pystray.Icon once run() has created it
        self._images = {True: make_icon(True), False: make_icon(False)}
        self._lock = threading.RLock()      # the menu and the worker thread both change the state

    @property
    def title(self) -> str:
        return TITLE_ON if self.on else TITLE_OFF

    def build_menu(self):
        import pystray

        return pystray.Menu(
            pystray.MenuItem(lambda _item: TEXT_TURN_OFF if self.on else TEXT_TURN_ON, self.toggle, default=True),
            pystray.MenuItem(TEXT_AUTOSTART, self.toggle_autostart, checked=lambda _item: autostart.is_enabled()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(TEXT_QUIT, self.quit),
        )

    def run(self) -> None:
        """Show the icon; blocks until Quit. Must be called on the main thread."""
        import pystray

        self.icon = pystray.Icon(APP_NAME, self._images[self.on], self.title, self.build_menu())
        try:
            self.icon.run()
        finally:
            self.worker.stop()

    # -- menu actions (pystray calls these on its message loop) -----------------
    def toggle(self, _icon=None, _item=None) -> None:
        self.set_on(not self.on)

    def set_on(self, on: bool) -> None:
        with self._lock:
            if on == self.on:
                return
            self._generation += 1
            self.on = on
            if on:
                generation = self._generation
                self.worker.on_status = lambda message: self._on_status(message, generation)
                try:
                    self.worker.start()
                except Exception as error:
                    log.exception("could not start capture worker")
                    self._on_status(ERROR_PREFIX + plain_message(error), generation)
                    return
                except BaseException:
                    self._generation += 1
                    self.on = False
                    self._refresh()
                    raise
            else:
                # Don't wait: the menu must not freeze while the camera closes
                # (the worker finishes that by itself, and a quick ON waits for it).
                self.worker.stop(timeout=0)
            self._refresh()

    def toggle_autostart(self, _icon=None, _item=None) -> None:
        wanted = not autostart.is_enabled()
        autostart.set_enabled(wanted)
        if autostart.is_enabled() != wanted:
            self._notify(MSG_AUTOSTART_FAILED)
        self._refresh()

    def quit(self, _icon=None, _item=None) -> None:
        with self._lock:
            self._generation += 1
            self.on = False
        self.worker.stop()      # without the lock: the worker's last message may need it
        if self.icon is not None:
            self.icon.stop()

    # -- from the worker thread --------------------------------------------------
    def _on_status(self, message: str, generation: int | None = None) -> None:
        log.info("%s", message)
        failed = message.startswith(ERROR_PREFIX)
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            if self.on and (failed or not self.worker.is_running):
                self.on = False
                self._refresh()
            if failed:
                self._notify(message[len(ERROR_PREFIX):])

    # -- display -------------------------------------------------------------------
    def _refresh(self) -> None:
        if self.icon is None:
            return
        try:
            self.icon.icon = self._images[self.on]
            self.icon.title = self.title
            self.icon.update_menu()
        except Exception:
            log.exception("could not update the tray icon")

    def _notify(self, message: str) -> None:
        if self.icon is None or not getattr(self.icon, "HAS_NOTIFICATION", False):
            return
        try:
            self.icon.notify(message[:MAX_NOTIFICATION_CHARS], APP_NAME)
        except Exception:
            log.exception("could not show a notification")


def main() -> int:
    use_utf8_console()
    guard = SingleInstance()
    try:
        if not guard.acquire():
            log.info("already running; nothing to do")
            return 0
        _make_dpi_aware()
        TrayApp().run()
        return 0
    finally:
        guard.release()


if __name__ == "__main__":
    sys.exit(main())
