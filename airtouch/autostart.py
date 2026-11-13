"""Start with Windows: one per-user value in the registry's Run key (no admin needed).

    HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
        AirTouch = "<venv>\\Scripts\\pythonw.exe" "<project>\\airtouch_tray.pyw"

All registry access goes through the module-level ``backend`` object, so tests
can swap in a fake and never touch the real Run key. Off Windows ``backend`` is
None and everything is a no-op returning False.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import PROJECT_ROOT

log = logging.getLogger("airtouch.autostart")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "AirTouch"


class WinregBackend:
    """String values under one HKEY_CURRENT_USER key."""

    def __init__(self, subkey: str = RUN_KEY):
        self.subkey = subkey

    def get(self, name: str) -> str | None:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.subkey) as key:
                return winreg.QueryValueEx(key, name)[0]
        except FileNotFoundError:
            return None

    def set(self, name: str, value: str) -> None:
        import winreg

        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, self.subkey, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)

    def delete(self, name: str) -> None:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.subkey, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, name)
        except FileNotFoundError:
            pass


backend = WinregBackend() if sys.platform == "win32" else None


def build_command(executable: str | Path | None = None, root: str | Path = PROJECT_ROOT) -> str:
    """The command Windows runs at log-on: pythonw.exe next to this Python (the venv's).

    Not resolved on purpose: a venv's python.exe is what selects the venv.
    """
    pythonw = Path(executable or sys.executable).with_name("pythonw.exe")
    return f'"{pythonw}" "{Path(root) / "airtouch_tray.pyw"}"'


def is_enabled() -> bool:
    """True only if the entry starts THIS copy: a value left behind by a moved
    folder would do nothing at log-on, so it shows as off and re-ticking repairs it."""
    if backend is None:
        return False
    try:
        current = backend.get(VALUE_NAME)
    except OSError:
        log.exception("could not read the autostart entry")
        return False
    return isinstance(current, str) and current.casefold() == build_command().casefold()


def enable() -> bool:
    """Returns True if the entry was written."""
    if backend is None:
        return False
    try:
        backend.set(VALUE_NAME, build_command())
    except OSError:
        log.exception("could not write the autostart entry")
        return False
    return True


def disable() -> bool:
    """Returns True if the entry is gone (also when there was none)."""
    if backend is None:
        return False
    try:
        backend.delete(VALUE_NAME)
    except OSError:
        log.exception("could not remove the autostart entry")
        return False
    return True


def set_enabled(on: bool) -> bool:
    return enable() if on else disable()
