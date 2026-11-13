"""Start-with-Windows entry. Uses a fake registry: the real Run key is never written.

(One test exercises the real winreg code against a throw-away key of its own,
which it deletes again. It refuses to run against the Run key.)
"""

from __future__ import annotations

import shlex
import sys
import uuid

import pytest

from airtouch import autostart
from airtouch.autostart import RUN_KEY, VALUE_NAME, WinregBackend
from airtouch.config import PROJECT_ROOT

WEIRD_ROOT = r"C:\Users\Example User\Projects 【测试】\AirTouch"
REAL_BACKEND = autostart.backend        # captured before any fixture swaps it out


class FakeRegistry:
    def __init__(self, values=None, fail: bool = False):
        self.values = dict(values or {})
        self.fail = fail

    def _check(self):
        if self.fail:
            raise PermissionError("access denied")

    def get(self, name):
        self._check()
        return self.values.get(name)

    def set(self, name, value):
        self._check()
        self.values[name] = value

    def delete(self, name):
        self._check()
        self.values.pop(name, None)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Every test in this file gets a fake registry, so nothing can reach the real one."""
    fake = FakeRegistry()
    monkeypatch.setattr(autostart, "backend", fake)
    return fake


# ------------------------------------------------------------------ build_command
def test_command_uses_pythonw_next_to_this_python():
    command = autostart.build_command()
    python_dir = sys.executable.rsplit("\\", 1)[0]
    assert command.startswith(f'"{python_dir}\\pythonw.exe" ')


def test_command_runs_the_tray_script_of_this_project():
    command = autostart.build_command()
    assert command.endswith(f' "{PROJECT_ROOT / "airtouch_tray.pyw"}"')
    assert (PROJECT_ROOT / "airtouch_tray.pyw").exists()


def test_command_has_no_console_python():
    assert "python.exe" not in autostart.build_command(r"C:\x\python.exe", r"C:\y")


def test_command_quotes_spaces_and_non_ascii_paths():
    command = autostart.build_command(rf"{WEIRD_ROOT}\.venv\Scripts\python.exe", WEIRD_ROOT)
    assert command == f'"{WEIRD_ROOT}\\.venv\\Scripts\\pythonw.exe" "{WEIRD_ROOT}\\airtouch_tray.pyw"'
    parts = shlex.split(command, posix=False)
    assert len(parts) == 2 and all(p.startswith('"') and p.endswith('"') for p in parts)


def test_command_is_the_same_when_already_running_under_pythonw():
    from_pythonw = autostart.build_command(r"C:\venv\Scripts\pythonw.exe", r"C:\p")
    from_python = autostart.build_command(r"C:\venv\Scripts\python.exe", r"C:\p")
    assert from_pythonw == from_python


# ------------------------------------------------------- enable / disable / state
def test_starts_disabled():
    assert autostart.is_enabled() is False


def test_enable_writes_the_command_under_the_agreed_name(registry):
    assert autostart.enable() is True
    assert registry.values == {"AirTouch": autostart.build_command()}
    assert autostart.is_enabled() is True


def test_disable_removes_only_our_value(registry):
    registry.values["SomeOtherApp"] = "other.exe"
    autostart.enable()
    assert autostart.disable() is True
    assert registry.values == {"SomeOtherApp": "other.exe"}
    assert autostart.is_enabled() is False


def test_disable_when_not_enabled_is_fine():
    assert autostart.disable() is True
    assert autostart.is_enabled() is False


def test_enable_twice_is_fine(registry):
    autostart.enable()
    autostart.enable()
    assert list(registry.values) == [VALUE_NAME]


@pytest.mark.parametrize("on", [True, False])
def test_set_enabled(on):
    assert autostart.set_enabled(on) is True
    assert autostart.is_enabled() is on


def test_entry_left_by_a_moved_folder_shows_as_off_and_can_be_repaired(registry):
    registry.values[VALUE_NAME] = r'"D:\old place\pythonw.exe" "D:\old place\airtouch_tray.pyw"'
    assert autostart.is_enabled() is False
    autostart.enable()
    assert autostart.is_enabled() is True
    assert registry.values[VALUE_NAME] == autostart.build_command()


def test_path_comparison_ignores_case_like_windows_does(registry):
    registry.values[VALUE_NAME] = autostart.build_command().upper()
    assert autostart.is_enabled() is True


def test_value_of_the_wrong_type_is_not_enabled(registry):
    registry.values[VALUE_NAME] = 1
    assert autostart.is_enabled() is False


def test_registry_errors_never_crash(registry):
    registry.fail = True
    assert autostart.is_enabled() is False
    assert autostart.enable() is False
    assert autostart.disable() is False


def test_without_a_registry_everything_is_a_harmless_false(monkeypatch):
    monkeypatch.setattr(autostart, "backend", None)
    assert autostart.is_enabled() is False
    assert autostart.enable() is False
    assert autostart.disable() is False
    assert autostart.set_enabled(True) is False
    assert autostart.set_enabled(False) is False


def test_module_default_backend_targets_the_run_key():
    if sys.platform == "win32":
        assert isinstance(REAL_BACKEND, WinregBackend)
        assert REAL_BACKEND.subkey == RUN_KEY == r"Software\Microsoft\Windows\CurrentVersion\Run"
    else:
        assert REAL_BACKEND is None


# -------------------------------------------------------------- real winreg code
@pytest.mark.skipif(sys.platform != "win32", reason="Windows registry")
def test_winreg_backend_round_trip_on_a_throwaway_key():
    import winreg

    parent = r"Software\AirTouchTest"
    subkey = rf"{parent}\{uuid.uuid4().hex}"
    assert "CurrentVersion\\Run" not in subkey
    backend = WinregBackend(subkey)
    value = f'"{WEIRD_ROOT}\\pythonw.exe" "{WEIRD_ROOT}\\airtouch_tray.pyw"'
    try:
        assert backend.get(VALUE_NAME) is None          # key doesn't exist yet
        backend.delete(VALUE_NAME)                      # nothing to delete: still fine
        backend.set(VALUE_NAME, value)
        assert backend.get(VALUE_NAME) == value
        assert backend.get("NoSuchValue") is None
        backend.delete("NoSuchValue")
        backend.delete(VALUE_NAME)
        assert backend.get(VALUE_NAME) is None
    finally:
        for key in (subkey, parent):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
            except OSError:
                pass
