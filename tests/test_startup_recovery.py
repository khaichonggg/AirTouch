"""Launcher failure paths use fake preparation and notification, never the OS UI."""

from pathlib import Path
import runpy

import pytest

from airtouch import tray


@pytest.fixture
def launcher(monkeypatch):
    # Loading definitions under a non-main name does not open logs or run the app.
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "airtouch_tray.pyw"))
    globals_ = namespace["main"].__globals__
    calls = []
    monkeypatch.setattr(globals_["os"], "chdir", lambda path: calls.append("chdir"))
    monkeypatch.setitem(globals_, "redirect_output", lambda: calls.append("redirect"))
    monkeypatch.setitem(globals_, "show_error", lambda message: calls.append(("error", message)))
    monkeypatch.setattr(globals_["logging"], "exception", lambda message: calls.append("log"))
    monkeypatch.setattr(tray, "main", lambda: calls.append("tray") or 0)
    # main adds its root to sys.path; leave the process's real list untouched.
    class FakeSys:
        path = []
    monkeypatch.setitem(globals_, "sys", FakeSys())
    return globals_, calls


@pytest.mark.parametrize("stage", ["chdir", "redirect_output"])
def test_preparation_failure_returns_error_and_notifies_instead_of_escaping(launcher, monkeypatch, stage):
    globals_, calls = launcher

    def denied(*args):
        raise PermissionError("startup preparation denied")

    if stage == "chdir":
        monkeypatch.setattr(globals_["os"], "chdir", denied)
    else:
        monkeypatch.setitem(globals_, stage, denied)

    assert globals_["main"]() == 1
    assert "tray" not in calls
    errors = [entry[1] for entry in calls if isinstance(entry, tuple) and entry[0] == "error"]
    assert len(errors) == 1
    assert "startup preparation denied" in errors[0]


def test_prepared_launcher_keeps_success_or_duplicate_start_exit_code(launcher):
    globals_, calls = launcher
    assert globals_["main"]() == 0
    assert calls == ["chdir", "redirect", "tray"]
