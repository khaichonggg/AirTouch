"""Native stderr ownership regressions; child processes never load MediaPipe."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from airtouch.config import PROJECT_ROOT


def run_child(source: str):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(source)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
    return json.loads(done.stdout), done.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native HANDLE lifetime")
@pytest.mark.parametrize("fail_inside", [False, True], ids=["normal", "exception"])
def test_native_stderr_handle_survives_initialization_scope(fail_inside):
    # A native CRT can retain this borrowed OS handle. Closing/replacing fd2
    # leaves that CRT with a dangling handle even if Python stderr is restored.
    observed, stderr = run_child(
        f"""
        import ctypes
        from ctypes import wintypes as wt
        import json
        import msvcrt
        import os
        from airtouch.hand_tracker import _quiet_native_stderr

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetHandleInformation.argtypes = (wt.HANDLE, ctypes.POINTER(wt.DWORD))
        kernel.GetHandleInformation.restype = wt.BOOL
        kernel.WriteFile.argtypes = (wt.HANDLE, wt.LPCVOID, wt.DWORD,
                                     ctypes.POINTER(wt.DWORD), wt.LPVOID)
        kernel.WriteFile.restype = wt.BOOL
        cached = msvcrt.get_osfhandle(2)
        stages = []

        def sample(stage):
            flags = wt.DWORD()
            error = None
            try:
                valid = bool(kernel.GetHandleInformation(cached, ctypes.byref(flags)))
            except OSError as exception:
                # Some Windows execution environments raise a structured
                # invalid-handle exception instead of returning FALSE.
                valid, error = False, repr(exception)
            marker = ("native-" + stage + "\\n").encode("ascii")
            written = wt.DWORD()
            # Do not send an invalid handle to WriteFile: validity itself is
            # sufficient evidence of the broken lifetime on the old code.
            write_ok = valid and bool(kernel.WriteFile(
                cached, marker, len(marker), ctypes.byref(written), None))
            stages.append({{"stage": stage, "handle": msvcrt.get_osfhandle(2),
                           "cached_valid": valid, "write_ok": write_ok,
                           "written": written.value, "error": error}})

        sample("before")
        try:
            with _quiet_native_stderr():
                sample("inside")
                if {fail_inside!r}:
                    raise ValueError("initialization failed")
        except ValueError:
            pass
        sample("after")
        os.write(2, b"python-after\\n")
        print(json.dumps({{"cached": cached, "stages": stages}}))
        """
    )
    for stage in observed["stages"]:
        details = json.dumps(stage, sort_keys=True)
        assert stage["cached_valid"], details
        assert stage["handle"] == observed["cached"], details
        assert stage["write_ok"], details
    assert stderr == b"native-before\nnative-inside\nnative-after\npython-after\n"


def test_overlapping_initialization_scopes_preserve_later_stderr():
    # The old context restored A's null sink after A had restored the original
    # stream. Real descriptor writes after both scopes must remain observable.
    observed, stderr = run_child(
        """
        import json
        import os
        import threading
        from airtouch.hand_tracker import _quiet_native_stderr

        entered_a = threading.Event()
        entered_b = threading.Event()
        release_a = threading.Event()
        release_b = threading.Event()
        ended_a = threading.Event()
        errors = []

        def worker(entered, release, ended=None):
            try:
                with _quiet_native_stderr():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("scope release timed out")
            except BaseException as error:
                errors.append(repr(error))
            finally:
                if ended is not None:
                    ended.set()

        first = threading.Thread(target=worker, args=(entered_a, release_a, ended_a), daemon=True)
        second = threading.Thread(target=worker, args=(entered_b, release_b), daemon=True)
        first.start()
        try:
            if not entered_a.wait(5):
                raise RuntimeError("first scope did not enter")
            second.start()
            # A locking implementation may legitimately delay B. Release A
            # regardless, so this regression also permits safe serialization.
            overlap = entered_b.wait(0.5)
            release_a.set()
            if not ended_a.wait(5):
                raise RuntimeError("first scope did not exit")
            release_b.set()
            first.join(5)
            second.join(5)
            if first.is_alive() or second.is_alive():
                raise RuntimeError("initialization scope deadlock")
        finally:
            release_a.set()
            release_b.set()
        os.write(2, b"stderr-after-overlap\\n")
        print(json.dumps({"errors": errors, "overlap": overlap}))
        """
    )
    assert observed["errors"] == []
    assert stderr == b"stderr-after-overlap\n", observed


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native HANDLE lifetime")
def test_project_pytest_defaults_preserve_borrowed_native_stderr(tmp_path):
    # Exercise the project's actual pytest configuration. Default fd capture
    # closes the process's borrowed native HANDLE before the first test runs.
    test_file = tmp_path / "test_empty.py"
    test_file.write_text("def test_empty():\n    assert True\n", encoding="utf-8")
    observed, _stderr = run_child(
        f"""
        import contextlib, ctypes, io, json, msvcrt
        from ctypes import wintypes as wt
        from pathlib import Path
        import pytest

        cached = msvcrt.get_osfhandle(2)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetHandleInformation.argtypes = (wt.HANDLE, ctypes.POINTER(wt.DWORD))
        kernel.GetHandleInformation.restype = wt.BOOL
        state = {{}}

        class Probe:
            def pytest_sessionstart(self, session):
                state["same_handle"] = msvcrt.get_osfhandle(2) == cached
                flags = wt.DWORD()
                try:
                    state["valid"] = bool(kernel.GetHandleInformation(cached, ctypes.byref(flags)))
                except OSError as error:
                    state["valid"] = False
                    state["error"] = repr(error)

        root = Path({str(PROJECT_ROOT)!r})
        args = ["-q", "-p", "no:cacheprovider", "--rootdir", str(root),
                "--basetemp", {str(tmp_path / 'child-temp')!r}, {str(test_file)!r}]
        config = root / "pytest.ini"
        if config.is_file():
            args += ["-c", str(config)]
        with contextlib.redirect_stdout(io.StringIO()):
            code = pytest.main(args, plugins=[Probe()])
        print(json.dumps({{"exit": code, "state": state}}))
        """
    )
    assert observed["exit"] == 0, observed
    assert observed["state"]["same_handle"] and observed["state"]["valid"], observed
