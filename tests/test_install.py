"""The installer and the .bat files.

Nothing is really installed. The .bat files are run for real, but only copies in
temporary folders with awkward names, with stand-ins for Python and for the app:
the project's real hand_mouse.py / tray never start, so no camera and no mouse.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import install
from airtouch.config import PROJECT_ROOT

BAT_FILES = ("run.bat", "tray.bat", "check.bat", "setup.bat")
AWKWARD_NAME = "AirTouch 测试 【 Final Year Project 】 v2"
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


# ------------------------------------------------------------------- venv_python
def test_venv_python_on_windows():
    assert install.venv_python("C:/x y", platform="win32") == Path("C:/x y/.venv/Scripts/python.exe")


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_venv_python_elsewhere(platform):
    assert install.venv_python("/x y", platform=platform).as_posix().endswith("/x y/.venv/bin/python")


def test_venv_python_default_is_this_project():
    assert install.venv_python().parent.parent == PROJECT_ROOT / ".venv"


# ------------------------------------------------------------------------- steps
def titles_and_commands(plan):
    return [step.command for step in plan]


def test_fresh_folder_gets_all_three_steps_in_order(tmp_path):
    plan = install.steps(tmp_path, python="BASEPY")
    venv_py = str(install.venv_python(tmp_path))
    assert titles_and_commands(plan) == [
        ("BASEPY", "-m", "venv", str(tmp_path / ".venv")),
        (venv_py, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(tmp_path / "requirements.txt")),
        (venv_py, "-m", "airtouch.download_model"),
    ]


def test_every_step_explains_itself_in_chinese(tmp_path):
    for step in install.steps(tmp_path):
        for text in (step.title, step.failure):
            assert any("\u4e00" <= ch <= "\u9fff" for ch in text)


def test_skip_model_leaves_out_the_download(tmp_path):
    commands = titles_and_commands(install.steps(tmp_path, skip_model=True))
    assert len(commands) == 2
    assert not any("airtouch.download_model" in c for c in commands)


def test_working_venv_of_this_python_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "venv_version", lambda root: tuple(sys.version_info[:2]))
    commands = titles_and_commands(install.steps(tmp_path))
    assert not any("venv" in c for c in commands)
    assert len(commands) == 2


def test_broken_venv_is_rebuilt_clean(tmp_path, monkeypatch):
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(install, "venv_version", lambda root: None)
    first = install.steps(tmp_path, python="BASEPY")[0].command
    assert first == ("BASEPY", "-m", "venv", "--clear", str(tmp_path / ".venv"))


def test_venv_from_another_python_version_is_rebuilt_clean(tmp_path, monkeypatch):
    """Otherwise someone who first tried a too-new Python stays stuck after installing 3.12."""
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(install, "venv_version", lambda root: (3, 99))
    assert "--clear" in install.steps(tmp_path)[0].command


def test_venv_version_of_a_missing_venv_is_none(tmp_path):
    assert install.venv_version(tmp_path) is None


def test_venv_version_of_a_broken_python_is_none(tmp_path):
    python = install.venv_python(tmp_path)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"this is not a program")
    assert install.venv_version(tmp_path) is None


def test_venv_version_of_the_real_venv_works():
    assert install.venv_version(PROJECT_ROOT) == tuple(sys.version_info[:2])


# ------------------------------------------------------------------------- main()
class Recorder:
    def __init__(self, failing_word: str | None = None, code: int = 7):
        self.commands: list[list[str]] = []
        self.cwd: list[str] = []
        self.failing_word = failing_word
        self.code = code

    def __call__(self, command, cwd=None):
        self.commands.append(command)
        self.cwd.append(cwd)
        return self.code if self.failing_word and self.failing_word in " ".join(command) else 0


def test_dry_run_prints_the_steps_and_does_nothing(tmp_path, capsys):
    runner = Recorder()
    assert install.main(["--dry-run"], root=tmp_path, run=runner) == 0
    out = capsys.readouterr().out
    assert runner.commands == []
    assert not (tmp_path / ".venv").exists()
    assert "预演" in out
    assert "-m venv" in out and "pip install" in out and "airtouch.download_model" in out


def test_dry_run_with_skip_model(tmp_path, capsys):
    assert install.main(["--dry-run", "--skip-model"], root=tmp_path, run=Recorder()) == 0
    out = capsys.readouterr().out
    assert "pip install" in out and "airtouch.download_model" not in out


def test_full_run_executes_every_step_in_the_project_folder(tmp_path, capsys):
    runner = Recorder()
    assert install.main([], root=tmp_path, run=runner) == 0
    assert [c[1:3] for c in runner.commands] == [["-m", "venv"], ["-m", "pip"], ["-m", "airtouch.download_model"]]
    assert set(runner.cwd) == {str(tmp_path)}
    out = capsys.readouterr().out
    if sys.platform == "win32":
        assert "run.bat" in out and "tray.bat" in out and "check.bat" in out


@pytest.mark.parametrize("word, message, done_steps", [
    ("venv", install.FAIL_VENV, 1),
    ("pip", install.FAIL_PIP, 2),
    ("airtouch.download_model", install.FAIL_MODEL, 3),
])
def test_failure_stops_with_a_plain_message_and_its_exit_code(tmp_path, capsys, word, message, done_steps):
    # tmp_path can itself contain these words only by accident: it is under pytest-of-*/
    runner = Recorder(failing_word=f"-m {word}")
    assert install.main([], root=tmp_path, run=runner) == 7
    assert len(runner.commands) == done_steps
    out = capsys.readouterr().out
    assert message in out and "全部装好了" not in out


def test_a_missing_program_is_a_normal_failure(tmp_path, capsys):
    def run(_command, cwd=None):
        raise FileNotFoundError("no such python")

    assert install.main([], root=tmp_path, run=run) == 1
    assert install.FAIL_VENV in capsys.readouterr().out


def test_too_old_python_is_refused_kindly(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(install, "MIN_PYTHON", (99, 0))
    runner = Recorder()
    assert install.main([], root=tmp_path, run=runner) == 1
    assert runner.commands == []
    assert "太旧" in capsys.readouterr().out


def test_dry_run_as_a_real_program_is_readable_utf8():
    done = subprocess.run([sys.executable, str(PROJECT_ROOT / "install.py"), "--dry-run"],
                          cwd=PROJECT_ROOT, capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr
    out = done.stdout.decode("utf-8")
    assert "预演" in out and "pip install" in out


def test_requirements_file_is_present_and_has_the_core_libraries():
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for name in ("mediapipe", "opencv-contrib-python", "pystray", "Pillow", "scikit-learn"):
        assert name in text


# ------------------------------------------------------------- .bat file encoding
@pytest.mark.parametrize("name", BAT_FILES)
def test_bat_is_utf8_without_bom_and_crlf(name):
    raw = (PROJECT_ROOT / name).read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8")
    assert b"\r\n" in raw
    assert raw.count(b"\n") == raw.count(b"\r\n"), "bare LF"
    assert raw.count(b"\r") == raw.count(b"\r\n"), "bare CR"
    assert raw.endswith(b"\r\n")


@pytest.mark.parametrize("name", BAT_FILES)
def test_bat_prepares_the_console_and_folder_before_anything_else(name):
    lines = [
        line for line in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lower().startswith("rem ")
    ]
    assert lines[0].lower() == "@echo off"
    assert lines[1] == "chcp 65001 >nul"
    assert 'cd /d "%~dp0"' in lines[:4]
    before = "\n".join(lines[:4])
    assert before.isascii(), "Chinese before chcp would be misread"


@pytest.mark.parametrize("name", BAT_FILES)
def test_no_parenthesis_blocks_that_chinese_text_could_break(name):
    """A plain ')' inside an echo would end a ( ... ) block early, so the files use goto."""
    for line in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines():
        assert not line.rstrip().endswith(" ("), line


# ---------------------------------------------------------------- running the .bats
def run_bat(folder: Path, name: str, *args: str, env: dict | None = None, cwd: Path | None = None):
    """Run it the way a double-click does: cmd /c ""full path"", from a hidden console of
    its own (so chcp can't touch any real console) and nothing to wait on at 'pause'."""
    quoted = " ".join(f'"{a}"' if " " in a else a for a in args)
    line = f'cmd /c ""{folder / name}" {quoted}"'
    done = subprocess.run(line, cwd=cwd or folder, env=env, capture_output=True, stdin=subprocess.DEVNULL,
                          timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    return done.returncode, done.stdout.decode("utf-8", errors="replace") + done.stderr.decode("utf-8", errors="replace")


@pytest.fixture
def folder(tmp_path) -> Path:
    """A copy of the .bat files in a folder with spaces and 【 】 in its name."""
    target = tmp_path / AWKWARD_NAME
    target.mkdir()
    for name in BAT_FILES:
        shutil.copy2(PROJECT_ROOT / name, target / name)
    return target


def fake_venv(folder: Path) -> None:
    """A working .venv with a real interpreter but nothing installed, from the project's."""
    scripts = PROJECT_ROOT / ".venv" / "Scripts"
    (folder / ".venv" / "Scripts").mkdir(parents=True)
    for exe in ("python.exe", "pythonw.exe"):
        shutil.copy2(scripts / exe, folder / ".venv" / "Scripts" / exe)
    shutil.copy2(PROJECT_ROOT / ".venv" / "pyvenv.cfg", folder / ".venv" / "pyvenv.cfg")


REPORT_ARGS = """
import json, os, sys
print("ARGS=" + json.dumps(sys.argv[1:]))
print("CWD=" + json.dumps(os.getcwd()))
print("EXE=" + json.dumps(sys.executable))
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"""


def env_with(**extra) -> dict:
    env = dict(os.environ)
    env.update(extra)
    return env


def value_of(key: str, output: str):
    line = next(li for li in output.splitlines() if li.startswith(key + "="))
    return json.loads(line[len(key) + 1:])


@windows_only
@pytest.mark.parametrize("name", ["run.bat", "tray.bat", "check.bat"])
def test_without_a_venv_it_says_to_run_setup_first(folder, name):
    code, out = run_bat(folder, name)
    assert code == 1
    assert "setup.bat" in out and "还没有安装好" in out


@windows_only
def test_run_bat_passes_arguments_through_and_runs_in_its_own_folder(folder, tmp_path):
    fake_venv(folder)
    (folder / "hand_mouse.py").write_text(REPORT_ARGS, encoding="utf-8")
    code, out = run_bat(folder, "run.bat", "--lite", "two words", cwd=tmp_path)
    assert code == 0, out
    assert value_of("ARGS", out) == ["--lite", "two words"]
    assert Path(value_of("CWD", out)) == folder
    assert Path(value_of("EXE", out)).parent == folder / ".venv" / "Scripts"
    assert ". . ." not in out, "no pause when everything went fine"


@windows_only
def test_run_bat_pauses_on_error_and_keeps_the_exit_code(folder):
    fake_venv(folder)
    (folder / "hand_mouse.py").write_text(REPORT_ARGS, encoding="utf-8")
    code, out = run_bat(folder, "run.bat", env=env_with(FAKE_EXIT="3"))
    assert code == 3
    assert "程序出了问题" in out
    assert ". . ." in out, "pause, so the window doesn't vanish"


@windows_only
@pytest.mark.parametrize("exit_code", [0, 2])
def test_check_bat_always_pauses_and_passes_arguments(folder, exit_code):
    fake_venv(folder)
    (folder / "check_gestures.py").write_text(REPORT_ARGS, encoding="utf-8")
    code, out = run_bat(folder, "check.bat", "--quick", env=env_with(FAKE_EXIT=str(exit_code)))
    assert code == exit_code
    assert value_of("ARGS", out) == ["--quick"]
    assert ". . ." in out


@windows_only
def test_tray_bat_starts_pythonw_without_waiting(folder, tmp_path):
    fake_venv(folder)
    marker = tmp_path / "tray_started.json"
    (folder / "airtouch_tray.pyw").write_text(
        "import json, os, sys\n"
        f"open(r'{marker}', 'w', encoding='utf-8').write(json.dumps([sys.executable, os.getcwd()]))\n",
        encoding="utf-8",
    )
    code, out = run_bat(folder, "tray.bat", cwd=tmp_path)
    assert code == 0, out
    deadline = time.perf_counter() + 30
    while not marker.exists() and time.perf_counter() < deadline:
        time.sleep(0.05)
    for _ in range(100):                # the writer may still be flushing
        try:
            executable, cwd = json.loads(marker.read_text(encoding="utf-8"))
            break
        except ValueError:
            time.sleep(0.05)
    assert Path(executable).name.lower() == "pythonw.exe"
    assert Path(executable).parent == folder / ".venv" / "Scripts"
    assert Path(cwd) == folder


# setup.bat: stand-ins for Python on a PATH that hides the real ones
def shim(bin_dir: Path, name: str, body: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / f"{name}.cmd").write_bytes(("@echo off\r\n" + body.replace("\n", "\r\n") + "\r\n").encode("ascii"))


def path_env(bin_dir: Path) -> dict:
    return env_with(PATH=f"{bin_dir};{os.environ['SystemRoot']}\\System32")


# Answers the "are you a usable Python?" probe (-c), and reports being asked to run install.py.
WORKING_PYTHON = 'if "%~2"=="install.py" (echo INSTALL-RUN-BY %~n0 %1 & exit /b 0)\nif "%~1"=="install.py" (echo INSTALL-RUN-BY %~n0 & exit /b 0)\nexit /b 0'
MICROSOFT_STORE_STUB = "echo Python was not found; run without arguments to install from the Microsoft Store 1>&2\nexit /b 9009"


@windows_only
def test_setup_without_python_explains_how_to_get_it(folder, tmp_path):
    bin_dir = tmp_path / "empty_bin"
    bin_dir.mkdir()
    code, out = run_bat(folder, "setup.bat", env=path_env(bin_dir))
    assert code == 1
    assert "python.org" in out and "Python 3.12" in out and "Add python.exe to PATH" in out
    assert ". . ." in out, "pause"


@windows_only
def test_setup_skips_the_microsoft_store_stub(folder, tmp_path):
    bin_dir = tmp_path / "stub_bin"
    shim(bin_dir, "python", MICROSOFT_STORE_STUB)
    code, out = run_bat(folder, "setup.bat", env=path_env(bin_dir))
    assert code == 1
    assert "INSTALL-RUN-BY" not in out
    assert "python.org" in out
    assert "Microsoft Store" not in out, "probing must stay silent"


@windows_only
def test_setup_prefers_python_312_from_the_launcher(folder, tmp_path):
    bin_dir = tmp_path / "py_bin"
    shim(bin_dir, "py", WORKING_PYTHON)
    shim(bin_dir, "python", MICROSOFT_STORE_STUB)
    code, out = run_bat(folder, "setup.bat", env=path_env(bin_dir))
    assert code == 0, out
    assert "INSTALL-RUN-BY py -3.12" in out
    assert ". . ." in out, "pause at the end"


@windows_only
def test_setup_falls_back_to_any_python_3_from_the_launcher(folder, tmp_path):
    bin_dir = tmp_path / "py3_bin"
    shim(bin_dir, "py", 'if "%~1"=="-3.12" exit /b 103\n' + WORKING_PYTHON)
    code, out = run_bat(folder, "setup.bat", env=path_env(bin_dir))
    assert code == 0, out
    assert "INSTALL-RUN-BY py -3" in out and "-3.12" not in out


@windows_only
def test_setup_falls_back_to_python_on_path(folder, tmp_path):
    bin_dir = tmp_path / "python_bin"
    shim(bin_dir, "python", WORKING_PYTHON)
    code, out = run_bat(folder, "setup.bat", env=path_env(bin_dir))
    assert code == 0, out
    assert "INSTALL-RUN-BY python" in out


@windows_only
def test_setup_rejects_a_python_that_is_too_old(folder, tmp_path):
    bin_dir = tmp_path / "old_bin"
    shim(bin_dir, "python", 'if "%~1"=="-c" exit /b 1\necho INSTALL-RUN-BY old\nexit /b 0')
    code, out = run_bat(folder, "setup.bat", env=path_env(bin_dir))
    assert code == 1
    assert "INSTALL-RUN-BY" not in out and "python.org" in out


@windows_only
def test_setup_reports_a_failed_install_and_keeps_its_exit_code(folder, tmp_path):
    bin_dir = tmp_path / "fail_bin"
    shim(bin_dir, "python", 'if "%~1"=="install.py" exit /b 4\nexit /b 0')
    code, out = run_bat(folder, "setup.bat", env=path_env(bin_dir))
    assert code == 4
    assert "这次没有装好" in out
    assert ". . ." in out


@windows_only
def test_setup_with_a_real_python_reaches_the_real_install_script(folder, tmp_path):
    """The whole chain once for real: setup.bat -> real Python -> a stand-in install.py."""
    (folder / "install.py").write_text(REPORT_ARGS, encoding="utf-8")
    code, out = run_bat(folder, "setup.bat", cwd=tmp_path)
    assert code == 0, out
    assert Path(value_of("CWD", out)) == folder
    assert value_of("ARGS", out) == []
