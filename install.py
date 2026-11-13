"""One-click installer: virtual environment, then libraries, then the hand model.

    python install.py               do everything (setup.bat runs this)
    python install.py --dry-run     only print what it would do
    python install.py --skip-model  leave out the hand-model download

Safe to run again: a working environment is kept, and libraries and the model
that are already there are not downloaded twice.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from airtouch.console import use_utf8_console

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 9)
VENV_PROBE_TIMEOUT_S = 60

FAIL_VENV = "没能建好运行环境。请重新安装 Python 3.12（安装时勾选 Add python.exe to PATH），再试一次。"
FAIL_PIP = ("没能装好需要的程序库。请检查网络连接，再试一次。\n"
            "如果提示找不到 mediapipe，说明这个 Python 版本太新了：请安装 Python 3.12，再试一次。")
FAIL_MODEL = "没能下载手部识别模型。请检查网络连接，再试一次。"

NEXT_STEPS_WINDOWS = """
全部装好了！接下来这样用：
  双击 run.bat     打开手势控制，会有一个摄像头画面的窗口
  双击 tray.bat    在屏幕右下角放一个开关，不用的时候关掉，不耗电
  双击 check.bat   检查一下手势能不能认出来
"""
NEXT_STEPS_OTHER = """
全部装好了！接下来这样用（在终端里输入）：
  .venv/bin/python hand_mouse.py        打开手势控制，会有一个摄像头画面的窗口
  .venv/bin/python airtouch_tray.pyw    放一个开关图标，不用的时候关掉，不耗电
  .venv/bin/python check_gestures.py    检查一下手势能不能认出来
"""


@dataclass(frozen=True)
class Step:
    title: str              # plain Chinese, shown to the user
    command: tuple[str, ...]
    failure: str            # plain Chinese, shown if the command fails


def venv_python(root: str | Path = ROOT, platform: str = sys.platform) -> Path:
    """Where the virtual environment's Python lives on this operating system."""
    venv = Path(root) / ".venv"
    return venv / "Scripts" / "python.exe" if platform == "win32" else venv / "bin" / "python"


def venv_version(root: str | Path = ROOT) -> tuple[int, int] | None:
    """(major, minor) of the environment's Python, or None if it can't run
    (a folder moved from another drive or computer looks fine but is broken)."""
    python = venv_python(root)
    if not python.exists():
        return None
    code = "import sys; print(sys.version_info[0], sys.version_info[1])"
    try:
        done = subprocess.run([str(python), "-c", code], capture_output=True, text=True,
                              timeout=VENV_PROBE_TIMEOUT_S)
        major, minor = done.stdout.split()
        return (int(major), int(minor)) if done.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def steps(root: str | Path = ROOT, skip_model: bool = False, python: str = sys.executable) -> list[Step]:
    """Everything still to be done, in order."""
    root = Path(root)
    venv_py = str(venv_python(root))
    plan = []
    if venv_version(root) != sys.version_info[:2]:
        # Missing, broken, or made by another Python version (for example one too
        # new to have the libraries, before the user installed 3.12): start clean.
        clear = ("--clear",) if (root / ".venv").exists() else ()
        plan.append(Step("建立运行环境（只放在这个文件夹里，不影响电脑上别的东西）",
                         (python, "-m", "venv", *clear, str(root / ".venv")), FAIL_VENV))
    plan.append(Step("安装需要的程序库（第一次要下载比较久，可能要几分钟，请耐心等）",
                     (venv_py, "-m", "pip", "install", "--disable-pip-version-check",
                      "-r", str(root / "requirements.txt")), FAIL_PIP))
    if not skip_model:
        plan.append(Step("准备手部识别模型（只需要下载一次）",
                         (venv_py, "-m", "airtouch.download_model"), FAIL_MODEL))
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirTouch 一键安装")
    parser.add_argument("--dry-run", action="store_true", help="只显示要做的步骤，什么都不做")
    parser.add_argument("--skip-model", action="store_true", help="不下载手部识别模型")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, root: str | Path = ROOT, run=subprocess.call) -> int:
    use_utf8_console()
    args = parse_args(argv)
    if sys.version_info[:2] < MIN_PYTHON:
        print("这个 Python 版本太旧了。请安装 Python 3.12，再试一次。")
        return 1

    plan = steps(root, args.skip_model)
    if args.dry_run:
        print("这次只是预演，什么都不会安装。真正安装时会做这些事：")
        for number, step in enumerate(plan, 1):
            print(f"  {number}. {step.title}")
            print(f"     {subprocess.list2cmdline(step.command)}")
        return 0

    for number, step in enumerate(plan, 1):
        print(f"\n[{number}/{len(plan)}] {step.title}", flush=True)
        try:
            code = run(list(step.command), cwd=str(root))
        except OSError:
            code = 1
        if code != 0:
            print(f"\n{step.failure}")
            return code
    print(NEXT_STEPS_WINDOWS if sys.platform == "win32" else NEXT_STEPS_OTHER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
