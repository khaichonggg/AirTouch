"""The tray app's entry point for pythonw.exe, which runs without a console window.

Started by tray.bat and by "start with Windows". With no console, anything
printed and any crash would simply vanish, so it all goes to logs/tray.log.
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "tray.log"
MAX_LOG_BYTES = 1_000_000


def redirect_output() -> None:
    """Send Python's output, and that of native libraries (fd 2), to the log file."""
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        too_big = LOG_PATH.exists() and LOG_PATH.stat().st_size > MAX_LOG_BYTES
        stream = open(LOG_PATH, "w" if too_big else "a", encoding="utf-8", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stderr = stream
    try:
        os.dup2(stream.fileno(), 2)
    except OSError:
        pass
    logging.basicConfig(stream=stream, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from airtouch.performance import configure_performance_log
        configure_performance_log(ROOT / "logs")
    except OSError:
        logging.exception("could not open performance log")


def show_error(message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "AirTouch", 0x10)      # 0x10 = error icon
    except Exception:
        pass


def main() -> int:
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        redirect_output()
        from airtouch.tray import main as tray_main

        return tray_main()
    except Exception as error:
        logging.exception("the tray app could not start")
        show_error(f"AirTouch 打不开：{error}\n请检查项目文件夹是否可以访问，或重新双击 setup.bat 安装。\n如果日志可以写入，详细原因在：\n{LOG_PATH}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
