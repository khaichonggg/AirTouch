"""Make Chinese console messages safe on any Windows setup.

When output is redirected (or the console uses an old code page), Python can
crash with UnicodeEncodeError on the first Chinese character. Every command-line
entry point calls ``use_utf8_console()`` first so that can never happen.
"""

from __future__ import annotations

import sys


def use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass    # pythonw has no console; test harnesses replace these streams
