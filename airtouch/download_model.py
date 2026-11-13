"""One-time download of MediaPipe's hand model - AirTouch's only network use.

Run:  python -m airtouch.download_model
"""

from __future__ import annotations

import sys
import urllib.request
import zlib
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .config import HAND_MODEL_PATH
from .console import use_utf8_console

_BASE = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/"
MODEL_URLS = (_BASE + "latest/hand_landmarker.task", _BASE + "1/hand_landmarker.task")
MIN_VALID_BYTES = 1_000_000     # anything smaller is an error page, not the model
MODEL_MEMBERS = ("hand_detector.tflite", "hand_landmarks_detector.tflite")


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AirTouch-installer"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _valid_model(data: bytes) -> bool:
    """Check the task container and TFLite headers, including every member's CRC."""
    if len(data) < MIN_VALID_BYTES:
        return False
    try:
        with ZipFile(BytesIO(data)) as archive:
            if not all(name in archive.namelist() for name in MODEL_MEMBERS):
                return False
            if archive.testzip() is not None:
                return False
            for name in MODEL_MEMBERS:
                with archive.open(name) as model:
                    if model.read(8)[4:8] != b"TFL3":
                        return False
        return True
    except (OSError, ValueError, BadZipFile, RuntimeError, NotImplementedError, zlib.error):
        return False


def ensure_model(path: str | Path = HAND_MODEL_PATH, quiet: bool = False, fetch=_fetch) -> Path:
    """Return the model's path, downloading it first if it isn't there yet.

    Written to a temporary name and renamed at the end, so a broken connection
    can never leave a half-finished file that looks like the real model.
    Raises RuntimeError with a plain message if every download attempt fails.
    """
    path = Path(path)
    if path.is_file():
        try:
            if _valid_model(path.read_bytes()):
                return path
        except OSError:
            pass
    say = (lambda _msg: None) if quiet else print
    say("正在下载手部识别模型（约 8MB，只需要下载一次）...")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    errors = []
    for url in MODEL_URLS:
        try:
            data = fetch(url)
        except Exception as e:
            errors.append(f"{url}: {e}")
            continue
        if not _valid_model(data):
            errors.append(f"{url}: invalid hand model ({len(data)} bytes)")
            continue
        tmp.write_bytes(data)
        tmp.replace(path)
        say(f"手部识别模型已准备好：{path}")
        return path
    raise RuntimeError(
        "手部识别模型下载失败。请检查网络连接后再试一次。\n" + "\n".join(errors)
    )


def main() -> int:
    use_utf8_console()
    try:
        ensure_model()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
