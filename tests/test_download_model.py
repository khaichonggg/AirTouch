"""ensure_model with a fake download: these tests never touch the network."""

from __future__ import annotations

from io import BytesIO
import random
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from airtouch import download_model
from airtouch.download_model import MIN_VALID_BYTES, MODEL_URLS, ensure_model

def model_bytes(names=("hand_detector.tflite", "hand_landmarks_detector.tflite"), magic=b"TFL3"):
    """A task container with TFLite headers, without loading a real native model."""
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name in names:
            archive.writestr(name, b"\x1c\x00\x00\x00" + magic + b"m" * (MIN_VALID_BYTES // 2))
    return stream.getvalue()


GOOD = model_bytes()
BAD_CRC = bytearray(GOOD)
BAD_CRC[GOOD.index(b"TFL3") + 8] ^= 1
BAD_CRC = bytes(BAD_CRC)
TINY = b"<html>not found</html>"


class FakeFetch:
    """Answers each URL in turn from ``answers`` (bytes, or an exception to raise)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def must_not_fetch(url):
    raise AssertionError("should not download")


def test_existing_valid_file_is_not_downloaded_again(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(GOOD)
    assert ensure_model(path, quiet=True, fetch=must_not_fetch) == path
    assert path.read_bytes() == GOOD


def test_download_writes_the_file_and_leaves_no_part_file(tmp_path):
    path = tmp_path / "models" / "hand_landmarker.task"     # folder doesn't exist yet
    fetch = FakeFetch(GOOD)
    assert ensure_model(path, quiet=True, fetch=fetch) == path
    assert path.read_bytes() == GOOD
    assert fetch.urls == [MODEL_URLS[0]]
    assert [p.name for p in path.parent.iterdir()] == ["hand_landmarker.task"]


def test_too_small_data_is_rejected_and_the_next_url_is_tried(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    fetch = FakeFetch(TINY, GOOD)
    ensure_model(path, quiet=True, fetch=fetch)
    assert path.read_bytes() == GOOD
    assert fetch.urls == list(MODEL_URLS)


def test_a_failed_url_falls_through_to_the_next(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    ensure_model(path, quiet=True, fetch=FakeFetch(OSError("no route"), GOOD))
    assert path.read_bytes() == GOOD


def test_all_failing_raises_a_plain_error_and_leaves_nothing_behind(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    with pytest.raises(RuntimeError) as err:
        ensure_model(path, quiet=True, fetch=FakeFetch(OSError("no route"), TimeoutError("slow")))
    assert "下载失败" in str(err.value) and "网络" in str(err.value)
    assert list(tmp_path.iterdir()) == []


def test_only_too_small_answers_raise_and_write_nothing(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    with pytest.raises(RuntimeError):
        ensure_model(path, quiet=True, fetch=FakeFetch(TINY, TINY))
    assert list(tmp_path.iterdir()) == []


def test_an_undersized_existing_file_is_replaced(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(TINY)
    ensure_model(path, quiet=True, fetch=FakeFetch(GOOD))
    assert path.read_bytes() == GOOD


@pytest.mark.parametrize("damaged", [
    b"not a task archive" * MIN_VALID_BYTES,
    model_bytes(names=("hand_detector.tflite", "unexpected.tflite")),
    model_bytes(magic=b"HTML"),
    BAD_CRC,
], ids=["not-zip", "missing-landmark-model", "bad-tflite-header", "crc-mismatch"])
def test_large_damaged_existing_model_is_replaced(tmp_path, damaged):
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(damaged)
    fetch = FakeFetch(GOOD)
    ensure_model(path, quiet=True, fetch=fetch)
    assert path.read_bytes() == GOOD
    assert fetch.urls == [MODEL_URLS[0]]


def test_large_damaged_download_is_rejected_before_fallback(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    fetch = FakeFetch(BAD_CRC, GOOD)
    ensure_model(path, quiet=True, fetch=fetch)
    assert path.read_bytes() == GOOD
    assert fetch.urls == list(MODEL_URLS)


def damaged_compressed_model():
    stream = BytesIO()
    rng = random.Random(0)
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for name in download_model.MODEL_MEMBERS:
            archive.writestr(name, b"\x1c\x00\x00\x00TFL3" + rng.randbytes(600_000))
        member = archive.getinfo("hand_detector.tflite")
        offset = member.header_offset + 30 + len(member.filename.encode()) + len(member.extra)
    data = bytearray(stream.getvalue())
    data[offset] = (data[offset] & 0xF8) | 0x07   # DEFLATE's reserved block type
    return bytes(data)


def test_invalid_compressed_existing_model_is_replaced(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(damaged_compressed_model())
    fetch = FakeFetch(GOOD)
    ensure_model(path, quiet=True, fetch=fetch)
    assert path.read_bytes() == GOOD
    assert fetch.urls == [MODEL_URLS[0]]


def test_invalid_compressed_download_tries_the_fallback_url(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    fetch = FakeFetch(damaged_compressed_model(), GOOD)
    ensure_model(path, quiet=True, fetch=fetch)
    assert path.read_bytes() == GOOD
    assert fetch.urls == list(MODEL_URLS)


def test_a_failed_download_keeps_what_was_there(tmp_path):
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(TINY)
    with pytest.raises(RuntimeError):
        ensure_model(path, quiet=True, fetch=FakeFetch(OSError("x"), OSError("y")))
    assert path.read_bytes() == TINY


def test_works_in_a_folder_with_a_non_ascii_name(tmp_path):
    folder = tmp_path / "Bachelor in Computer Science 【 Final Year Project 】" / "模型"
    path = folder / "hand_landmarker.task"
    assert ensure_model(path, quiet=True, fetch=FakeFetch(GOOD)) == path
    assert path.read_bytes() == GOOD
    assert [p.name for p in folder.iterdir()] == ["hand_landmarker.task"]
    assert ensure_model(path, quiet=True, fetch=must_not_fetch) == path


def test_quiet_prints_nothing_and_normal_mode_explains_itself(tmp_path, capsys):
    ensure_model(tmp_path / "a.task", quiet=True, fetch=FakeFetch(GOOD))
    assert capsys.readouterr().out == ""
    ensure_model(tmp_path / "b.task", fetch=FakeFetch(GOOD))
    out = capsys.readouterr().out
    assert "下载" in out and "只需要下载一次" in out


def test_main_reports_failure_in_plain_words(monkeypatch, capsys):
    def fail():
        raise RuntimeError("手部识别模型下载失败。")

    monkeypatch.setattr(download_model, "ensure_model", fail)
    assert download_model.main() == 1
    assert "下载失败" in capsys.readouterr().err


def test_main_returns_zero_on_success(monkeypatch):
    monkeypatch.setattr(download_model, "ensure_model", lambda: None)
    assert download_model.main() == 0
