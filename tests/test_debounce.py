"""Debouncer: one bad frame must never flip the gesture."""

from __future__ import annotations

import pytest

from airtouch.debounce import Debouncer


def feed(d: Debouncer, labels) -> list[str]:
    return [d.update(label) for label in labels]


def test_starts_on_none_unless_told_otherwise():
    assert Debouncer().current == "none"
    assert Debouncer(3, initial="point").current == "point"


def test_default_needs_three_frames():
    assert feed(Debouncer(), ["point"] * 3) == ["none", "none", "point"]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_new_label_needs_exactly_n_consecutive_frames(n):
    d = Debouncer(n)
    out = feed(d, ["point"] * n)
    assert out[:-1] == ["none"] * (n - 1)
    assert out[-1] == "point"
    assert d.current == "point"


def test_isolated_flicker_is_suppressed():
    d = Debouncer(3)
    feed(d, ["point"] * 3)
    assert feed(d, ["fist", "point", "scroll", "point", "fist", "point"]) == ["point"] * 6


def test_flicker_of_fewer_than_n_frames_is_suppressed():
    d = Debouncer(3)
    feed(d, ["point"] * 3)
    assert feed(d, ["fist", "fist", "point"]) == ["point"] * 3


def test_alternating_labels_never_get_through():
    d = Debouncer(3)
    assert feed(d, ["fist", "scroll"] * 20) == ["none"] * 40


def test_an_interrupted_run_starts_counting_again():
    d = Debouncer(3)
    assert feed(d, ["point", "point", "none", "point", "point"]) == ["none"] * 5
    assert d.update("point") == "point"


def test_a_different_candidate_restarts_the_count():
    d = Debouncer(3)
    assert feed(d, ["point", "point", "fist", "fist"]) == ["none"] * 4
    assert d.update("fist") == "fist"


def test_switching_back_also_needs_n_frames():
    d = Debouncer(3)
    feed(d, ["point"] * 3)
    assert feed(d, ["none"] * 3) == ["point", "point", "none"]


def test_returned_label_is_always_the_current_one():
    d = Debouncer(2)
    for label in ["a", "a", "b", "a", "b", "b", "c"]:
        assert d.update(label) == d.current


def test_reset_sets_the_label_and_forgets_a_partial_run():
    d = Debouncer(3)
    feed(d, ["point", "point"])
    d.reset()
    assert d.current == "none"
    assert feed(d, ["point", "point"]) == ["none", "none"]     # the old two frames do not count
    d.reset("fist")
    assert d.current == "fist"
    assert d.update("fist") == "fist"


def test_n_equal_one_passes_everything_through():
    labels = ["point", "fist", "fist", "none", "scroll", "point"]
    assert feed(Debouncer(1), labels) == labels


@pytest.mark.parametrize("n", [0, -5])
def test_frame_count_is_at_least_one(n):
    d = Debouncer(n)
    assert d.frames == 1
    assert d.update("point") == "point"
