"""The user-study kit: SUS scoring, Fitts' law measures, the target task's logic.

Everything here is headless: no webcam, no window, and the real mouse is never
moved (the gesture condition runs on synthetic hands through ``WindowBackend``).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import random
import re
import subprocess
import sys
import types
from datetime import datetime

import cv2
import numpy as np
import pytest

from airtouch import controller as controller_module
from airtouch.config import PROJECT_ROOT, Config
from airtouch.controller import MouseController
from airtouch.engine import GestureEngine
from airtouch.landmarks import anchor_point
from evaluation import analyze_results as ar
from evaluation import read_table
from evaluation import sus_score as sus
from evaluation import task_runner as tr
from ml.synthetic_data import HandPose, make_hand


# ------------------------------------------------------------------- SUS score
def test_sus_all_threes_is_50():
    assert sus.sus_score([3] * 10) == 50.0


def test_sus_best_case_is_100():
    assert sus.sus_score([5, 1] * 5) == 100.0


def test_sus_worst_case_is_0():
    assert sus.sus_score([1, 5] * 5) == 0.0


def test_sus_mixed_answers_known_value():
    # odd items 3+3+4+3+4 = 17, even items 3+4+3+3+4 = 17 -> 34 x 2.5
    assert sus.sus_score([4, 2, 4, 1, 5, 2, 4, 2, 5, 1]) == 85.0


def test_sus_accepts_the_number_forms_excel_writes():
    assert sus.sus_score(["3", 3.0, " 3 ", "3.0", 3, 3, 3, 3, 3, 3]) == 50.0


@pytest.mark.parametrize("bad", [
    [3] * 9,
    [3] * 11,
    [],
    [0] + [3] * 9,
    [6] + [3] * 9,
    [3.5] + [3] * 9,
    ["x"] + [3] * 9,
    [""] + [3] * 9,
    [None] + [3] * 9,
    [True] + [3] * 9,
    [float("nan")] + [3] * 9,
    [float("inf")] + [3] * 9,
    "3333333333",
    5,
])
def test_sus_invalid_answers_raise(bad):
    with pytest.raises(sus.SusError):
        sus.sus_score(bad)


@pytest.mark.parametrize("score,adjective", [
    (0, "Worst imaginable"), (16.39, "Worst imaginable"),
    (16.4, "Awful"), (27.99, "Awful"),
    (28.0, "Poor"), (43.29, "Poor"),
    (43.3, "OK"), (61.14, "OK"),
    (61.15, "Good"), (78.44, "Good"),
    (78.45, "Excellent"), (88.19, "Excellent"),
    (88.2, "Best imaginable"), (100, "Best imaginable"),
])
def test_adjective_boundaries(score, adjective):
    assert sus.adjective_rating(score) == adjective


def test_adjective_cutoffs_are_midpoints_of_the_published_means():
    scale = sus.ADJECTIVE_SCALE
    means = [12.5, 20.3, 35.7, 50.9, 71.4, 85.5, 90.9]
    assert [m for _low, _adj, m in scale] == means
    for (low, _adj, _m), a, b in zip(scale[1:], means, means[1:]):
        assert low == pytest.approx((a + b) / 2)


@pytest.mark.parametrize("score,letter", [
    (0, "F"), (59.9, "F"), (60, "D"), (69.9, "D"), (70, "C"), (79.9, "C"),
    (80, "B"), (89.9, "B"), (90, "A"), (100, "A"),
])
def test_grade_boundaries(score, letter):
    assert sus.grade(score) == letter


@pytest.mark.parametrize("score,verdict", [
    (49.9, "Not acceptable"), (50, "Marginal"), (70, "Marginal"), (70.1, "Acceptable"),
])
def test_acceptability_boundaries(score, verdict):
    assert sus.acceptability(score) == verdict


@pytest.mark.parametrize("score", [-0.1, 100.1, float("nan"), float("inf")])
def test_adjective_and_grade_reject_impossible_scores(score):
    with pytest.raises(sus.SusError):
        sus.adjective_rating(score)
    with pytest.raises(sus.SusError):
        sus.grade(score)


def _write_sus(path, text, encoding="utf-8"):
    path.write_bytes(text.encode(encoding))
    return path


SUS_HEADER = "participant,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\n"


def test_sus_cli_scores_every_participant_with_mean_and_sd(tmp_path, capsys):
    path = _write_sus(tmp_path / "sus.csv", SUS_HEADER
                      + "P01,3,3,3,3,3,3,3,3,3,3\n"
                      + "P02,5,1,5,1,5,1,5,1,5,1\n"
                      + "P03,1,5,1,5,1,5,1,5,1,5\n")
    assert sus.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert re.search(r"P01\s+50\.0\s+OK\s+F", out)
    assert re.search(r"P02\s+100\.0\s+Best imaginable\s+A", out)
    assert re.search(r"P03\s+0\.0\s+Worst imaginable\s+F", out)
    assert re.search(r"Mean\s+50\.0", out)
    assert re.search(r"SD\s+50\.0", out)


def test_sus_csv_as_excel_saves_it(tmp_path):
    # byte-order mark, capitalised headers, semicolons, a row of empty cells
    text = "Participant;Q1;Q2;Q3;Q4;Q5;Q6;Q7;Q8;Q9;Q10\nP01;3;3;3;3;3;3;3;3;3;3\n;;;;;;;;;;\n"
    results = sus.score_csv(_write_sus(tmp_path / "s.csv", text, "utf-8-sig"))
    assert [(r.participant, r.score) for r in results] == [("P01", 50.0)]


def test_sus_csv_without_participant_column_numbers_the_rows(tmp_path):
    text = "q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\n3,3,3,3,3,3,3,3,3,3\n5,1,5,1,5,1,5,1,5,1\n"
    results = sus.score_csv(_write_sus(tmp_path / "s.csv", text))
    assert [(r.participant, r.score) for r in results] == [("P1", 50.0), ("P2", 100.0)]


def test_sus_csv_reports_every_problem_and_scores_nothing(tmp_path, capsys):
    path = _write_sus(tmp_path / "sus.csv", SUS_HEADER
                      + "P01,3,3,3,3,3,3,3,3,3,3\n"
                      + "P02,3,3,6,3,3,3,3,3,3,3\n"
                      + "P03,3,3,3,,3,3,3,3,3,3\n"
                      + "P01,3,3,3,3,3,3,3,3,3,3\n")
    with pytest.raises(sus.SusError) as e:
        sus.score_csv(path)
    message = str(e.value)
    assert "第 3 行（P02）" in message and "第 3 题" in message and "6" in message
    assert "第 4 行（P03）" in message and "第 4 题没有填" in message
    assert "第 5 行（P01）" in message and "已经出现过" in message
    assert sus.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""       # no partial results
    assert "第 3 行" in captured.err


@pytest.mark.parametrize("text,fragment", [
    ("participant,q1,q2\nP01,3,3\n", "缺少这些列"),
    (SUS_HEADER, "没有任何人的答案"),
    ("", "缺少这些列"),
])
def test_sus_csv_wrong_shape_gives_a_plain_message(tmp_path, text, fragment):
    with pytest.raises(sus.SusError, match=fragment):
        sus.score_csv(_write_sus(tmp_path / "s.csv", text))


def test_sus_missing_file_gives_a_plain_message(tmp_path, capsys):
    assert sus.main([str(tmp_path / "nope.csv")]) == 2
    assert "找不到" in capsys.readouterr().err


def test_sus_single_participant_has_no_sd(tmp_path, capsys):
    path = _write_sus(tmp_path / "s.csv", SUS_HEADER + "P01,3,3,3,3,3,3,3,3,3,3\n")
    assert sus.main([str(path)]) == 0
    assert re.search(r"SD\s+n/a", capsys.readouterr().out)


# ------------------------------------------------------------------ read_table
def test_read_table_handles_gbk_tabs_and_line_numbers(tmp_path):
    path = tmp_path / "t.csv"
    path.write_bytes("Name\tScore\n\n张三\t1\n".encode("gbk"))
    table = read_table(path)
    assert table.columns == ["name", "score"]
    assert table.rows == [(3, {"name": "张三", "score": "1"})]


# ------------------------------------------------------------ Fitts' law measures
def _trial(dx=0.0, mt=1000.0, amplitude=100.0, width=20.0, hit=True, participant="P1",
           input_name="gesture", block=1, source="s.csv"):
    """A movement along the x axis: start (0, 0), target (A, 0), click dx past it."""
    return ar.Trial(participant, input_name, block, amplitude, width, target=(amplitude, 0.0),
                    start=(0.0, 0.0), click=(amplitude + dx, 0.0), hit=hit, mt_ms=mt, source=source)


def _sequence(dxs=(-2, 2, -1, 1, 0), **kw):
    return [_trial(dx=dx, **kw) for dx in dxs]


# Worked by hand for dx = -2, 2, -1, 1, 0 and A = 100:
#   SD = sqrt(10 / 4) = 1.58114;  We = 4.133 x 1.58114 = 6.53485;  Ae = 100
#   IDe = log2(100 / 6.53485 + 1) = log2(16.30258) = 4.02702 bits
SD_X, WE, IDE = 1.5811388, 6.5348468, 4.02702


def test_id_is_shannon_formulation():
    assert ar.fitts_id(300, 100) == pytest.approx(2.0)
    assert ar.fitts_id(700, 100) == pytest.approx(3.0)


def test_deviation_is_projected_on_the_task_axis():
    def dev(click):
        return ar.Trial("P", "gesture", 1, 50, 10, target=(30, 40), start=(0, 0), click=click,
                        hit=True, mt_ms=1).deviation

    assert dev((33, 44)) == pytest.approx(5.0)      # overshoot along the 3-4-5 axis
    assert dev((27, 36)) == pytest.approx(-5.0)     # undershoot
    assert dev((26, 43)) == pytest.approx(0.0)      # purely sideways: does not count


def test_we_ide_and_throughput_known_values():
    s = ar.sequence_stats(_sequence())
    assert s.n == 5
    assert s.sd_x == pytest.approx(SD_X, abs=1e-6)
    assert s.we == pytest.approx(WE, abs=1e-6)
    assert s.ae == pytest.approx(100.0)
    assert s.ide == pytest.approx(IDE, abs=1e-4)
    assert s.mt_ms == pytest.approx(1000.0)
    assert s.tp == pytest.approx(IDE, abs=1e-4)     # MT = 1 s, so bits/s = bits


def test_effective_amplitude_follows_a_consistent_overshoot():
    # dx = 10, 6, 8: SD = 2, We = 8.266, Ae = 108, IDe = log2(14.06557) = 3.81410
    trials = [_trial(dx=10, mt=400), _trial(dx=6, mt=500), _trial(dx=8, mt=600)]
    s = ar.sequence_stats(trials)
    assert s.we == pytest.approx(8.266)
    assert s.ae == pytest.approx(108.0)
    assert s.ide == pytest.approx(3.81410, abs=1e-4)
    assert s.mt_ms == pytest.approx(500.0)
    assert s.tp == pytest.approx(3.81410 / 0.5, abs=1e-3)


def test_error_rate_counts_misses():
    trials = _sequence((-2, 2, -1, 1))
    trials.append(_trial(dx=30, hit=False))
    assert ar.sequence_stats(trials).error_rate == pytest.approx(0.2)


def test_no_spread_gives_no_throughput_instead_of_a_crash():
    s = ar.sequence_stats(_sequence((3, 3, 3)))
    assert s.we == 0.0 and s.ide is None and s.tp is None
    summary = ar.summarize([s])[0]
    assert summary.overall.tp is None
    assert summary.overall.mt_ms == pytest.approx(1000.0)
    assert "n/a" in ar.format_report([summary])


def test_throughput_is_a_mean_of_means():
    # P1 has two conditions (TP = I and 2I, mean 1.5I); P2 has one (TP = 4I).
    # Each person counts once: (1.5I + 4I) / 2 = 2.75I, not the pooled 7I / 3.
    trials = (_sequence(width=20.0, mt=1000.0, participant="P1")
              + _sequence(width=40.0, mt=500.0, participant="P1")
              + _sequence(width=20.0, mt=250.0, participant="P2"))
    seqs, too_short = ar.split_sequences(trials)
    assert too_short == 0 and len(seqs) == 3
    summary = ar.summarize(seqs)[0]
    assert summary.people["P1"].tp == pytest.approx(1.5 * IDE, abs=1e-3)
    assert summary.people["P2"].tp == pytest.approx(4 * IDE, abs=1e-3)
    assert summary.overall.tp == pytest.approx(2.75 * IDE, abs=1e-3)


def test_blocks_inside_a_condition_are_averaged_first():
    # Same person: block 1 and 2 of one condition, then a second condition. The two
    # blocks must not count double against the other condition.
    trials = (_sequence(block=1, width=20.0, mt=1000.0) + _sequence(block=2, width=20.0, mt=500.0)
              + _sequence(block=1, width=40.0, mt=250.0))
    seqs, _ = ar.split_sequences(trials)
    assert len(seqs) == 3
    # condition 1: (I + 2I) / 2 = 1.5I; condition 2: 4I; mean 2.75I
    assert ar.summarize(seqs)[0].overall.tp == pytest.approx(2.75 * IDE, abs=1e-3)


def test_two_sessions_never_merge_into_one_sequence():
    trials = _sequence(source="a.csv") + _sequence(source="b.csv")
    seqs, _ = ar.split_sequences(trials)
    assert [s.n for s in seqs] == [5, 5]


def test_too_short_sequences_are_counted_not_used():
    seqs, too_short = ar.split_sequences(_sequence((1, 2)) + _sequence(width=40.0))
    assert too_short == 1 and len(seqs) == 1


def test_linear_fit_exact_line():
    fit = ar.linear_fit([1, 2, 3], [300, 500, 700])
    assert (fit.a, fit.b, fit.r2) == (pytest.approx(100), pytest.approx(200), pytest.approx(1.0))


def test_linear_fit_known_r_squared():
    # Sxy = 9.5, Sxx = 5, b = 1.9, a = 0; SSres = 0.70, SStot = 18.75
    fit = ar.linear_fit([1, 2, 3, 4], [2, 4, 5, 8])
    assert fit.b == pytest.approx(1.9)
    assert fit.a == pytest.approx(0.0, abs=1e-9)
    assert fit.r2 == pytest.approx(1 - 0.70 / 18.75)


def test_linear_fit_degenerate_inputs():
    assert ar.linear_fit([2, 2], [100, 200]) is None
    assert ar.linear_fit([], []) is None
    assert math.isnan(ar.linear_fit([1, 2], [5, 5]).r2)


def test_summary_regression_recovers_a_known_fitts_line():
    trials = []
    for amplitude in (100.0, 200.0, 400.0):
        mt = 100 + 200 * ar.fitts_id(amplitude, 20.0)
        trials += _sequence(amplitude=amplitude, width=20.0, mt=mt)
    summary = ar.summarize(ar.split_sequences(trials)[0])[0]
    assert summary.fit_id.a == pytest.approx(100.0)
    assert summary.fit_id.b == pytest.approx(200.0)
    assert summary.fit_id.r2 == pytest.approx(1.0)
    assert summary.fit_id.n == 3
    assert summary.fit_ide is not None


def _people_summary(name, tps):
    measures = {p: ar.Measures(10, 1000.0, 0.0, 10.0, 100.0, 3.0, tp) for p, tp in tps.items()}
    return ar.InputSummary(name, measures["P1"], [], measures, None, None)


def test_paired_t_test_known_values():
    # differences 1, 2, 3: mean 2, SD 1, t = 2 / (1 / sqrt(3)) = 3.4641, df 2, p = 0.0742
    test = ar.paired_throughput_test([
        _people_summary("gesture", {"P1": 2.0, "P2": 3.0, "P3": 4.0}),
        _people_summary("mouse", {"P1": 1.0, "P2": 1.0, "P3": 1.0}),
    ])
    assert test.first == "gesture" and test.second == "mouse"
    assert test.n == 3 and test.df == 2
    assert test.mean_difference == pytest.approx(2.0)
    assert test.t == pytest.approx(3.4641, abs=1e-3)
    assert test.p == pytest.approx(0.0742, abs=1e-3)


def test_paired_t_test_needs_two_inputs_and_shared_people_and_some_variation():
    one = _people_summary("gesture", {"P1": 2.0, "P2": 3.0})
    assert ar.paired_throughput_test([one]) is None
    assert ar.paired_throughput_test([one, _people_summary("mouse", {"P1": 1.0, "P9": 1.0})]) is None
    assert ar.paired_throughput_test([one, _people_summary("mouse", {"P1": 1.0, "P2": 2.0})]) is None


@pytest.mark.parametrize("trial,flagged", [
    (_trial(dx=0.0, hit=True), False),                  # a hit is never an outlier
    (_trial(dx=-95.0, hit=True, width=500.0), False),
    (_trial(dx=15.0, hit=False), False),                # an ordinary miss
    (_trial(dx=-60.0, hit=False, width=100.0), True),   # moved only 40: under A / 2
    (_trial(dx=45.0, hit=False), True),                 # 45 from centre: over 2W = 40
])
def test_spatial_outlier_rule(trial, flagged):
    assert trial.is_spatial_outlier is flagged


# ---------------------------------------------------------------- target layout
def test_ring_count_positions_and_spacing():
    center, amplitude, n = (640.0, 380.0), 400, 11
    points = tr.ring_positions(center, amplitude, n)
    assert len(points) == n
    for p in points:
        assert math.dist(p, center) == pytest.approx(amplitude / 2)
    assert points[0] == pytest.approx((640.0, 180.0))       # first one at the top
    assert points[1][0] > center[0]                         # then clockwise on screen
    steps = [math.dist(a, b) for a, b in zip(points, points[1:] + points[:1])]
    assert steps == pytest.approx([steps[0]] * n)


@pytest.mark.parametrize("n", [3, 5, 9, 11, 13])
def test_alternating_order_visits_every_target_once(n):
    assert sorted(tr.alternating_order(n)) == list(range(n))


def test_alternating_order_known_sequence():
    assert tr.alternating_order(5) == [0, 3, 1, 4, 2]


@pytest.mark.parametrize("n", [5, 11, 13])
def test_alternating_order_always_jumps_across_the_ring(n):
    points = tr.ring_positions((0, 0), 400, n)
    order = tr.alternating_order(n)
    closed = order + order[:1]
    jumps = [math.dist(points[a], points[b]) for a, b in zip(closed, closed[1:])]
    assert jumps == pytest.approx([jumps[0]] * n)           # odd count: all the same length
    assert jumps[0] == pytest.approx(400 * math.cos(math.pi / (2 * n)))
    assert jumps[0] > 0.95 * 400                            # nearly the full diameter


def test_default_conditions_fit_do_not_overlap_and_span_a_useful_difficulty_range():
    tr.check_fits(tr.WINDOW_SIZE, tr.AMPLITUDES, tr.WIDTHS)
    assert tr.NUM_TARGETS % 2 == 1
    neighbour_gap = min(tr.AMPLITUDES) * math.sin(math.pi / tr.NUM_TARGETS)
    assert neighbour_gap >= max(tr.WIDTHS)
    ids = [ar.fitts_id(a, w) for a in tr.AMPLITUDES for w in tr.WIDTHS]
    assert min(ids) < 2.5 and max(ids) > 4.0


def test_window_too_small_is_refused():
    with pytest.raises(ValueError, match="窗口太小"):
        tr.check_fits((800, 500), tr.AMPLITUDES, tr.WIDTHS)


def test_hit_test_inside_edge_outside():
    target = (100.0, 100.0)
    assert tr.hit_test(100, 100, target, 40)
    assert tr.hit_test(120, 100, target, 40)        # exactly on the edge
    assert tr.hit_test(100 + 14, 100 + 14, target, 40)
    assert not tr.hit_test(120.5, 100, target, 40)
    assert not tr.hit_test(100 + 15, 100 + 15, target, 40)


# ------------------------------------------------------------- trial sequencing
def _small_session(**kw):
    defaults = dict(blocks=1, amplitudes=(300,), widths=(60,), n_targets=5, seed=1)
    return tr.FittsSession("P01", "gesture", **{**defaults, **kw})


def _finish(session, t=0.0, step=0.5):
    """Select every target dead centre. Returns the logged trials."""
    records = []
    while not session.finished:
        t += step
        record = session.select(*session.current_target, t)
        if record is not None:
            records.append(record)
    return records


def test_start_target_starts_the_clock_and_logs_nothing():
    session = _small_session()
    assert session.waiting
    start = session.current_target
    # a miss on the start target is ignored: still waiting, target unchanged
    assert session.select(start[0] + 100, start[1], 1.0) is None
    assert session.waiting and session.current_target == start
    assert session.select(*start, 2.0) is None
    assert not session.waiting
    assert session.trial_number == 1
    assert session.trials_done == 0


def test_trial_fields_time_and_start_point():
    session = _small_session()
    start = session.current_target
    session.select(start[0] + 3, start[1], 10.0)      # inside the target, off centre
    first_target = session.current_target
    rec = session.select(first_target[0] + 40, first_target[1], 10.75)  # a miss
    assert rec.participant == "P01" and rec.input == "gesture"
    assert (rec.block, rec.trial, rec.amplitude, rec.width) == (1, 1, 300, 60)
    assert rec.target == first_target
    assert rec.start == (start[0] + 3, start[1])    # the actual previous click
    assert rec.click == (first_target[0] + 40, first_target[1])
    assert rec.hit is False
    assert rec.movement_time_ms == pytest.approx(750.0)
    datetime.fromisoformat(rec.timestamp)
    # a miss still moves on, and the next trial starts from where that click landed
    assert session.trial_number == 2
    next_target = session.current_target
    rec2 = session.select(*next_target, 11.0)
    assert rec2.start == rec.click
    assert rec2.hit is True
    assert rec2.movement_time_ms == pytest.approx(250.0)


def test_sequence_is_closed_and_follows_the_alternating_order():
    session = _small_session()
    targets = list(session.targets)
    start = session.current_target
    records = _finish(session)
    assert [r.trial for r in records] == [1, 2, 3, 4, 5]
    assert [r.target for r in records] == [targets[i] for i in (3, 1, 4, 2, 0)]
    assert records[-1].target == start
    assert session.finished and session.current_target is None
    assert session.select(0, 0, 99.0) is None


def test_full_plan_runs_every_condition_once_per_block():
    session = tr.FittsSession("P01", "mouse", blocks=2, n_targets=5, seed=3)
    records = _finish(session)
    assert len(records) == 2 * 9 * 5
    assert session.trials_done == len(records) and session.hits == len(records)
    for block in (1, 2):
        cells = [(r.amplitude, r.width) for r in records if r.block == block]
        assert len(cells) == 9 * 5
        assert set(cells) == {(a, w) for a in tr.AMPLITUDES for w in tr.WIDTHS}
        assert all(cells.count(c) == 5 for c in set(cells))


def test_condition_order_is_random_per_block_and_reproducible():
    plan = tr.plan_sequences(2, tr.AMPLITUDES, tr.WIDTHS, seed=11)
    block1 = [(c.amplitude, c.width) for c in plan if c.block == 1]
    block2 = [(c.amplitude, c.width) for c in plan if c.block == 2]
    assert block1 != block2
    assert plan == tr.plan_sequences(2, tr.AMPLITUDES, tr.WIDTHS, seed=11)
    assert plan != tr.plan_sequences(2, tr.AMPLITUDES, tr.WIDTHS, seed=12)


def test_default_order_depends_on_participant_and_input_but_not_on_the_process():
    def plan(participant, input_name):
        return tr.FittsSession(participant, input_name).plan

    assert plan("P01", "mouse") == plan("P01", "mouse")
    assert plan("P01", "mouse") != plan("P02", "mouse")
    assert plan("P01", "mouse") != plan("P01", "gesture")


@pytest.mark.parametrize("kwargs", [
    dict(blocks=0), dict(n_targets=10), dict(n_targets=1), dict(size=(800, 500)),
])
def test_bad_session_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        tr.FittsSession("P01", "gesture", **kwargs)


def test_unknown_input_is_refused():
    with pytest.raises(ValueError):
        tr.FittsSession("P01", "touchpad")


@pytest.mark.parametrize("name", ["../evil", "a/b", "a\\b", "", "  ", "a b", "x" * 41, "a:b", "a.csv"])
def test_unsafe_participant_codes_are_refused(name):
    with pytest.raises(ValueError):
        tr.clean_participant(name)


@pytest.mark.parametrize("name", ["P01", "p-1_a", "参与者1"])
def test_safe_participant_codes_are_kept(name):
    assert tr.clean_participant(f" {name} ") == name


# --------------------------------------------------------------- WindowBackend
def test_window_backend_reports_the_window_as_the_screen():
    backend = tr.WindowBackend(800, 600)
    assert backend.screen_size() == (800, 600)
    assert backend.pos == (400, 300)


def test_window_backend_clamps_to_the_window():
    backend = tr.WindowBackend(800, 600)
    backend.move(100, 200)
    assert backend.pos == (100, 200)
    backend.move(-5, -9)
    assert backend.pos == (0, 0)
    backend.move(5000, 5000)
    assert backend.pos == (799, 599)


def test_window_backend_presses_remember_where_they_happened():
    backend = tr.WindowBackend(800, 600)
    backend.move(10, 20)
    backend.press("left")
    assert backend.down == {"left"}
    backend.move(30, 40)
    backend.release("left")
    backend.press("right")
    assert backend.down == {"right"}
    assert backend.pop_presses() == [("left", 10, 20), ("right", 30, 40)]
    assert backend.pop_presses() == []


def test_window_backend_ignores_scroll_and_zoom():
    backend = tr.WindowBackend()
    backend.scroll(3.5)
    backend.zoom(-1.0)
    assert backend.pos == (backend.size[0] // 2, backend.size[1] // 2)
    assert backend.pop_presses() == []


def test_controller_maps_the_active_box_onto_window_pixels():
    backend = tr.WindowBackend(1280, 720)
    mapper = MouseController(Config(), backend)
    x0, y0, w, h = mapper.active_box
    assert mapper.to_screen(x0, y0) == (0, 0)
    assert mapper.to_screen(x0 + w, y0 + h) == (1279, 719)
    assert mapper.to_screen(0.0, 0.0) == (0, 0)         # outside the box: clamped
    assert mapper.to_screen(1.0, 1.0) == (1279, 719)
    # same pixels per physical camera unit on both axes
    assert 1280 / (w * Config().aspect) == pytest.approx(720 / h)


# ----------------------------------------------------------------- CSV logging
def test_csv_path_format(tmp_path):
    path = tr.session_csv_path("P01", "gesture", tmp_path, when=datetime(2026, 11, 3, 9, 8, 7))
    assert path == tmp_path / "P01_gesture_20261103-090807.csv"


def test_csv_path_refuses_unsafe_participant(tmp_path):
    with pytest.raises(ValueError):
        tr.session_csv_path("../P01", "gesture", tmp_path)


def test_csv_columns_cover_what_the_analysis_needs():
    assert set(ar.REQUIRED_COLUMNS) <= set(tr.CSV_FIELDS)


def test_csv_round_trip(tmp_path):
    path = tmp_path / "nested" / "P01_gesture_x.csv"
    session = _small_session()
    with tr.TrialLogger(path) as logger:
        session.on_trial = logger
        records = _finish(session)
    assert logger.count == 5
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert tuple(rows[0]) == tr.CSV_FIELDS
    assert len(rows) == 5
    assert all(r["hit"] == "1" for r in rows)
    trials, notes = ar.load_trials([path])
    assert notes == []
    assert len(trials) == 5
    for trial, rec in zip(trials, records):
        assert trial.participant == rec.participant and trial.input == rec.input
        assert (trial.block, trial.amplitude, trial.width) == (rec.block, rec.amplitude, rec.width)
        assert trial.target == pytest.approx(rec.target, abs=0.01)
        assert trial.start == pytest.approx(rec.start, abs=0.01)
        assert trial.click == pytest.approx(rec.click, abs=0.01)
        assert trial.hit == rec.hit
        assert trial.mt_ms == pytest.approx(rec.movement_time_ms, abs=0.1)


def test_logger_saves_each_trial_at_once(tmp_path):
    path = tmp_path / "live.csv"
    session = _small_session()
    logger = tr.TrialLogger(path)
    session.on_trial = logger
    session.select(*session.current_target, 1.0)
    session.select(*session.current_target, 2.0)
    with path.open(encoding="utf-8") as f:      # still open: a crash now would lose nothing
        assert len(f.readlines()) == 2
    logger.close()
    logger.close()      # harmless twice


def test_logger_never_overwrites_a_session(tmp_path):
    path = tmp_path / "same.csv"
    path.write_text("precious", encoding="utf-8")
    with pytest.raises(FileExistsError):
        tr.TrialLogger(path)
    assert path.read_text(encoding="utf-8") == "precious"


def test_logger_leaves_no_file_for_an_empty_session(tmp_path):
    path = tmp_path / "empty.csv"
    with tr.TrialLogger(path):
        assert path.exists()
    assert not path.exists()


# ---------------------------------------------------- analysis of real-shaped logs
def _simulate(folder, participant, input_name, a, b, spread, seed):
    """A believable session: MT = a + b * ID with noise, clicks scattered round the target."""
    rng = random.Random(seed)
    path = tr.session_csv_path(participant, input_name, folder, when=datetime(2026, 11, 3, 10, 0, seed))
    with tr.TrialLogger(path) as logger:
        session = tr.FittsSession(participant, input_name, blocks=2, on_trial=logger, seed=seed)
        t = 0.0
        while not session.finished:
            target, c = session.current_target, session.condition
            if session.waiting:
                session.select(*target, t)
                continue
            t += (a + b * ar.fitts_id(c.amplitude, c.width) + rng.gauss(0, 15)) / 1000
            session.select(target[0] + rng.gauss(0, c.width * spread),
                           target[1] + rng.gauss(0, c.width * spread), t)
    return path


def test_analysis_on_a_folder_of_sessions(tmp_path, capsys):
    results = tmp_path / "results"
    _simulate(results, "P01", "gesture", 400, 450, 0.25, 1)
    _simulate(results, "P02", "gesture", 450, 420, 0.25, 2)
    _simulate(results, "P03", "gesture", 380, 480, 0.25, 3)
    _simulate(results, "P01", "mouse", 120, 170, 0.20, 4)
    _simulate(results, "P02", "mouse", 100, 180, 0.20, 5)
    _simulate(results, "P03", "mouse", 130, 160, 0.20, 6)

    files, notes = ar.find_csv_files([results])
    assert len(files) == 6 and notes == []
    trials, notes = ar.load_trials(files)
    assert notes == []
    assert len(trials) == 6 * 2 * 9 * 11
    summaries = ar.summarize(ar.split_sequences(trials)[0])
    gesture, mouse = summaries
    assert gesture.input == "gesture" and mouse.input == "mouse"
    assert len(gesture.people) == 3 and len(gesture.conditions) == 9
    assert gesture.overall.trials == 3 * 2 * 9 * 11
    assert mouse.overall.tp > gesture.overall.tp > 0
    assert mouse.overall.mt_ms < gesture.overall.mt_ms
    # the simulated law comes back out of the regression
    assert gesture.fit_id.b == pytest.approx(450, rel=0.1)
    assert gesture.fit_id.r2 > 0.95
    assert mouse.fit_id.r2 > 0.95

    plot = tmp_path / "plots" / "fitts.png"
    assert ar.main([str(results), "--detail", "--plot", str(plot)]) == 0
    out = capsys.readouterr().out
    for fragment in ("gesture", "mouse", "TP(bits/s)", "MT =", "R^2", "Per participant", "P03",
                     "Per A x W condition", "Paired t-test", "t(2)"):
        assert fragment in out
    assert plot.read_bytes().startswith(b"\x89PNG")


def test_plot_defaults_to_the_results_folder(tmp_path, capsys):
    _simulate(tmp_path, "P01", "mouse", 120, 170, 0.20, 1)
    assert ar.main([str(tmp_path), "--plot"]) == 0
    assert (tmp_path / "fitts_plot.png").read_bytes().startswith(b"\x89PNG")
    assert "fitts_plot.png" in capsys.readouterr().out


def test_analysis_of_named_files_only(tmp_path, capsys):
    one = _simulate(tmp_path, "P01", "mouse", 120, 170, 0.20, 1)
    _simulate(tmp_path, "P02", "gesture", 400, 450, 0.25, 2)
    assert ar.main([str(one)]) == 0
    out = capsys.readouterr().out
    assert "P01" in out and "P02" not in out and "gesture" not in out


def test_exclude_outliers_drops_accidental_clicks(tmp_path, capsys):
    results = tmp_path / "results"
    results.mkdir()
    # 11 clean trials and one accidental click right after the previous selection
    fields = list(tr.CSV_FIELDS)
    with (results / "P01_gesture_x.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for i in range(12):
            accident = i == 11
            dx = [-2, 2, -1, 1, 0, 3, -3, 2, -2, 1, -1][i] if not accident else -99.0
            w.writerow(dict(participant="P01", input="gesture", block=1, trial=i + 1, amplitude=100, width=20,
                            target_x=100, target_y=0, start_x=0, start_y=0, click_x=100 + dx, click_y=0,
                            hit=int(not accident), movement_time_ms=800, timestamp="t"))
    assert ar.main([str(results)]) == 0
    kept = capsys.readouterr().out
    assert "1 次明显的误触" in kept and "--exclude-outliers" in kept
    kept_we = float(re.search(r"gesture\s+1\s+12\s+[\d.]+\s+[\d.]+\s+([\d.]+)", kept).group(1))

    assert ar.main([str(results), "--exclude-outliers"]) == 0
    dropped = capsys.readouterr().out
    assert "已去掉 1 次" in dropped
    dropped_we = float(re.search(r"gesture\s+1\s+11\s+[\d.]+\s+[\d.]+\s+([\d.]+)", dropped).group(1))
    assert dropped_we < kept_we / 3      # one accident had been inflating the spread


def test_analysis_empty_folder_gives_a_plain_message(tmp_path, capsys):
    assert ar.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "没有结果文件" in out and "task_runner" in out


def test_analysis_missing_path_gives_a_plain_message(tmp_path, capsys):
    assert ar.main([str(tmp_path / "nowhere")]) == 1
    assert "找不到" in capsys.readouterr().out


def test_analysis_skips_files_with_missing_columns_but_uses_the_rest(tmp_path, capsys):
    _simulate(tmp_path, "P01", "mouse", 120, 170, 0.20, 1)
    (tmp_path / "sus.csv").write_text(SUS_HEADER + "P01,3,3,3,3,3,3,3,3,3,3\n", encoding="utf-8")
    assert ar.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "sus.csv 不是测试结果文件" in out and "movement_time_ms" in out
    assert "P01" in out and "TP(bits/s)" in out


def test_analysis_only_bad_files_gives_a_plain_message(tmp_path, capsys):
    (tmp_path / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert ar.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "不是测试结果文件" in out and "没有可以分析的数据" in out


def test_analysis_skips_bad_rows_and_says_which(tmp_path, capsys):
    path = _simulate(tmp_path, "P01", "mouse", 120, 170, 0.20, 1)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("mouse,", "mouse,oops", 1)          # block is not a number
    lines[2] = re.sub(r",1,[\d.]+,(?=[^,]*$)|,0,[\d.]+,(?=[^,]*$)", ",1,-5.0,", lines[2])  # negative time
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    trials, notes = ar.load_trials([path])
    assert len(trials) == 2 * 9 * 11 - 2
    assert len(notes) == 1 and "2 行" in notes[0] and "第 2, 3 行" in notes[0]
    assert ar.main([str(path)]) == 0
    assert "2 行数据不对" in capsys.readouterr().out


def test_analysis_session_too_short_gives_a_plain_message(tmp_path, capsys):
    session = _small_session()
    path = tmp_path / "short.csv"
    with tr.TrialLogger(path) as logger:
        session.on_trial = logger
        session.select(*session.current_target, 1.0)
        session.select(*session.current_target, 2.0)
    assert ar.main([str(path)]) == 1
    assert "太少" in capsys.readouterr().out


def test_analysis_notes_sequences_without_spread(tmp_path, capsys):
    # every click dead centre: a perfect (scripted) session has no spread at all
    session = _small_session()
    path = tmp_path / "perfect.csv"
    with tr.TrialLogger(path) as logger:
        session.on_trial = logger
        _finish(session)
    assert ar.main([str(path)]) == 0
    assert "完全没有散开" in capsys.readouterr().out


# ------------------------------------------------------------------- drawing
def _bgr_at(img, p):
    return tuple(int(v) for v in img[int(round(p[1])), int(round(p[0]))])


def test_render_highlights_the_current_target_without_opening_a_window():
    session = _small_session()
    img = tr.render(session)
    assert img.shape == (720, 1280, 3)
    assert _bgr_at(img, session.current_target) == tr.TARGET_COLOR
    other = session.targets[1]
    assert _bgr_at(img, other) != tr.TARGET_COLOR


def test_render_shows_cursor_preview_miss_and_finished_screen():
    session = _small_session()
    session.select(*session.current_target, 1.0)
    target = session.current_target
    session.select(target[0] + 200, target[1], 2.0)      # a miss
    assert session.last.hit is False
    frame = np.full((480, 640, 3), 123, dtype=np.uint8)
    hand = make_hand("point", np.random.default_rng(0), pose=HandPose(), noise=0)
    box = (0.2, 0.2, 0.6, 0.6)
    img = tr.render(session, cursor=(50, 500), pressed=True, frame=frame, hands=[hand], box=box, status="point")
    assert _bgr_at(img, (50, 500)) == tr.PRESSED_COLOR
    assert _bgr_at(img, session.last.click) != tuple(int(v) for v in tr.blank_canvas(1, 1)[0, 0])
    # the preview sits in the top right corner, clear of the ring
    assert _bgr_at(img, (1280 - tr.MARGIN - 2, tr.TOP_BAR + tr.MARGIN + 150)) == (123, 123, 123)
    _finish(session)
    assert tr.render(session).shape == (720, 1280, 3)


# ------------------------------------------------------------------ main loops
class FakeDisplay:
    """Stands in for CvDisplay; ``on_wait`` runs on every wait_key, like an event pump."""

    def __init__(self, on_wait=None, keys=(), limit=100_000):
        self.on_wait = on_wait
        self.keys = list(keys)
        self.limit = limit
        self.shown = 0
        self.callback = None
        self.is_closed = False
        self.close_calls = 0

    def show(self, img):
        self.shown += 1

    def wait_key(self, ms=1):
        self.limit -= 1
        assert self.limit > 0, "the loop never ended"
        if self.on_wait is not None:
            self.on_wait(self)
        return self.keys.pop(0) if self.keys else 255

    def on_mouse(self, callback):
        self.callback = callback

    def closed(self):
        return self.is_closed

    def close(self):
        self.close_calls += 1


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_mouse_loop_turns_clicks_into_selections_timed_at_the_event():
    session = _small_session(n_targets=3)
    clock = FakeClock()

    def click(display):
        clock.t += 0.4
        if not session.finished:
            display.callback(cv2.EVENT_LBUTTONDOWN, *map(round, session.current_target), 0, None)

    records = []
    session.on_trial = records.append
    display = FakeDisplay(on_wait=click)
    assert tr.mouse_loop(session, display, clock) is True
    assert session.finished
    assert len(records) == 3 and all(r.hit for r in records)
    assert [r.movement_time_ms for r in records] == pytest.approx([400.0, 400.0, 400.0])
    assert display.shown >= 4


def test_mouse_input_counts_only_left_presses():
    clock = FakeClock()
    mouse = tr.MouseInput(clock)
    mouse.on_event(cv2.EVENT_MOUSEMOVE, 1, 2, 0, None)
    mouse.on_event(cv2.EVENT_RBUTTONDOWN, 3, 4, 0, None)
    mouse.on_event(cv2.EVENT_LBUTTONUP, 5, 6, 0, None)
    assert mouse.drain() == []
    mouse.on_event(cv2.EVENT_LBUTTONDOWN, 7, 8, 0, None)
    mouse.on_event(cv2.EVENT_LBUTTONDBLCLK, 7, 8, 0, None)
    assert mouse.drain() == [(7, 8, 100.0), (7, 8, 100.0)]
    assert mouse.drain() == []


@pytest.mark.parametrize("key", [27, ord("q"), ord("Q")])
def test_mouse_loop_stops_on_quit_keys(key):
    session = _small_session()
    assert tr.mouse_loop(session, FakeDisplay(keys=[255, key]), FakeClock()) is False
    assert not session.finished


def test_mouse_loop_stops_when_the_window_is_closed():
    def close_it(display):
        display.is_closed = True

    assert tr.mouse_loop(_small_session(), FakeDisplay(on_wait=close_it), FakeClock()) is False


class FakeCamera:
    def __init__(self):
        self.seq = 0
        self.released = 0

    def read_if_new(self, last_seq):
        self.seq += 1
        return self.seq, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released += 1


class ScriptedHand:
    """Stands in for the tracker: aims the hand at the current target, then pinches."""

    SETTLE_FRAMES = 30      # let the cursor filter arrive before pressing
    POINT = dict(index=0.05, middle=0.9, ring=0.9, pinky=0.9)
    PINCH = dict(index=0.45, middle=0.7, ring=0.7, pinky=0.7)

    def __init__(self, session, box, clock=None, aspect=4 / 3):
        self.session = session
        self.box = box
        self.clock = clock or FakeClock()
        self.rng = np.random.default_rng(0)
        self.aspect = aspect
        self.key = None
        self.frames = 0
        self.last_ms = -1
        self.closed = 0
        template = make_hand("point", self.rng, pose=HandPose(0.5, 0.55), curls=self.POINT, noise=0)
        ax, ay = anchor_point(template)
        self.offset = (0.5 - ax, 0.55 - ay)

    def _pose_for(self, target):
        x0, y0, bw, bh = self.box
        width, height = self.session.size
        ax = x0 + bw * target[0] / (width - 1)
        ay = y0 + bh * target[1] / (height - 1)
        return HandPose(ax + self.offset[0], ay + self.offset[1])

    def detect(self, frame, t_ms):
        assert t_ms > self.last_ms, "MediaPipe needs strictly increasing timestamps"
        self.last_ms = t_ms
        self.clock.t += 1 / 30
        target = self.session.current_target
        if target is None:
            return []
        key = (self.session.index, self.session.trial_number)
        if key != self.key:
            self.key, self.frames = key, 0
        self.frames += 1
        pose = self._pose_for(target)
        if self.frames <= self.SETTLE_FRAMES:
            return [make_hand("point", self.rng, pose=pose, curls=self.POINT, noise=0)]
        return [make_hand("pinch_index", self.rng, pose=pose, curls=self.PINCH, thumb_gap=0.0, noise=0)]

    def close(self):
        self.closed += 1


def _no_real_mouse(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the real mouse backend must never be created")

    monkeypatch.setattr(controller_module, "make_backend", forbidden)
    monkeypatch.setattr(controller_module.Win32Backend, "__init__", forbidden)
    monkeypatch.setattr(controller_module.PynputBackend, "__init__", forbidden)


def test_gesture_loop_selects_targets_with_a_synthetic_hand(monkeypatch):
    _no_real_mouse(monkeypatch)
    cfg = Config()
    session = _small_session()
    records = []
    session.on_trial = records.append
    backend = tr.WindowBackend(*session.size)
    engine = GestureEngine(cfg, backend)
    clock = FakeClock()
    tracker = ScriptedHand(session, engine.controller.active_box, clock)
    camera, display = FakeCamera(), FakeDisplay()
    assert tr.gesture_loop(session, engine, backend, camera, tracker, display, clock=clock) is True
    assert session.finished
    assert len(records) == 5
    assert all(r.hit for r in records)
    # one pinch per target: settle time plus the press frame, at 30 frames a second
    assert all(900 < r.movement_time_ms < 1200 for r in records)
    assert display.shown > 100


def test_gesture_loop_stops_on_quit_key():
    session = _small_session()
    backend = tr.WindowBackend(*session.size)
    engine = GestureEngine(Config(), backend)
    tracker = ScriptedHand(session, engine.controller.active_box)
    display = FakeDisplay(keys=[255, 255, 27])
    assert tr.gesture_loop(session, engine, backend, FakeCamera(), tracker, display) is False
    assert not session.finished


def test_gesture_loop_ignores_right_button_presses():
    session = _small_session()
    backend = tr.WindowBackend(*session.size)
    engine = GestureEngine(Config(), backend)
    start = session.current_target
    backend.move(*map(round, start))
    backend.press("right")

    class OneFrame(ScriptedHand):
        def detect(self, frame, t_ms):
            return []

    display = FakeDisplay(keys=[27])
    tr.gesture_loop(session, engine, backend, FakeCamera(), OneFrame(session, engine.controller.active_box), display)
    assert session.waiting      # the right press did not count as selecting the start target


def test_run_gesture_imports_lazily_and_always_cleans_up(monkeypatch):
    _no_real_mouse(monkeypatch)
    session = _small_session()
    cameras, trackers = [], []

    class FakeThreadedCamera(FakeCamera):
        def __init__(self, index, width, height, fps):
            super().__init__()
            self.args = (index, width, height, fps)
            cameras.append(self)

        def start(self):
            return self

    class FakeTracker:
        def __init__(self, cfg):
            assert cfg.max_hands == 1 and cfg.dwell_enabled is False
            box = MouseController(cfg, tr.WindowBackend(*session.size)).active_box
            self.inner = ScriptedHand(session, box)
            trackers.append(self.inner)

        def detect(self, frame, t_ms):
            return self.inner.detect(frame, t_ms)

        def close(self):
            self.inner.close()

    monkeypatch.setitem(sys.modules, "airtouch.camera", types.SimpleNamespace(ThreadedCamera=FakeThreadedCamera))
    monkeypatch.setitem(sys.modules, "airtouch.hand_tracker", types.SimpleNamespace(HandTracker=FakeTracker))
    cfg = Config(max_hands=2, dwell_enabled=True, camera_index=3)
    # stop at once: the camera and tracker must still be let go of
    assert tr.run_gesture(session, FakeDisplay(keys=[27]), cfg) is False
    assert cameras[0].args == (3, 640, 480, 30)
    assert cameras[0].released == 1 and trackers[0].closed == 1

    # the real camera / tracker errors surface as an exception after clean-up
    class BrokenTracker:
        def __init__(self, cfg):
            raise RuntimeError("no model")

    monkeypatch.setitem(sys.modules, "airtouch.hand_tracker", types.SimpleNamespace(HandTracker=BrokenTracker))
    with pytest.raises(RuntimeError):
        tr.run_gesture(session, FakeDisplay(), cfg)
    assert cameras[1].released == 1


def test_importing_the_kit_loads_no_camera_or_tracker_code():
    code = (
        "import sys\n"
        "import evaluation.task_runner, evaluation.analyze_results, evaluation.sus_score\n"
        "loaded = [m for m in ('mediapipe', 'airtouch.camera', 'airtouch.hand_tracker') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True, timeout=120)


def test_sus_and_analysis_do_not_even_need_opencv():
    code = (
        "import sys\n"
        "import evaluation.analyze_results, evaluation.sus_score\n"
        "assert 'cv2' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True, timeout=120)


# ------------------------------------------------------------------------- CLI
def _drive_session(session, display):
    """Replaces the window loop: selects every target dead centre."""
    _finish(session)
    return True


def _mouse_cli(tmp_path, *extra):
    return ["--participant", "P01", "--input", "mouse", "--blocks", "1", "--targets", "3",
            "--results", str(tmp_path), *extra]


def test_cli_full_mouse_session_saves_one_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tr, "run_mouse", _drive_session)
    displays = []

    def factory():
        displays.append(FakeDisplay())
        return displays[-1]

    assert tr.main(_mouse_cli(tmp_path), display_factory=factory) == 0
    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 1
    assert re.fullmatch(r"P01_mouse_\d{8}-\d{6}\.csv", files[0].name)
    trials, _ = ar.load_trials(files)
    assert len(trials) == 9 * 3
    assert displays[0].close_calls == 1
    out = capsys.readouterr().out
    assert "已全部做完" in out and "结果已保存" in out and "点中 27 次" in out


def test_cli_practice_saves_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tr, "run_mouse", _drive_session)
    assert tr.main(_mouse_cli(tmp_path, "--practice"), display_factory=FakeDisplay) == 0
    assert list(tmp_path.iterdir()) == []
    assert "练习" in capsys.readouterr().out


def test_cli_stopping_before_any_trial_leaves_no_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tr, "run_mouse", lambda session, display: False)
    assert tr.main(_mouse_cli(tmp_path), display_factory=FakeDisplay) == 0
    assert list(tmp_path.iterdir()) == []
    out = capsys.readouterr().out
    assert "提前停止" in out and "没有保存" in out


def test_cli_crash_keeps_what_was_recorded_and_says_so_plainly(tmp_path, monkeypatch, capsys):
    def crash(session, display):
        session.select(*session.current_target, 1.0)
        session.select(*session.current_target, 2.0)
        raise RuntimeError("摄像头打不开")

    monkeypatch.setattr(tr, "run_mouse", crash)
    display = FakeDisplay()
    assert tr.main(_mouse_cli(tmp_path), display_factory=lambda: display) == 1
    assert "摄像头打不开" in capsys.readouterr().err
    assert display.close_calls == 1
    trials, _ = ar.load_trials(list(tmp_path.glob("*.csv")))
    assert len(trials) == 1


@pytest.mark.parametrize("args", [
    ["--participant", "../evil", "--input", "mouse"],
    ["--participant", "P01", "--input", "mouse", "--blocks", "0"],
    ["--participant", "P01", "--input", "mouse", "--targets", "4"],
    ["--participant", "P01", "--input", "mouse", "--size", "640x400"],
])
def test_cli_bad_settings_stop_before_any_window(args, tmp_path, capsys):
    def no_window():
        raise AssertionError("no window should open")

    assert tr.main(args + ["--results", str(tmp_path)], display_factory=no_window) == 2
    assert capsys.readouterr().err.strip()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("args", [
    ["--input", "mouse"],                                   # no participant
    ["--participant", "P01"],                               # no input
    ["--participant", "P01", "--input", "touchpad"],
    ["--participant", "P01", "--input", "mouse", "--size", "big"],
])
def test_cli_wrong_arguments_exit_cleanly(args, capsys):
    with pytest.raises(SystemExit) as e:
        tr.main(args, display_factory=FakeDisplay)
    assert e.value.code == 2


# ------------------------------------------------------------------- repo files
def test_config_example_is_in_sync_with_the_code():
    example = json.loads((PROJECT_ROOT / "config.example.json").read_text(encoding="utf-8"))
    assert example == dataclasses.asdict(Config())


def test_config_example_loads_cleanly():
    assert Config.load(PROJECT_ROOT / "config.example.json") == Config()


def test_gitignore_protects_private_and_generated_files():
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in (".venv/", "__pycache__/", "*.pyc", ".pytest_cache/", "evaluation/results/", "logs/",
                  "images/", "config.json", "models/hand_landmarker.task", "data/recorded.csv"):
        assert entry in lines
