"""Cursor filters: steady when the hand is still, responsive when it moves fast."""

from __future__ import annotations

import math

import numpy as np
import pytest

from airtouch.smoothing import EMAFilter, MovingAverageFilter, OneEuroFilter

DT = 1 / 30       # the live app runs at about 30 frames per second


def run(filt, samples, dt: float = DT) -> np.ndarray:
    """Feed a sequence of samples at a constant frame rate; returns the outputs."""
    return np.array([filt(x, i * dt) for i, x in enumerate(samples)])


def app_filter() -> OneEuroFilter:
    """Filter with the app's default settings (Config.smoothing/beta/d_cutoff)."""
    return OneEuroFilter(min_cutoff=1.2, beta=1.0, d_cutoff=1.0)


# -------------------------------------------------------------------- One-Euro
def test_first_sample_passes_straight_through():
    assert float(app_filter()(0.7, 0.0)) == 0.7


def test_one_euro_converges_to_a_constant():
    out = run(app_filter(), [0.0] + [1.0] * 180)
    assert out[-1] == pytest.approx(1.0, abs=1e-3)


def test_one_euro_step_response_never_overshoots():
    out = run(OneEuroFilter(min_cutoff=1.2, beta=5.0, d_cutoff=1.0), [0.0] + [1.0] * 120)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert np.all(np.diff(out) >= -1e-12)        # keeps rising towards the target


def test_one_euro_reduces_jitter_on_a_still_noisy_signal():
    noisy = 0.5 + np.random.default_rng(0).normal(0, 0.003, 600)
    out = run(app_filter(), noisy)
    assert np.std(out[60:]) < 0.5 * np.std(noisy[60:])


def test_one_euro_stays_centred_on_a_still_noisy_signal():
    noisy = 0.5 + np.random.default_rng(1).normal(0, 0.003, 600)
    assert np.mean(run(app_filter(), noisy)[60:]) == pytest.approx(0.5, abs=1e-3)


def test_one_euro_lags_less_than_an_ema_with_the_same_stationary_jitter():
    # The whole point of One-Euro: for equal steadiness at rest, less lag in motion.
    noise = np.random.default_rng(0).normal(0, 0.002, 1200)

    def jitter(filt) -> float:
        return float(np.std(run(filt, noise)[100:]))

    target = jitter(app_filter())
    lo, hi = 0.01, 1.0                           # bisect the EMA alpha (bigger = more jitter)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if jitter(EMAFilter(mid)) < target else (lo, mid)
    alpha = (lo + hi) / 2
    assert jitter(EMAFilter(alpha)) == pytest.approx(target, rel=0.02)

    ramp = 1.5 * np.arange(60) * DT              # a fast, clean movement

    def lag(filt) -> float:
        return float(np.mean(ramp[30:] - run(filt, ramp)[30:]))

    assert 0 < lag(app_filter()) < 0.6 * lag(EMAFilter(alpha))


def test_one_euro_with_beta_zero_is_a_fixed_low_pass():
    # alpha from the published formula: tau = 1 / (2 pi f), alpha = 1 / (1 + tau / dt)
    cutoff = 2.0
    alpha = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / DT)
    samples = np.random.default_rng(2).uniform(0, 1, 50)
    one_euro = run(OneEuroFilter(min_cutoff=cutoff, beta=0.0), samples)
    ema = run(EMAFilter(alpha), samples)
    assert one_euro == pytest.approx(ema)


def test_higher_beta_means_less_lag():
    ramp = 1.0 * np.arange(60) * DT
    lags = [np.mean(ramp[30:] - run(OneEuroFilter(1.0, beta), ramp)[30:]) for beta in (0.0, 1.0, 5.0)]
    assert lags[0] > lags[1] > lags[2] > 0


def test_lower_cutoff_means_less_jitter():
    noisy = np.random.default_rng(3).normal(0, 0.01, 600)
    spread = [np.std(run(OneEuroFilter(cutoff, 0.0), noisy)[60:]) for cutoff in (5.0, 1.0, 0.2)]
    assert spread[0] > spread[1] > spread[2]


def test_reset_forgets_everything():
    f = app_filter()
    run(f, [0.0, 1.0, 2.0])
    f.reset()
    assert float(f(5.0, 100.0)) == 5.0          # like a brand-new filter: no gliding in
    fresh = app_filter()
    fresh(5.0, 100.0)
    assert float(f(6.0, 100.0 + DT)) == pytest.approx(float(fresh(6.0, 100.0 + DT)))


@pytest.mark.parametrize("next_t", [1.0, 0.5, -3.0], ids=["same-time", "earlier", "negative"])
def test_non_increasing_time_keeps_the_previous_estimate(next_t):
    f = app_filter()
    f(0.0, 0.9)
    previous = float(f(1.0, 1.0))
    out = float(f(1000.0, next_t))
    assert out == previous                      # no division by zero, no jump


def test_filter_recovers_after_a_bad_timestamp():
    f = app_filter()
    f(1.0, 1.0)
    f(50.0, 1.0)
    f(50.0, 0.5)
    out = float(f(2.0, 1.0 + DT))
    assert math.isfinite(out)
    assert 1.0 <= out <= 2.0                    # the ignored samples left no trace


def test_extreme_timing_stays_finite():
    f = app_filter()
    f(0.0, 0.0)
    assert math.isfinite(float(f(1.0, 1e-12)))  # nearly duplicate timestamp
    assert math.isfinite(float(f(1.0, 1e6)))    # very long gap


def test_vector_input_keeps_its_shape():
    f = app_filter()
    out = f(np.array([0.2, 0.8]), 0.0)
    assert out.shape == (2,)
    out = f(np.array([0.3, 0.7]), DT)
    assert out.shape == (2,)
    assert out[0] > 0.2 and out[1] < 0.8


def test_vector_axes_get_identical_smoothing():
    # The speed is the vector norm, so a straight-line move stays a straight line:
    # if y is always 2x at the input, it must be 2x at the output as well.
    f = app_filter()
    xs = np.random.default_rng(4).uniform(0, 1, 100)
    out = np.array([f(np.array([x, 2 * x]), i * DT) for i, x in enumerate(xs)])
    assert out[:, 1] == pytest.approx(2 * out[:, 0])


def test_changing_the_returned_array_does_not_disturb_the_filter():
    f = app_filter()
    first = f(np.array([1.0, 2.0]), 0.0)
    first[:] = 999.0
    second = f(np.array([1.0, 2.0]), DT)
    second[:] = -999.0
    assert f(np.array([1.0, 2.0]), 2 * DT) == pytest.approx([1.0, 2.0])


# ------------------------------------------------------------------------- EMA
def test_ema_formula():
    f = EMAFilter(alpha=0.25)
    assert float(f(1.0)) == 1.0                 # first sample passes through
    assert float(f(2.0)) == pytest.approx(0.25 * 2.0 + 0.75 * 1.0)
    assert float(f(2.0)) == pytest.approx(0.25 * 2.0 + 0.75 * 1.25)


def test_ema_alpha_one_passes_everything_through():
    samples = [0.3, 0.9, -1.0, 4.0]
    assert run(EMAFilter(1.0), samples) == pytest.approx(samples)


def test_ema_converges_and_reduces_jitter():
    assert run(EMAFilter(0.2), [0.0] + [1.0] * 100)[-1] == pytest.approx(1.0, abs=1e-6)
    noisy = np.random.default_rng(5).normal(0, 0.01, 600)
    assert np.std(run(EMAFilter(0.2), noisy)[60:]) < 0.5 * np.std(noisy[60:])


def test_ema_reset_and_vectors():
    f = EMAFilter(0.5)
    f(np.array([0.0, 0.0]))
    assert f(np.array([1.0, 2.0])) == pytest.approx([0.5, 1.0])
    f.reset()
    assert f(np.array([8.0, 9.0])) == pytest.approx([8.0, 9.0])


# -------------------------------------------------------------- moving average
def test_moving_average_uses_only_the_last_n_samples():
    f = MovingAverageFilter(3)
    assert [float(f(x)) for x in (3.0, 6.0, 9.0, 12.0)] == pytest.approx([3.0, 4.5, 6.0, 9.0])


def test_moving_average_window_is_at_least_one():
    for n in (1, 0, -5):
        f = MovingAverageFilter(n)
        assert f.n == 1
        assert [float(f(x)) for x in (1.0, 2.0, 3.0)] == pytest.approx([1.0, 2.0, 3.0])


def test_moving_average_reset_and_vectors():
    f = MovingAverageFilter(2)
    f(np.array([0.0, 2.0]))
    assert f(np.array([2.0, 4.0])) == pytest.approx([1.0, 3.0])
    f.reset()
    assert f(np.array([8.0, 9.0])) == pytest.approx([8.0, 9.0])


def test_moving_average_reduces_jitter():
    noisy = np.random.default_rng(6).normal(0, 0.01, 600)
    assert np.std(run(MovingAverageFilter(5), noisy)[10:]) < 0.6 * np.std(noisy[10:])


@pytest.mark.parametrize("filt", [OneEuroFilter(1.0), EMAFilter(0.5), MovingAverageFilter(3)],
                         ids=["one-euro", "ema", "moving-average"])
def test_all_filters_share_one_call_signature(filt):
    filt.reset()
    assert float(filt(0.5, 0.0)) == 0.5
    assert float(filt(0.5, DT)) == pytest.approx(0.5)
