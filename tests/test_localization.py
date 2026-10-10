import numpy as np
import pytest

from fence_guardian.config import PROP_VELOCITY_M_S as V
from fence_guardian.localization import estimate_onset, locate_toa
from fence_guardian.signals import render_window, sample_event


def _arrivals(x_true, pos, t0=0.0):
    return [t0 + abs(x_true - p) / V for p in pos]


def test_exact_recovery_noise_free():
    pos = [0.0, 5000.0, 10000.0]
    for x in (250.0, 3700.0, 9100.0):
        est = locate_toa(pos, _arrivals(x, pos, t0=12.345), V)
        assert est.x_m == pytest.approx(x, abs=1.0)


def test_two_sensors_suffice_and_emission_time_is_unknown():
    est = locate_toa([0.0, 8000.0], _arrivals(2500.0, [0.0, 8000.0], t0=99.0), V)
    assert est.x_m == pytest.approx(2500.0, abs=1.0)


def test_source_outside_sensor_span_is_unobservable():
    pos = [0.0, 5000.0, 10000.0]
    est = locate_toa(pos, _arrivals(12000.0, pos), V)
    assert est.x_m <= 10000.0           # can never be placed beyond the last sensor


def test_requires_two_sensors():
    with pytest.raises(ValueError):
        locate_toa([0.0], [0.0], V)


def test_timing_error_translates_to_distance():
    pos = [0.0, 10000.0]
    t = _arrivals(5000.0, pos)
    t[1] += 1e-6                                       # 1 us late at one sensor
    assert locate_toa(pos, t, V).x_m == pytest.approx(5000.0 - V * 1e-6 / 2, abs=2.0)


def test_onset_estimate_is_close_to_truth_at_high_snr():
    rng = np.random.default_rng(4)
    errs = []
    for _ in range(30):
        spec = sample_event("energizer", rng)
        onset = 0.0025 + rng.uniform(0, 5e-6)
        x = render_window(spec, rng, onset_s=onset, snr_db=50)
        hat = estimate_onset(x, 200_000)
        assert hat is not None
        errs.append(hat - onset)
    assert np.median(np.abs(errs)) < 25e-6           # within ~5 samples at 200 kS/s
