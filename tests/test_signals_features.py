import numpy as np

from fence_guardian.config import CLASSES, FS_HZ
from fence_guardian.features import FEATURE_NAMES, extract_features
from fence_guardian.signals import make_window, quantise, sample_event, render_window


def test_window_shape_and_adc_range():
    rng = np.random.default_rng(0)
    for kind in CLASSES:
        x = make_window(kind, rng)
        assert x.shape == (int(0.010 * FS_HZ),)
        assert np.abs(x).max() <= 10.0


def test_quantisation_is_12_bit():
    step = 20.0 / 4096
    q = quantise(np.array([0.1234, -3.3333, 2.0]))
    assert np.allclose(q / step, np.round(q / step))


def test_fractional_onset_shifts_waveform_smoothly():
    rng = np.random.default_rng(1)
    spec = sample_event("energizer", rng)
    a = render_window(spec, np.random.default_rng(5), onset_s=0.0030, snr_db=80)
    b = render_window(spec, np.random.default_rng(5), onset_s=0.0030 + 0.5 / FS_HZ, snr_db=80)
    assert not np.allclose(a, b)          # half-sample shift changes the samples


def test_feature_vector_is_finite_and_named():
    rng = np.random.default_rng(2)
    for kind in CLASSES:
        f = extract_features(make_window(kind, rng), FS_HZ)
        assert f.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(f))


def test_class_signatures_are_physically_ordered():
    """Fast-edged events have shorter rise times than slow motor inrush; switching rings more than a fence pulse."""
    rng = np.random.default_rng(3)

    def med(kind, name):
        i = FEATURE_NAMES.index(name)
        return np.median([extract_features(make_window(kind, rng, snr_db=40), FS_HZ)[i] for _ in range(60)])

    assert med("energizer", "rise_time_us") < med("motor_start", "rise_time_us")
    assert med("switching", "ring_count") > med("energizer", "ring_count")
    assert med("lightning", "hp_peak_pu") > med("energizer", "hp_peak_pu")
