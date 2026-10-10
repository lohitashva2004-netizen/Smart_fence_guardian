"""Edge-friendly features from one capture window (all cheap enough for an ESP32-class MCU)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import kurtosis

from .config import LINE_HZ, NOISE_REGION_S

FEATURE_NAMES = [
    "hp_peak_pu", "peak_to_noise_db", "max_slew_pu_per_us", "rise_time_us", "width_us", "decay_us",
    "spec_centroid_khz", "hf_ratio", "dom_freq_khz", "ring_count", "lf_residual_rms", "raw_crest", "hp_kurtosis",
]


@lru_cache(maxsize=16)
def _hp_sos(fs: float, fc: float):
    return butter(2, fc, btype="highpass", fs=fs, output="sos")


def highpass(x: np.ndarray, fs: float, fc: float = 1000.0) -> np.ndarray:
    """Causal 2nd-order Butterworth high-pass, initialised to steady state to avoid a start-up transient."""
    sos = _hp_sos(float(fs), float(fc))
    y, _ = sosfilt(sos, x, zi=sosfilt_zi(sos) * x[0])
    return y


def noise_sigma(hp: np.ndarray, fs: float) -> float:
    """Noise floor from the event-free pre-trigger region of the high-passed window."""
    i0, i1 = (int(s * fs) for s in NOISE_REGION_S)
    return float(np.std(hp[i0:i1]) + 1e-9)


@lru_cache(maxsize=16)
def _mains_basis(n: int, fs: float):
    t = np.arange(n) / fs
    cols = [np.ones(n)]
    for h in (1, 3, 5):
        cols += [np.sin(2 * np.pi * h * LINE_HZ * t), np.cos(2 * np.pi * h * LINE_HZ * t)]
    return np.stack(cols, axis=1)


def _back_cross(a, ip, level):
    idx = np.flatnonzero(a[:ip] < level)
    if idx.size == 0:
        return 0.0
    i = idx[-1]
    return i + (level - a[i]) / (a[i + 1] - a[i])


def _fwd_cross(a, ip, level):
    idx = np.flatnonzero(a[ip + 1:] < level)
    if idx.size == 0:
        return float(len(a) - 1)
    i = ip + idx[0]
    return i + (a[i] - level) / (a[i] - a[i + 1])


def extract_features(x: np.ndarray, fs: float) -> np.ndarray:
    n = len(x)
    hp = highpass(x, fs)
    a = np.abs(hp)
    sigma = noise_sigma(hp, fs)
    ip = int(np.argmax(a[int(0.0003 * fs):])) + int(0.0003 * fs)   # skip filter settling
    peak = float(a[ip])
    win_us = n / fs * 1e6

    if peak > 4 * sigma and 0 < ip < n - 1:
        rise = (_back_cross(a, ip, 0.9 * peak) - _back_cross(a, ip, 0.1 * peak)) / fs * 1e6
        width = (_fwd_cross(a, ip, 0.5 * peak) - _back_cross(a, ip, 0.5 * peak)) / fs * 1e6
        decay = (_fwd_cross(a, ip, peak / np.e) - ip) / fs * 1e6
        seg = hp[ip: ip + int(2e-3 * fs)]
        s = np.sign(seg[np.abs(seg) > 0.15 * peak])
        rings = int(np.sum(s[1:] != s[:-1]))
    else:  # no clear event: use "very long / featureless" sentinels
        rise = width = decay = win_us
        rings = 0

    spec = np.abs(np.fft.rfft(hp * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    tot = spec.sum() + 1e-18
    centroid = float((f * spec).sum() / tot) / 1e3
    hf_ratio = float(spec[f > 10e3].sum() / tot)
    dom = float(f[int(np.argmax(spec))]) / 1e3

    basis = _mains_basis(n, float(fs))
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    lf_res = float(np.sqrt(np.mean((x - basis @ coef) ** 2)))

    return np.array([
        peak, 20 * np.log10(peak / sigma + 1e-9), float(np.max(np.abs(np.diff(x)))) * fs / 1e6,
        rise, width, decay, centroid, hf_ratio, dom, rings, lf_res,
        float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-9)), float(kurtosis(hp)),
    ], dtype=float)
