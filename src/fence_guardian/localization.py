"""Time-of-arrival (TOA) localisation of a pulse along a feeder."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NOISE_REGION_S
from .features import highpass, noise_sigma


def estimate_onset(x: np.ndarray, fs: float, k_sigma: float = 6.0, min_threshold_pu: float = 0.02) -> float | None:
    """First threshold crossing of the high-passed window, with linear interpolation (seconds in window)."""
    hp = np.abs(highpass(x, fs))
    thr = max(k_sigma * noise_sigma(highpass(x, fs), fs), min_threshold_pu)
    start = int(NOISE_REGION_S[1] * fs)
    idx = np.flatnonzero(hp[start:] > thr)
    if idx.size == 0:
        return None
    i = start + int(idx[0])
    if i == 0 or hp[i] == hp[i - 1]:
        return i / fs
    return (i - 1 + (thr - hp[i - 1]) / (hp[i] - hp[i - 1])) / fs


@dataclass
class LocationEstimate:
    x_m: float
    t0_s: float
    rms_residual_s: float


def locate_toa(positions, arrival_times, velocity: float, step_m: float = 1.0) -> LocationEstimate:
    """Least-squares position on a line, with the (unknown) emission time eliminated analytically.

    Only positions *between* the outermost sensors are searched: a source beyond them is unobservable
    from arrival-time differences alone (every sensor sees it at the same relative delay).
    """
    pos, t = np.asarray(positions, float), np.asarray(arrival_times, float)
    if len(pos) < 2:
        raise ValueError("need at least two sensors")
    xs = np.arange(pos.min(), pos.max() + step_m, step_m)
    r = t[None, :] - np.abs(xs[:, None] - pos[None, :]) / velocity
    t0 = r.mean(axis=1)
    cost = ((r - t0[:, None]) ** 2).sum(axis=1)
    k = int(np.argmin(cost))
    return LocationEstimate(float(xs[k]), float(t0[k]), float(np.sqrt(cost[k] / len(pos))))
