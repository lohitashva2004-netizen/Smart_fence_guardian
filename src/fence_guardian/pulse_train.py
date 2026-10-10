"""Temporal validation: a fence energizer fires at a *regular* interval; lightning, motors and switching do not.

The window classifier is imperfect at single-pulse level, so the system only raises an alert once enough
candidate pulses line up on a regular time grid (tolerating missed pulses and interleaved false candidates).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PulseTrain:
    period_s: float
    member_indices: list[int]      # indices into the candidate list passed to ``find_pulse_train``
    member_times: list[float]


def find_pulse_train(times, *, interval_range=(0.8, 2.5), tol_s=0.015, min_pulses=5, max_span_s=60.0) -> PulseTrain | None:
    """Best periodic chain among candidate pulse timestamps, or None if fewer than ``min_pulses`` align."""
    ts = np.asarray(times, dtype=float)
    n = len(ts)
    if n < min_pulses:
        return None
    best: PulseTrain | None = None
    for i in range(n - 1):
        for j in range(i + 1, n):
            period = ts[j] - ts[i]
            if period < interval_range[0]:
                continue
            if period > interval_range[1]:
                break
            members = _grid_members(ts, ts[i], period, tol_s)
            if len(members) >= min_pulses:        # refine period by least squares, then recount once
                k = np.round((ts[members] - ts[i]) / period)
                slope, icpt = np.polyfit(k, ts[members], 1)
                members = _grid_members(ts, icpt, slope, tol_s)
                period = slope
            if (len(members) >= min_pulses and interval_range[0] <= period <= interval_range[1]
                    and ts[members].max() - ts[members].min() <= max_span_s
                    and (best is None or len(members) > len(best.member_indices))):
                best = PulseTrain(float(period), members.tolist(), ts[members].tolist())
    return best


def _grid_members(ts, anchor, period, tol):
    k = np.round((ts - anchor) / period)
    resid = np.abs(ts - (anchor + k * period))
    idx = np.flatnonzero(resid <= tol)
    _, first = np.unique(k[idx], return_index=True)   # at most one pulse per grid slot
    return idx[first]


class PulseTrainValidator:
    """Streaming wrapper: feed candidate timestamps, get a PulseTrain once one is established."""

    def __init__(self, horizon_s: float = 30.0, **kw):
        self.horizon_s, self.kw, self.times = horizon_s, kw, []

    def update(self, t: float) -> PulseTrain | None:
        self.times = [x for x in self.times if t - x <= self.horizon_s] + [t]
        return find_pulse_train(self.times, **self.kw)
