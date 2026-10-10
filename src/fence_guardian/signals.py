"""Synthetic sensor waveforms for the five event classes.

Everything is expressed in per-unit (pu) of the nominal 50 Hz peak. Event shapes are
*assumed* parametric models (double exponentials, damped sinusoids); they encode the
qualitative physics (fast edges for fence pulses and lightning, ringing for switching,
slow inrush for motors) but are not measured data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (ADC_BITS, ADC_FULL_SCALE_PU, CLASSES, FS_HZ, LINE_HZ, ONSET_RANGE_S, WINDOW_S)


@dataclass(frozen=True)
class EventSpec:
    """Shape parameters of one physical event, independent of where it is observed."""
    kind: str
    params: dict = field(default_factory=dict)


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_event(kind: str, rng: np.random.Generator) -> EventSpec:
    """Draw random shape parameters for an event class."""
    if kind == "background":
        # Occasionally a small impulsive-noise burst, so "no event" is not trivially clean.
        if rng.random() < 0.3:
            return EventSpec(kind, dict(A=_loguniform(rng, 0.05, 0.4), tau1=rng.uniform(2e-6, 10e-6),
                                        tau2=rng.uniform(30e-6, 100e-6)))
        return EventSpec(kind, {})
    if kind == "motor_start":
        return EventSpec(kind, dict(A=rng.uniform(1.0, 3.5), tr=rng.uniform(0.3e-3, 1.5e-3),
                                    tau=rng.uniform(15e-3, 80e-3), phi=rng.uniform(0, 2 * np.pi)))
    if kind == "lightning":
        ring = rng.random() < 0.5
        return EventSpec(kind, dict(A=_loguniform(rng, 2.0, 12.0), tau1=rng.uniform(1e-6, 12e-6),
                                    tau2=rng.uniform(30e-6, 150e-6), ring=ring,
                                    f_ring=rng.uniform(20e3, 60e3), tau_ring=rng.uniform(50e-6, 200e-6)))
    if kind == "switching":
        return EventSpec(kind, dict(A=rng.uniform(0.4, 3.0), f=rng.uniform(1e3, 12e3),
                                    tau=rng.uniform(0.2e-3, 1.5e-3), phi=rng.uniform(0, np.pi)))
    if kind == "energizer":
        ring = rng.random() < 0.3
        return EventSpec(kind, dict(A=_loguniform(rng, 0.4, 3.5), tau1=rng.uniform(6e-6, 25e-6),
                                    tau2=rng.uniform(90e-6, 350e-6), ring=ring,
                                    f_ring=rng.uniform(5e3, 30e3), tau_ring=rng.uniform(100e-6, 400e-6)))
    raise ValueError(f"unknown event kind: {kind}")


def _double_exp(dt: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    """Double exponential, zero before onset, normalised to unit peak."""
    out = np.zeros_like(dt)
    m = dt >= 0
    d = dt[m]
    tp = tau1 * tau2 / (tau2 - tau1) * np.log(tau2 / tau1)
    peak = np.exp(-tp / tau2) - np.exp(-tp / tau1)
    out[m] = (np.exp(-d / tau2) - np.exp(-d / tau1)) / peak
    return out


def render_event(spec: EventSpec, t: np.ndarray, onset: float, amp_scale: float = 1.0) -> np.ndarray:
    """Sample the event's waveform on time axis ``t`` (seconds) with the given onset time.

    The onset may fall *between* samples, so fractional arrival delays are exact.
    """
    dt = t - onset
    p = spec.params
    if spec.kind == "background":
        if not p:
            return np.zeros_like(t)
        return amp_scale * p["A"] * _double_exp(dt, p["tau1"], p["tau2"])
    if spec.kind == "motor_start":
        m = dt >= 0
        env = np.zeros_like(t)
        env[m] = (1 - np.exp(-dt[m] / p["tr"])) * np.exp(-dt[m] / p["tau"])
        return amp_scale * p["A"] * env * np.sin(2 * np.pi * LINE_HZ * t + p["phi"])
    if spec.kind in ("lightning", "energizer"):
        y = p["A"] * _double_exp(dt, p["tau1"], p["tau2"])
        if p["ring"]:
            m = dt >= 0
            r = np.zeros_like(t)
            r[m] = np.exp(-dt[m] / p["tau_ring"]) * np.sin(2 * np.pi * p["f_ring"] * dt[m])
            y = y + 0.25 * p["A"] * r
        return amp_scale * y
    if spec.kind == "switching":
        m = dt >= 0
        y = np.zeros_like(t)
        y[m] = np.exp(-dt[m] / p["tau"]) * np.sin(2 * np.pi * p["f"] * dt[m] + p["phi"])
        return amp_scale * p["A"] * y
    raise ValueError(spec.kind)


def render_background(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Nominal 50 Hz waveform with mild 3rd/5th harmonic distortion and random phase."""
    ph = rng.uniform(0, 2 * np.pi, 3)
    return (np.sin(2 * np.pi * LINE_HZ * t + ph[0])
            + rng.uniform(0.01, 0.08) * np.sin(2 * np.pi * 3 * LINE_HZ * t + ph[1])
            + rng.uniform(0.01, 0.05) * np.sin(2 * np.pi * 5 * LINE_HZ * t + ph[2]))


def quantise(x: np.ndarray, bits: int = ADC_BITS, full_scale: float = ADC_FULL_SCALE_PU) -> np.ndarray:
    step = 2 * full_scale / (2 ** bits)
    return np.clip(np.round(x / step) * step, -full_scale, full_scale - step)


def render_window(spec: EventSpec, rng: np.random.Generator, *, fs: float = FS_HZ, window_s: float = WINDOW_S,
                  onset_s: float | None = None, snr_db: float = 40.0, amp_scale: float = 1.0) -> np.ndarray:
    """One ADC capture: mains + event + white noise, then 12-bit quantisation.

    ``snr_db`` is the ratio of the 1 pu mains peak to the noise standard deviation.
    """
    n = int(round(window_s * fs))
    t = np.arange(n) / fs
    if onset_s is None:
        onset_s = rng.uniform(*ONSET_RANGE_S)
    x = render_background(t, rng) + render_event(spec, t, onset_s, amp_scale)
    x = x + rng.normal(0.0, 10 ** (-snr_db / 20.0), n)
    return quantise(x)


def make_window(kind: str, rng: np.random.Generator, **kw) -> np.ndarray:
    return render_window(sample_event(kind, rng), rng, **kw)
