"""Event-stream scenarios: an (optional) illegal fence plus lightning, motor starts and switching surges."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FEEDER_LENGTH_M, FS_HZ
from .guardian import Guardian, group_reports
from .signals import sample_event
from .sim import observe_event

DEFAULT_NODES = [("major", 0.0), ("sub1", 5_000.0), ("sub2", FEEDER_LENGTH_M)]


@dataclass
class ScenarioConfig:
    duration_s: float = 60.0
    fence_x_m: float | None = 3_700.0           # None = no illegal fence on the feeder
    fence_period_s: float = 1.2
    fence_jitter_s: float = 0.002
    lightning_rate_hz: float = 1 / 60
    motor_rate_hz: float = 1 / 100
    switching_rate_hz: float = 1 / 60
    fs: float = FS_HZ
    snr_db: float = 35.0
    sigma_sync_s: float = 1e-6
    stop_at_alert: bool = True


def generate_events(cfg: ScenarioConfig, rng: np.random.Generator) -> list[dict]:
    ev = []
    for kind, rate in (("lightning", cfg.lightning_rate_hz), ("motor_start", cfg.motor_rate_hz),
                       ("switching", cfg.switching_rate_hz)):
        n = rng.poisson(rate * cfg.duration_s)
        for t in rng.uniform(0, cfg.duration_s, n):
            ev.append(dict(t=float(t), kind=kind, x=float(rng.uniform(0, FEEDER_LENGTH_M))))
    if cfg.fence_x_m is not None:
        t = rng.uniform(0, cfg.fence_period_s)
        while t < cfg.duration_s:
            ev.append(dict(t=float(t), kind="energizer", x=cfg.fence_x_m))
            t += cfg.fence_period_s + rng.normal(0, cfg.fence_jitter_s)
    return sorted(ev, key=lambda e: e["t"])


def run_scenario(model, cfg: ScenarioConfig, rng: np.random.Generator, nodes=DEFAULT_NODES, **guardian_kw) -> dict:
    guardian = Guardian(model, **guardian_kw)
    events = generate_events(cfg, rng)
    log = []
    for e in events:
        spec = sample_event(e["kind"], rng)
        reports = observe_event(spec, e["x"], e["t"], nodes, rng, fs=cfg.fs, snr_db=cfg.snr_db,
                                sigma_sync_s=cfg.sigma_sync_s, features_for="major")
        for grp in group_reports(reports, window_s=200e-6):   # 10 km / c is ~35 us; 200 us leaves margin
            alert = guardian.handle_event(grp)
            log.append((e["t"], e["kind"]))
            if alert and cfg.stop_at_alert:
                return dict(alert=alert, guardian=guardian, events=events, log=log)
    return dict(alert=guardian.alert, guardian=guardian, events=events, log=log)
