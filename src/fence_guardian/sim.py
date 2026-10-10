"""Multi-sensor feeder simulation: renders one event at every sensor with propagation delay, attenuation and sync error."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ATTEN_LENGTH_M, FS_HZ, NOISE_REGION_S, PROP_VELOCITY_M_S
from .features import extract_features
from .localization import estimate_onset
from .signals import EventSpec, render_window


@dataclass
class NodeReport:
    node_id: str
    position_m: float
    arrival_s: float | None          # estimated absolute onset time (None = pulse not detected)
    features: np.ndarray | None = None


def observe_event(spec: EventSpec, x_event_m: float, t_event_s: float, nodes: list[tuple[str, float]],
                  rng: np.random.Generator, *, fs: float = FS_HZ, window_s: float = 0.010, snr_db: float = 35.0,
                  sigma_sync_s: float = 1e-6, v: float = PROP_VELOCITY_M_S, atten_len_m: float | None = ATTEN_LENGTH_M,
                  features_for: str | None = None) -> list[NodeReport]:
    """Render ``spec`` at each node: propagation delay, amplitude loss, asynchronous sampling clock, sync error."""
    reports = []
    for node_id, pos in nodes:
        d = abs(x_event_m - pos)
        amp = 1.0 if atten_len_m is None else float(np.exp(-d / atten_len_m))
        onset_true = 0.0025 + rng.uniform(0, 1) / fs        # event is asynchronous to this node's sample clock
        x = render_window(spec, rng, fs=fs, window_s=window_s, onset_s=onset_true, snr_db=snr_db, amp_scale=amp)
        onset_hat = estimate_onset(x, fs)
        arrival = None
        if onset_hat is not None:
            arrival = t_event_s + d / v + (onset_hat - onset_true) + rng.normal(0.0, sigma_sync_s)
        feats = extract_features(x, fs) if node_id == features_for else None
        reports.append(NodeReport(node_id, pos, arrival, feats))
    return reports
