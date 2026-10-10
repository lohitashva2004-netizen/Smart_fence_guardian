"""Gateway logic: classify -> validate periodicity -> localise -> alert."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ENERGIZER, PROP_VELOCITY_M_S
from .features import FEATURE_NAMES
from .localization import locate_toa
from .messaging import AlertMessage
from .pulse_train import PulseTrainValidator
from .sim import NodeReport


def group_reports(reports: list[NodeReport], window_s: float) -> list[list[NodeReport]]:
    """Cluster per-node detections that belong to the same physical event (timestamps within ``window_s``)."""
    seen = sorted((r for r in reports if r.arrival_s is not None), key=lambda r: r.arrival_s)
    groups: list[list[NodeReport]] = []
    for r in seen:
        if groups and r.arrival_s - groups[-1][-1].arrival_s <= window_s:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


@dataclass
class Guardian:
    model: object                               # any classifier with predict_proba over CLASSES
    major_node: str = "major"
    p_threshold: float = 0.5
    velocity: float = PROP_VELOCITY_M_S
    substation: str = "SS-01"
    feeder: str = "F-01"
    validator_kw: dict = field(default_factory=dict)

    def __post_init__(self):
        self.validator = PulseTrainValidator(**self.validator_kw)
        self.candidates: list[list[NodeReport]] = []
        self.n_events = 0
        self.n_candidates = 0
        self.alert: AlertMessage | None = None
        self.history: list[tuple] = []     # (timestamp, predicted class, P(energizer)) per processed event

    def handle_event(self, group: list[NodeReport]) -> AlertMessage | None:
        """Process one grouped event. Returns an alert the first time a fence is confirmed."""
        self.n_events += 1
        major = next((r for r in group if r.node_id == self.major_node and r.features is not None), None)
        if major is None:
            self.history.append((None, None, 0.0))
            return None
        p = self.model.predict_proba(pd.DataFrame([major.features], columns=FEATURE_NAMES))[0]
        classes = list(getattr(self.model, "classes_", range(len(p))))
        self.history.append((major.arrival_s, int(classes[int(np.argmax(p))]), float(p[classes.index(ENERGIZER)])))
        if p[classes.index(ENERGIZER)] < self.p_threshold or np.argmax(p) != classes.index(ENERGIZER):
            return None
        self.n_candidates += 1
        self.candidates.append(group)
        train = self.validator.update(major.arrival_s)
        if train is None or self.alert is not None:
            return None
        # Candidate list inside the validator is time-trimmed; map members back by timestamp.
        members = [g for g in self.candidates
                   if any(abs(next(r.arrival_s for r in g if r.node_id == self.major_node) - mt) < 1e-9
                          for mt in train.member_times)]
        xs = []
        for g in members:
            ok = [r for r in g if r.arrival_s is not None]
            if len(ok) >= 2:
                xs.append(locate_toa([r.position_m for r in ok], [r.arrival_s for r in ok], self.velocity).x_m)
        notes = [] if xs else ["too few sensors saw the pulse to localise"]
        self.alert = AlertMessage(self.substation, self.feeder, float(major.arrival_s), train.period_s,
                                  len(train.member_times), float(np.median(xs)) if xs else None, notes=notes)
        return self.alert
