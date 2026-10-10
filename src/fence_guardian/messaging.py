"""Wire format for sensor-node -> gateway/cloud messages (JSON over MQTT or LoRa-WAN payloads)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 1


def event_topic(substation: str, feeder: str, node: str) -> str:
    return f"fence/{substation}/{feeder}/{node}/event"


def alert_topic(substation: str) -> str:
    return f"fence/{substation}/alert"


@dataclass
class EventMessage:
    node_id: str
    seq: int
    timestamp_ns: int            # GPS-disciplined onset time, nanoseconds since epoch
    features: list[float]
    edge_class: int | None = None
    schema: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "EventMessage":
        d = json.loads(s)
        if d.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {d.get('schema')}")
        return cls(**d)


@dataclass
class AlertMessage:
    substation: str
    feeder: str
    raised_at_s: float
    pulse_period_s: float
    n_pulses: int
    location_m: float | None
    location_method: str = "toa"
    notes: list[str] = field(default_factory=list)
    schema: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "AlertMessage":
        d = json.loads(s)
        if d.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {d.get('schema')}")
        return cls(**d)
