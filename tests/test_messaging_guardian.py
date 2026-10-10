import numpy as np
import pytest

from fence_guardian.guardian import group_reports
from fence_guardian.messaging import AlertMessage, EventMessage, alert_topic, event_topic
from fence_guardian.scenario import ScenarioConfig, run_scenario
from fence_guardian.sim import NodeReport


def test_event_message_roundtrip_and_topic():
    m = EventMessage("sub1", 7, 1_700_000_000_123_456_789, [0.1, 2.5], edge_class=4)
    assert EventMessage.from_json(m.to_json()) == m
    assert event_topic("SS-01", "F-02", "sub1") == "fence/SS-01/F-02/sub1/event"
    assert alert_topic("SS-01") == "fence/SS-01/alert"


def test_alert_roundtrip_and_schema_check():
    a = AlertMessage("SS-01", "F-01", 12.5, 1.2, 5, 3650.0)
    assert AlertMessage.from_json(a.to_json()) == a
    with pytest.raises(ValueError):
        AlertMessage.from_json('{"schema": 99}')


def test_group_reports_clusters_by_time():
    r = [NodeReport("a", 0, 1.000000), NodeReport("b", 5000, 1.000017), NodeReport("c", 1e4, 1.000034),
         NodeReport("a", 0, 2.5), NodeReport("b", 5000, None)]
    groups = group_reports(r, window_s=200e-6)
    assert [len(g) for g in groups] == [3, 1]


def test_fence_is_detected_and_localised(small_models):
    rf = small_models["Random Forest (gateway)"].set_params(n_jobs=1)
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(4):
        cfg = ScenarioConfig(fence_x_m=float(rng.uniform(800, 9200)), fence_period_s=float(rng.uniform(1.0, 1.6)))
        res = run_scenario(rf, cfg, rng)
        if res["alert"] is not None:
            hits += 1
            assert abs(res["alert"].location_m - cfg.fence_x_m) < 1500
            assert abs(res["alert"].pulse_period_s - cfg.fence_period_s) < 0.02
    assert hits >= 3


def test_no_fence_no_alert(small_models):
    rf = small_models["Random Forest (gateway)"].set_params(n_jobs=1)
    cfg = ScenarioConfig(duration_s=600, fence_x_m=None, lightning_rate_hz=1 / 20, motor_rate_hz=1 / 30,
                         switching_rate_hz=1 / 20, stop_at_alert=False)
    res = run_scenario(rf, cfg, np.random.default_rng(5))
    assert res["alert"] is None
