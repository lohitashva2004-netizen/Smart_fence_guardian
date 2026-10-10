"""End-to-end experiments: every number and figure in the README is produced here."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from . import plotting as P
from .classifier import evaluate, export_tree_to_c, train_models
from .config import CLASSES, ENERGIZER, FEEDER_LENGTH_M, FS_HZ, PROP_VELOCITY_M_S, SEED
from .dataset import build_dataset
from .features import FEATURE_NAMES
from .localization import locate_toa
from .scenario import DEFAULT_NODES, ScenarioConfig, run_scenario
from .signals import sample_event
from .sim import observe_event

FULL = dict(n_train=800, n_test=400, snr_levels=[50, 40, 30, 20, 15, 10], n_snr=250, fs_list=[50e3, 100e3, 200e3, 500e3],
            n_fs_train=500, n_fs_test=250, loc_fs=[50e3, 100e3, 200e3, 500e3, 1e6], loc_sync=[1e-7, 5e-7, 1e-6, 2e-6, 5e-6],
            n_loc=100, n_scatter=300, n_sensors=200, span_pos=list(range(0, 12001, 500)), n_span=30,
            e2e_snrs=[35, 25, 15, 10], n_e2e=60, neg_hours_35=40, neg_hours_20=10)
QUICK = dict(n_train=120, n_test=60, snr_levels=[40, 20], n_snr=40, fs_list=[50e3, 200e3], n_fs_train=100, n_fs_test=50,
             loc_fs=[50e3, 200e3], loc_sync=[1e-6], n_loc=15, n_scatter=20, n_sensors=15, span_pos=[0, 6000, 12000], n_span=5,
             e2e_snrs=[35], n_e2e=6, neg_hours_35=1, neg_hours_20=1)


def localization_trials(n, nodes, fs, sigma_sync, rng, *, snr_db=35.0, x_range=(300.0, FEEDER_LENGTH_M - 300.0), window_s=0.004):
    truth, est, missed = [], [], 0
    for _ in range(n):
        xt = rng.uniform(*x_range)
        reps = observe_event(sample_event("energizer", rng), xt, 0.0, nodes, rng, fs=fs, window_s=window_s,
                             snr_db=snr_db, sigma_sync_s=sigma_sync)
        ok = [r for r in reps if r.arrival_s is not None]
        if len(ok) < 2:
            missed += 1
            continue
        est.append(locate_toa([r.position_m for r in ok], [r.arrival_s for r in ok], PROP_VELOCITY_M_S).x_m)
        truth.append(xt)
    return np.array(truth), np.array(est), missed


def _stats(err):
    e = np.abs(err)
    return dict(median_m=float(np.median(e)), p90_m=float(np.percentile(e, 90)), bias_m=float(np.mean(err)), n=int(len(e)))


def run(output_dir="results", firmware_dir="firmware/esp32_node/src", quick: bool = False, verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    cfg = QUICK if quick else FULL
    out = Path(output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    M: dict = {"config": {"seed": SEED, "fs_hz": FS_HZ, "quick": quick}}

    # 1. Signals and classification ----------------------------------------
    log("[1/6] signal gallery + classifier")
    P.signal_gallery(fig_dir)
    Xtr, ytr = build_dataset(cfg["n_train"], seed=1)
    Xte, yte = build_dataset(cfg["n_test"], seed=2)
    models = train_models(Xtr, ytr)
    M["classification"] = {name: evaluate(m, Xte, yte) for name, m in models.items()}
    rf = models["Random Forest (gateway)"]
    tree = models["Decision Tree (edge, depth 6)"]
    P.feature_space(Xte, yte, fig_dir)
    P.confusion(M["classification"]["Random Forest (gateway)"]["confusion_matrix"], fig_dir)
    P.importance(rf, fig_dir)
    M["rf_feature_importance"] = dict(zip(FEATURE_NAMES, map(float, rf.feature_importances_)))
    fw = Path(firmware_dir)
    fw.mkdir(parents=True, exist_ok=True)
    (fw / "fence_classifier.h").write_text(export_tree_to_c(tree, FEATURE_NAMES, CLASSES))

    # 2. Noise robustness ----------------------------------------------------
    log("[2/6] SNR robustness")
    snr_f1, snr_rec = [], []
    for s in cfg["snr_levels"]:
        X, y = build_dataset(cfg["n_snr"], snr_db=float(s), seed=100 + int(s))
        e = evaluate(rf, X, y)
        snr_f1.append(e["macro_f1"]); snr_rec.append(e["energizer_recall"])
    P.snr_robustness(cfg["snr_levels"], snr_f1, snr_rec, fig_dir)
    M["snr_robustness"] = dict(snr_db=cfg["snr_levels"], macro_f1=snr_f1, energizer_recall=snr_rec)

    # 3. Sampling-rate study -------------------------------------------------
    log("[3/6] sampling-rate study")
    fs_f1, fs_rec = [], []
    for fs in cfg["fs_list"]:
        Xa, ya = build_dataset(cfg["n_fs_train"], fs=fs, seed=11)
        Xb, yb = build_dataset(cfg["n_fs_test"], fs=fs, seed=12)
        m = train_models(Xa, ya)["Random Forest (gateway)"]
        e = evaluate(m, Xb, yb)
        fs_f1.append(e["macro_f1"]); fs_rec.append(e["energizer_recall"])
    P.sampling_rate(cfg["fs_list"], fs_f1, fs_rec, fig_dir)
    M["sampling_rate"] = dict(fs_hz=cfg["fs_list"], macro_f1=fs_f1, energizer_recall=fs_rec)

    # 4. Localisation ----------------------------------------------------------
    log("[4/6] localisation studies")
    rng = np.random.default_rng(SEED)
    grid = np.zeros((len(cfg["loc_fs"]), len(cfg["loc_sync"])))
    for i, fs in enumerate(cfg["loc_fs"]):
        for j, sg in enumerate(cfg["loc_sync"]):
            t, e, _ = localization_trials(cfg["n_loc"], DEFAULT_NODES, fs, sg, rng)
            grid[i, j] = np.median(np.abs(e - t))
    P.localization_heatmap(cfg["loc_fs"], cfg["loc_sync"], grid, fig_dir)
    M["localization_grid"] = dict(fs_hz=cfg["loc_fs"], sync_s=cfg["loc_sync"], median_error_m=grid.tolist())

    t, e, missed = localization_trials(cfg["n_scatter"], DEFAULT_NODES, 200e3, 1e-6, rng)
    P.localization_scatter(t, e, fig_dir, "3 sensors, 200 kS/s, 1 us sync")
    M["localization_default"] = {**_stats(e - t), "missed_detections": missed}

    sets = {"2 sensors": [("major", 0.0), ("sub2", 10_000.0)], "3 sensors": DEFAULT_NODES,
            "4 sensors": [("major", 0.0), ("s1", 10_000 / 3), ("s2", 20_000 / 3), ("sub2", 10_000.0)]}
    med, p90, cnt = [], [], {}
    for name, nodes in sets.items():
        t2, e2, _ = localization_trials(cfg["n_sensors"], nodes, 200e3, 1e-6, rng)
        s = _stats(e2 - t2); med.append(s["median_m"]); p90.append(s["p90_m"]); cnt[name] = s
    P.sensor_count(list(sets), med, p90, fig_dir)
    M["localization_vs_sensor_count"] = cnt

    xs = cfg["span_pos"]; smed, sp90 = [], []
    for x in xs:
        errs = []
        for _ in range(cfg["n_span"]):
            reps = observe_event(sample_event("energizer", rng), float(x), 0.0, DEFAULT_NODES, rng, fs=200e3, window_s=0.004)
            ok = [r for r in reps if r.arrival_s is not None]
            if len(ok) >= 2:
                errs.append(abs(locate_toa([r.position_m for r in ok], [r.arrival_s for r in ok], PROP_VELOCITY_M_S).x_m - x))
        smed.append(float(np.median(errs))); sp90.append(float(np.percentile(errs, 90)))
    P.span_ambiguity(xs, smed, sp90, [p for _, p in DEFAULT_NODES], fig_dir)
    M["span_ambiguity"] = dict(true_x_m=xs, median_error_m=smed, p90_error_m=sp90)

    # 5. End-to-end ------------------------------------------------------------
    log("[5/6] end-to-end scenarios")
    rf.set_params(n_jobs=1)
    e2e = {}
    for snr in cfg["e2e_snrs"]:
        det, tta, lerr = 0, [], []
        for _ in range(cfg["n_e2e"]):
            c = ScenarioConfig(fence_x_m=float(rng.uniform(300, 9700)), fence_period_s=float(rng.uniform(1.0, 1.6)), snr_db=float(snr))
            r = run_scenario(rf, c, rng)
            a = r["alert"]
            if a is not None:
                det += 1
                first = min(ev["t"] for ev in r["events"] if ev["kind"] == "energizer")
                tta.append(a.raised_at_s - first)
                if a.location_m is not None:
                    lerr.append(a.location_m - c.fence_x_m)
        e2e[snr] = dict(detection_rate=det / cfg["n_e2e"], n_runs=cfg["n_e2e"],
                        time_to_alert_s=tta, location_error_m=lerr,
                        median_abs_loc_error_m=float(np.median(np.abs(lerr))) if lerr else None)
    main = e2e[cfg["e2e_snrs"][0]]
    M["end_to_end"] = {str(k): {kk: vv for kk, vv in v.items() if kk not in ("time_to_alert_s", "location_error_m")} |
                       dict(median_time_to_alert_s=float(np.median(v["time_to_alert_s"])) if v["time_to_alert_s"] else None,
                            p90_time_to_alert_s=float(np.percentile(v["time_to_alert_s"], 90)) if v["time_to_alert_s"] else None,
                            median_abs_loc_error_m=float(np.median(np.abs(v["location_error_m"]))) if v["location_error_m"] else None,
                            p90_abs_loc_error_m=float(np.percentile(np.abs(v["location_error_m"]), 90)) if v["location_error_m"] else None)
                       for k, v in e2e.items()}
    P.snr_detection(cfg["e2e_snrs"], [e2e[s]["detection_rate"] for s in cfg["e2e_snrs"]],
                    [e2e[s]["median_abs_loc_error_m"] or 0 for s in cfg["e2e_snrs"]], fig_dir)

    # 6. False alarms -----------------------------------------------------------
    log("[6/6] false-alarm study")
    fa = {}
    for label, snr, hours in (("snr35", 35.0, cfg["neg_hours_35"]), ("snr20", 20.0, cfg["neg_hours_20"])):
        alerts = ev = cand = 0
        for _ in range(hours):
            c = ScenarioConfig(duration_s=3600, fence_x_m=None, lightning_rate_hz=1 / 30, motor_rate_hz=1 / 60,
                               switching_rate_hz=1 / 30, snr_db=snr, stop_at_alert=False)
            r = run_scenario(rf, c, rng)
            alerts += r["alert"] is not None
            ev += r["guardian"].n_events
            cand += r["guardian"].n_candidates
        fa[label] = dict(simulated_hours=hours, events=ev, single_event_false_candidates=cand,
                         false_candidate_rate=cand / max(ev, 1), false_alerts=alerts)
    M["false_alarms"] = fa
    main_fa = fa["snr35"]
    P.end_to_end(main["time_to_alert_s"], main["location_error_m"],
                 {"single-event false\ncandidates": main_fa["single_event_false_candidates"] / main_fa["simulated_hours"],
                  "system false\nalerts": main_fa["false_alerts"] / main_fa["simulated_hours"]}, fig_dir)

    # Timeline example --------------------------------------------------------------
    c = ScenarioConfig(duration_s=14, fence_x_m=6200.0, fence_period_s=1.3, lightning_rate_hz=0.3, motor_rate_hz=0.2,
                       switching_rate_hz=0.3, stop_at_alert=False)
    r = run_scenario(rf, c, np.random.default_rng(3))
    P.timeline(r["guardian"], r["events"], r["alert"], fig_dir)

    (out / "metrics.json").write_text(json.dumps(M, indent=2, default=float))
    return M


if __name__ == "__main__":
    run()
