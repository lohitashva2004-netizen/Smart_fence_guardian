import numpy as np

from fence_guardian.pulse_train import PulseTrainValidator, find_pulse_train


def test_regular_train_is_detected():
    ts = 3.0 + 1.2 * np.arange(6)
    train = find_pulse_train(ts)
    assert train is not None and abs(train.period_s - 1.2) < 1e-6 and len(train.member_indices) == 6


def test_tolerates_missed_pulses_and_interleaved_noise():
    ts = list(3.0 + 1.25 * np.array([0, 1, 3, 4, 6, 7]))        # two pulses missed
    ts += [3.7, 4.1, 8.9, 10.33]                                  # unrelated false candidates
    train = find_pulse_train(sorted(ts))
    assert train is not None and abs(train.period_s - 1.25) < 5e-3 and len(train.member_indices) >= 5


def test_random_events_do_not_form_a_train():
    rng = np.random.default_rng(0)
    false_trains = sum(find_pulse_train(np.sort(rng.uniform(0, 30, 8))) is not None for _ in range(300))
    assert false_trains <= 3


def test_irregular_intervals_rejected():
    assert find_pulse_train([0, 1.0, 2.4, 3.1, 5.0, 6.9]) is None


def test_period_outside_range_rejected():
    assert find_pulse_train(0.2 * np.arange(10)) is None          # 5 Hz buzzer, not a fence energizer


def test_streaming_validator_confirms_on_fifth_pulse():
    v = PulseTrainValidator()
    results = [v.update(t) for t in 1.1 * np.arange(1, 7)]
    assert [r is not None for r in results] == [False, False, False, False, True, True]
