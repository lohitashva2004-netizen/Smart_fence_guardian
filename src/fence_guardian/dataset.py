"""Labelled feature datasets built from the signal simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES, FS_HZ, SEED, WINDOW_S
from .features import FEATURE_NAMES, extract_features
from .signals import make_window


def build_dataset(n_per_class: int, *, fs: float = FS_HZ, snr_db=(20.0, 45.0), amp_range=(0.5, 1.0),
                  window_s: float = WINDOW_S, seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (features, integer labels). ``snr_db`` may be a (lo, hi) range or a single value.

    ``amp_range`` models attenuation between the event and the sensor.
    """
    rng = np.random.default_rng(seed)
    lo, hi = (snr_db, snr_db) if np.isscalar(snr_db) else snr_db
    rows, labels = [], []
    for label, kind in enumerate(CLASSES):
        for _ in range(n_per_class):
            x = make_window(kind, rng, fs=fs, window_s=window_s, snr_db=rng.uniform(lo, hi),
                            amp_scale=rng.uniform(*amp_range))
            rows.append(extract_features(x, fs))
            labels.append(label)
    return pd.DataFrame(rows, columns=FEATURE_NAMES), np.array(labels)
