import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def small_models():
    from fence_guardian.classifier import train_models
    from fence_guardian.dataset import build_dataset

    X, y = build_dataset(150, seed=1)
    return train_models(X, y)
