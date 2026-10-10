import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from fence_guardian.classifier import export_tree_to_c
from fence_guardian.config import CLASSES
from fence_guardian.dataset import build_dataset
from fence_guardian.features import FEATURE_NAMES
from sklearn.tree import DecisionTreeClassifier


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_exported_c_tree_matches_python_predictions(tmp_path):
    X, y = build_dataset(100, seed=21)
    tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=0).fit(X, y)
    (tmp_path / "fence_classifier.h").write_text(export_tree_to_c(tree, FEATURE_NAMES, CLASSES))
    (tmp_path / "main.c").write_text(textwrap.dedent("""
        #include <stdio.h>
        #include "fence_classifier.h"
        int main(void) {
            float f[FG_N_FEATURES];
            for (;;) {
                for (int i = 0; i < FG_N_FEATURES; i++) if (scanf("%f", &f[i]) != 1) return 0;
                printf("%d\\n", fg_classify(f));
            }
        }
    """))
    subprocess.run(["gcc", "-O1", "-o", str(tmp_path / "t"), str(tmp_path / "main.c")], check=True)
    Xt, _ = build_dataset(40, seed=22)
    stdin = "\n".join(" ".join(repr(float(np.float32(v))) for v in row) for row in Xt.values)
    out = subprocess.run([str(tmp_path / "t")], input=stdin, capture_output=True, text=True, check=True).stdout.split()
    assert [int(v) for v in out] == tree.predict(Xt).tolist()
