#!/usr/bin/env python
"""Regenerate every metric and figure: python scripts/run_experiments.py [--quick]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fence_guardian.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sizes for smoke-testing (CI)")
    args = ap.parse_args()
    run(output_dir=ROOT / "results", firmware_dir=ROOT / "firmware/esp32_node/src", quick=args.quick)
    print("done ->", ROOT / "results")
