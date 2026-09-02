"""Run all experiments and write summary."""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    summary = [f"Run at {datetime.now().isoformat()}", ""]
    failed = []

    modules = [
        ("exp1_dijkstra", "exp1_dijkstra"),
        ("exp2_beneficial_cycle", "exp2_beneficial_cycle"),
        ("exp3_ara_pipeline", "exp3_ara_pipeline"),
        ("exp4_solvecore", "exp4_solvecore"),
        ("exp7_scale_synthetic", "exp7_scale_synthetic"),
        ("exp9_ara_decoy", "exp9_ara_decoy"),
        ("exp10_toy_bcn", "exp10_toy_bcn"),
        ("exp12_depth", "exp12_depth"),
        ("exp13_policy_gap", "exp13_policy_gap"),
        # Regression guard, not a paper experiment: this is what caught the SCC
        # and floating-point bugs in the detection path.
        ("exp11_detection", "exp11_detection"),
    ]

    for label, modname in modules:
        summary.append(f"=== {label} ===")
        try:
            mod = __import__(modname)
            res = mod.run()
            summary.append(f"OK: {res}")
        except Exception as e:
            failed.append(label)
            summary.append(f"FAILED: {e}")
            summary.append(traceback.format_exc())
        summary.append("")

    summary.append(f"Total failed: {len(failed)}")
    if failed:
        summary.append("Failed: " + ", ".join(failed))

    out = os.path.join(RESULTS, "summary.txt")
    with open(out, "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
