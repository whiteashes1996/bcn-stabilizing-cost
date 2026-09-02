"""Run large-scale experiments exp6-8."""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    summary = [f"Large-scale run at {datetime.now().isoformat()}", ""]
    failed = []
    for label, modname in [
        ("exp6_scale_real", "exp6_scale_real"),
        ("exp7_scale_synthetic", "exp7_scale_synthetic"),
        ("exp8_dijkstra_warmstart", "exp8_dijkstra_warmstart"),
    ]:
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
    out = os.path.join(RESULTS, "summary_large.txt")
    with open(out, "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
