#!/usr/bin/env python
"""Print the state of every run directory: rate, loss and validation error.

    python scripts/run_status.py <runs-root>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    for run in sorted(root.glob("*/curve.jsonl")):
        rows = [json.loads(line) for line in run.read_text().splitlines() if line]
        train = [r for r in rows if "loss" in r]
        valid = [r for r in rows if "valid_rollout_mse_free" in r]
        if not train:
            continue
        last = train[-1]
        rate = ""
        if len(train) >= 2:
            previous = train[-2]
            delta_t = last["seconds"] - previous["seconds"]
            if delta_t > 0:
                rate = f"{(last['step'] - previous['step']) / delta_t:6.1f} it/s"
        best = min((r["valid_rollout_mse_free"] for r in valid), default=float("nan"))
        print(
            f"{run.parent.name:<24} step {last['step']:>9,}  "
            f"loss {last['loss']:.4f}  {rate}  "
            f"valid rollout best {best:.3e}  "
            f"{last['seconds'] / 3600:5.2f} h"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
