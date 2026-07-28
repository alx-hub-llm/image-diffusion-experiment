#!/usr/bin/env python3
"""
compare.py – quick side-by-side comparison of two or more run folders.

Usage:
    python compare.py runs/runA_… runs/runB_… [runs/runC_…]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_summary(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_summary.json"
    if not p.is_file():
        return {"error": f"no run_summary.json in {run_dir}"}
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python compare.py <run_dir> [<run_dir> …]", file=sys.stderr)
        return 1

    runs = [Path(a) for a in argv[1:]]
    summaries = []
    for r in runs:
        if not r.is_dir():
            print(f"ERROR: not a directory: {r}", file=sys.stderr)
            return 1
        summaries.append((r.name, load_summary(r)))

    # Header
    print(f"{'run':<40} {'exps':>4} {'avg_s/img':>10} {'gpu':<20}")
    print("-" * 80)
    for name, s in summaries:
        if "error" in s:
            print(f"{name:<40}  {s['error']}")
            continue
        n = s.get("num_experiments", "?")
        # average of the per-experiment averages when available
        avgs = [e.get("avg_sec_per_image") for e in s.get("experiments", [])
                if e.get("avg_sec_per_image") is not None]
        avg = f"{sum(avgs)/len(avgs):.2f}" if avgs else "–"
        gpu = (s.get("gpu_name") or "?")[:20]
        print(f"{name:<40} {n:>4} {avg:>10} {gpu:<20}")

    # Per-experiment detail for the first two runs (if present)
    if len(summaries) >= 2:
        print("\nPer-experiment detail (first two runs):")
        a_name, a = summaries[0]
        b_name, b = summaries[1]
        a_exps = {e.get("index"): e for e in a.get("experiments", [])}
        b_exps = {e.get("index"): e for e in b.get("experiments", [])}
        indices = sorted(set(a_exps) | set(b_exps))
        print(f"{'idx':>3}  {'model':<35}  {a_name[:18]:>10}  {b_name[:18]:>10}")
        for i in indices:
            ae = a_exps.get(i, {})
            be = b_exps.get(i, {})
            model = (ae.get("model") or be.get("model") or "?")[:35]
            a_val = ae.get("avg_sec_per_image")
            b_val = be.get("avg_sec_per_image")
            a_s = f"{a_val:.2f}" if a_val is not None else "–"
            b_s = f"{b_val:.2f}" if b_val is not None else "–"
            print(f"{i:>3}  {model:<35}  {a_s:>10}  {b_s:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
