# Branch 05 – resume-and-compare

Built on top of `04-metrics-csv-png-meta`.

## What this branch adds

### Resume / skip logic

Before starting an experiment the runner checks whether the target folder already contains the expected number of `.png` files.

- If yes → skip (print a short message) and keep the existing results.
- If no  → run as usual (or re-run if the previous attempt failed).

This makes it safe to re-launch a queue after an interruption or OOM.

### compare.py

Minimal CLI that loads two (or more) run folders and prints a side-by-side table of the key metrics from their `run_summary.json` / `metrics.csv` files.

```bash
python compare.py runs/runA_… runs/runB_…
```

Output is plain text (and optionally a small Markdown table) so you can quickly see which settings were faster / used less VRAM.

### Suggested merge order

1. `01-queue-and-naming`
2. `02-flatten-and-summary`
3. `03-richer-metadata`
4. `04-metrics-csv-png-meta`
5. `05-resume-and-compare`

Each branch is based on the previous one, so merging them sequentially keeps history clean.
