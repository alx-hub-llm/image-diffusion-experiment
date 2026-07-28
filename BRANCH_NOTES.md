# Branch 02 – flatten-and-summary

Built on top of `01-queue-and-naming`.

## What changed

### Flattened experiment folder
Images, `meta.json` and `performance.log` now live **in the same directory**:

```
runs/<xml-stem>_<ts>/
├── 001_<ts>_<model>/
│   ├── 0000.png
│   ├── 0001.png
│   ├── meta.json
│   └── performance.log
├── 002_…
├── <original>.xml          ← copy of the source XML
└── run_summary.json        ← aggregate of the whole run
```

No nested `images/` sub-folder any more.

### run_summary.json
Contains high-level info for every experiment in the run (model, image count, avg sec/image, size, dtype). Useful for quick overview and later comparison tools.

### Source XML preserved
The original experiment definition is copied into the run folder so the run is fully self-contained.
