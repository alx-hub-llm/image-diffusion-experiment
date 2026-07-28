# Branch 01 – queue-and-naming

## What changed

### Queue mode (default when no argument is given)
- Scan `experiments/pending/` for `*.xml`
- Process each file in turn
- On success move the XML to `experiments/done/`
- Failed files stay in `pending/` so they can be retried

### Single-file mode still works
```bash
python experiment.py path/to/file.xml
```

### Run folder naming
```
runs/<xml-stem>_<YYYYMMDD_HHMMSS>/
```
Example: `runs/astronaut_mars_20260728_161200/`

### New CLI flags
- `--pending-dir` (default `experiments/pending`)
- `--done-dir`   (default `experiments/done`)
- `--runs-dir`   (unchanged, default `runs`)

## How to test
```bash
mkdir -p experiments/pending experiments/done
cp experiments.example.xml experiments/pending/
python experiment.py          # processes the queue
ls runs/
ls experiments/done/
```
