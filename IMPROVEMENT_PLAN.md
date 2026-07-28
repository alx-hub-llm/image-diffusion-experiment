# Improvement Plan – Image Diffusion Experiment Framework

Concise roadmap for the next iterations. Focus: better input handling, cleaner output layout, richer metadata, and easy comparison across runs.

---

## 1. Immediate structural changes (next commit)

### 1.1 Input queue
- `experiments/pending/` – drop any number of `.xml` files here.
- `experiments/done/` – completed XMLs are moved here (with optional timestamp suffix).
- Runner scans `pending/`, processes each file in turn (or in parallel later), then moves it.
- Optional: `--watch` mode that polls / uses inotify and keeps running.

### 1.2 Run folder naming
- Current: `runs/YYYYMMDD_HHMMSS/`
- New: `runs/<xml-stem>_<YYYYMMDD_HHMMSS>/`  
  Example: `runs/astronaut_mars_20260728_161200/`
- Makes runs immediately identifiable by the experiment definition that produced them.

### 1.3 Flatten result layout
- Everything belonging to one experiment lives in **one folder**:
  ```
  runs/<xml-stem>_<ts>/
  ├── 001_modelname/
  │   ├── 0000.png
  │   ├── 0001.png
  │   ├── meta.json
  │   └── performance.log
  ├── 002_…
  └── run_summary.json          # aggregate of the whole XML
  ```
- No nested `images/` sub-folder unless the number of images becomes large.
- Copy (or symlink) the original XML into the run folder for full reproducibility.

---

## 2. Metadata collection – make it comparison-ready

### 2.1 Per-experiment (`meta.json`)
Keep current fields and add:
- Exact code version (git commit hash or `experiment.py` checksum)
- Full environment snapshot (torch / diffusers / CUDA driver / Python versions)
- Peak + average VRAM, seconds/image, total wall time
- Any warnings that were emitted (size / dtype / batch mismatches)
- Seed list actually used (important when seed is omitted)

### 2.2 Per-run aggregate (`run_summary.json`)
- List of all experiment indices + key metrics
- Total images, total time, overall average sec/image
- Pointers to the source XML and the individual experiment folders

### 2.3 Flat tabular export (for analysis)
- After every run (or on demand) emit `metrics.csv` / `metrics.parquet` with one row per image or per batch.
- Columns: run_id, experiment_index, model, dtype, width, height, steps, guidance, seed, sec_per_image, peak_vram_mb, prompt_hash, …
- Enables immediate `pandas` / DuckDB / spreadsheet comparison across dozens of runs.

### 2.4 Self-describing images
- Embed the most important metadata as PNG text chunks (or EXIF) so an image file alone still carries its prompt, model, seed and settings.

---

## 3. Input improvements – enable systematic comparison

### 3.1 Richer experiment definitions
- Support parameter grids inside one XML (or switch to YAML/TOML for readability):
  ```yaml
  matrix:
    model: [sd15, nsfw-v2]
    steps: [15, 20, 30]
    guidance: [5.0, 7.5, 9.0]
  ```
  The runner expands the cartesian product automatically and tags each combination.
- Add free-form `tags` / `group` fields so related experiments can be filtered later.
- Optional `compare_with: <previous-run-id>` field for explicit A/B linking.

### 3.2 Schema & validation
- Move from ad-hoc XML parsing to a Pydantic model (or jsonschema).
- Fail fast with clear messages instead of silent defaults.
- Keep XML as a supported format, but document YAML as the preferred one for complex matrices.

### 3.3 Model configuration outside code
- Extract `MODEL_CONFIGS` into `models.yaml` so new models and their optimal sizes / max-batch values can be added without editing Python.

---

## 4. Workflow & tooling improvements

| Idea | Benefit |
|------|--------|
| Central SQLite / DuckDB of all past runs | Query “show me all fp16 512² runs of model X ordered by sec/image” |
| `compare.py` CLI | Load two or more run folders → side-by-side table + optional HTML report with thumbnails |
| Automatic thumbnail grid + HTML summary per run | Quick visual QA without opening every PNG |
| Resume / skip logic | If an experiment folder already contains the expected number of images, skip it |
| Resource-aware scheduling | Detect free VRAM and choose batch size dynamically (or fall back to sequential CPU offload) |
| “Probe” mode | For a given model + size, automatically find the largest safe batch size and write it back into `models.yaml` |
| Git-aware provenance | Store the exact commit that produced a run; refuse to compare runs from different code versions without a flag |

---

## 5. Suggested implementation order

1. **Queue + naming + flatten** (section 1) – pure organisational win, low risk.
2. **Richer metadata + CSV export** (section 2) – unlocks all later analysis.
3. **YAML + matrix expansion + models.yaml** (section 3) – makes systematic experiments painless.
4. **compare.py + HTML report + optional DB** (section 4) – turns the collected data into insight.

---

## 6. Guiding principles

- Every artefact (image, log, meta) must be self-contained and relocatable.
- Prefer tabular / structured data over free-form logs for anything that will be compared.
- Keep the happy path dead simple: drop an XML/YAML into `pending/`, run the script, inspect `runs/`.
- Never require a database or extra services for the basic workflow; advanced tooling is optional.

This plan keeps the current strength (XML-driven, hardware-aware, performance-logged) while removing the friction that prevents rapid iteration and meaningful comparison.
