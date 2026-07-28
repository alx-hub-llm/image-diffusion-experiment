# Branch 04 – metrics-csv-png-meta

Built on top of `03-richer-metadata`.

## What this branch adds

### metrics.csv (one row per image / batch)

Written into the run folder:

```
run_id,experiment_index,model,dtype,width,height,steps,guidance,seed,batch,sec_per_image,peak_vram_mb,prompt_hash
...
```

Ready for `pandas`, spreadsheets or any later comparison script.

### Self-describing PNGs

Key metadata is embedded as PNG text chunks (via Pillow `PngInfo`):

- prompt / negative_prompt
- model
- seed
- dtype, size, steps, guidance
- run name + experiment index

An image file alone still carries its origin information.

### Implementation notes

- CSV is written (or appended) after each experiment finishes.
- PNG metadata is added at save time; no extra post-processing pass needed.
