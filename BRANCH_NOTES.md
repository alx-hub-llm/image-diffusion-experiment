# Branch 03 – richer-metadata

Built on top of `02-flatten-and-summary`.

## What this branch adds

### Expanded `meta.json`

Each experiment folder now records additional fields for reproducibility and comparison:

```json
{
  "environment": {
    "python": "3.12.x",
    "torch": "2.x.x+cu126",
    "diffusers": "…",
    "transformers": "…",
    "accelerate": "…",
    "cuda_driver": "580.xx"   // from nvidia-smi when available
  },
  "warnings": [
    "size 768×512 differs from optimal 512×512",
    "dtype=fp16 but recommended is fp32"
  ],
  "seeds_used": [42, 43, 44],
  "git_commit": "a1b2c3d"   // short hash if inside a git repo
}
```

### Implementation approach

- `check_against_model_config` returns a list of warning strings instead of only printing them.
- The list is stored in `meta.json` under the `warnings` key.
- After each batch the actual seed that was used is appended to `seeds_used`.
- A small helper gathers package versions and (optionally) the git HEAD.

These fields make later comparison across runs reliable and let you see exactly what the runner decided at runtime.
