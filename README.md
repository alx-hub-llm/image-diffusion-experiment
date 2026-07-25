# Image Diffusion Experiment (Stable Diffusion 1.5)

Experiment framework for **Stable Diffusion 1.5** text-to-image generation on **Fedora 42** with an **NVIDIA GTX 980 Ti** (Maxwell, compute capability 5.2, 6 GB VRAM).

## Contents

| File | Purpose |
|------|---------|
| `install_system.sh` | `dnf` – RPM Fusion NVIDIA **580xx** legacy drivers (Maxwell) + CUDA + Python |
| `install_python.sh` | Creates `.venv` and installs PyTorch **cu126** (still has `sm_52`) + Diffusers |
| `requirements.txt` | Python deps |
| `smoke_test.py` | Original single-image smoke test |
| **`experiment.py`** | **Main runner** – XML-driven multi-experiment batches with timing & meta |
| `experiments.example.xml` | Example experiment description |

## Hardware notes

- GTX 980 Ti → RPM Fusion package series **nvidia-580xx**.
- PyTorch CUDA 12.6 wheels still contain Maxwell (`sm_52`) kernels.
- 6 GB VRAM → stick to **fp32** + attention slicing; the code warns when you deviate from the optimal settings stored in `MODEL_CONFIGS`.

## Quick start (system + Python)

```bash
# 1. System packages (needs reboot)
sudo ./install_system.sh
sudo reboot

# 2. Verify driver
nvidia-smi

# 3. Python env
./install_python.sh
source .venv/bin/activate
```

## Running experiments

```bash
# Use the supplied example (or copy & edit it)
python experiment.py experiments.example.xml
```

### What happens

1. A **run folder** is created under `runs/YYYYMMDD_HHMMSS/`.
2. For every `<experiment>` in the XML a **sub-folder** is created:
   ```
   001_YYYYMMDD_HHMMSS_<prompt-slug>/
   ├── meta.json          # model, prompts, all settings
   ├── performance.log    # load time, per-batch timings, peak VRAM
   └── images/
       ├── 0000.png
       ├── 0001.png
       └── …
   ```
3. The script checks the requested size / dtype / batch_size against the hard-coded `MODEL_CONFIGS` table and prints a warning when they differ from the known-good values for this GPU.

### XML schema (minimal)

```xml
<run>
  <experiment>
    <model>stable-diffusion-v1-5/stable-diffusion-v1-5</model>
    <prompt>your positive prompt here</prompt>
    <!-- optional -->
    <negative_prompt>…</negative_prompt>
    <width>512</width>
    <height>512</height>
    <dtype>fp32</dtype>          <!-- fp16 | fp32 | bf16 -->
    <num_images>4</num_images>
    <batch_size>1</batch_size>
    <num_inference_steps>20</num_inference_steps>
    <guidance_scale>7.5</guidance_scale>
    <seed>42</seed>              <!-- omit for random -->
  </experiment>
  <!-- more <experiment> elements … -->
</run>
```

### Model configuration table

Edit the `MODEL_CONFIGS` dict at the top of `experiment.py` to add new models or refine the optimal size / max batch counts after you measure them on your card. The runner will automatically warn when an experiment requests values outside those limits.

## Performance data

Every experiment folder contains a `performance.log` that records:

- pipeline load time
- per-batch wall-clock time and seconds-per-image
- peak CUDA memory (MB)

Collect these logs across runs to decide the best `batch_size` / resolution / dtype trade-offs for your hardware.

## Troubleshooting

- **Driver not loading** – wait for akmods to finish, then reboot. Check `/var/log/akmods/akmods.log`.
- **OOM** – reduce `batch_size` or resolution; keep `dtype=fp32` on Maxwell.
- **First model download** – needs internet (~4 GB). Subsequent runs use the HF cache.

Happy experimenting!
