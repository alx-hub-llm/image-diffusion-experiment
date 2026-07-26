#!/usr/bin/env python3
"""
experiment.py – Batch experiment runner for Stable Diffusion

Reads an XML description of experiments, runs them in a timestamped run
folder hierarchy, records images + metadata + performance logs.

Target hardware: GTX 980 Ti (6 GB, Maxwell sm_52) but configurable via
the MODEL_CONFIGS table.

Usage:
    python experiment.py experiments.xml
    python experiment.py path/to/my_experiments.xml
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import gc
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
from PIL import Image


# ---------------------------------------------------------------------------
# Model configuration table (optimal settings for this hardware)
# ---------------------------------------------------------------------------
# Keys are HuggingFace model IDs.
# Adjust max_batch_* and optimal_size based on your VRAM measurements.
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": {
        "optimal_size": (512, 512),
        "pipeline":"stablediff",
        "max_batch_fp32": 1,          # safe on 6 GB with attention slicing
        "max_batch_fp16": 2,          # may work, but Maxwell FP16 is limited
        "recommended_dtype": "fp32",  # safest / most compatible on Maxwell
        "notes": "Classic SD 1.5 – 6 GB VRAM budget is tight",
    },
    "Kernel/sd-nsfw": {
        "optimal_size": (512, 512),
        "pipeline":"stablediff",
        "max_batch_fp32": 1,          # safe on 6 GB with attention slicing
        "max_batch_fp16": 2,          # may work, but Maxwell FP16 is limited
        "recommended_dtype": "fp32",  # safest / most compatible on Maxwell
        "notes": "Classic SD 1.5 – 6 GB VRAM budget is tight",
    },
    "UnfilteredAI/NSFW-gen-v2.1": {
        "optimal_size": (512, 512),
        "pipeline":"diff",
        "max_batch_fp32": 1,          # safe on 6 GB with attention slicing
        "max_batch_fp16": 2,          # may work, but Maxwell FP16 is limited
        "recommended_dtype": "fp32",  # safest / most compatible on Maxwell
        "notes": "Classic SD 1.5 – 6 GB VRAM budget is tight",
    },
    "Heartsync/NSFW-Uncensored": {
        "optimal_size": (768, 768),
        "pipeline":"diff",
        "max_batch_fp32": 1,          # safe on 6 GB with attention slicing
        "max_batch_fp16": 2,          # may work, but Maxwell FP16 is limited
        "recommended_dtype": "fp16",  # safest / most compatible on Maxwell
        "notes": "Classic SD 1.5 – 6 GB VRAM budget is tight",
    },
    "runwayml/stable-diffusion-v1-5": {  # alias / older name
        "optimal_size": (512, 512),
        "max_batch_fp32": 1,
        "max_batch_fp16": 2,
        "recommended_dtype": "fp32",
        "notes": "Same weights as stable-diffusion-v1-5/…",
    },
    # Add more models here as you test them, e.g.:
    # "stabilityai/stable-diffusion-2-1-base": { ... },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    """Parsed from a single <experiment> element."""
    index: int
    model: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    dtype: str = "fp32"                 # "fp16" | "fp32" | "bf16"
    num_images: int = 1
    batch_size: int = 1
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None          # None = random each batch
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dtype_from_string(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32", "full"):
        return torch.float32
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported scalar type: {s!r} (use fp16 / fp32 / bf16)")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def warn(msg: str) -> None:
    print(f"  WARNING: {msg}", file=sys.stderr)


def check_against_model_config(spec: ExperimentSpec) -> None:
    """Emit warnings when the experiment diverges from known-good settings."""
    cfg = MODEL_CONFIGS.get(spec.model)
    if cfg is None:
        warn(f"Model {spec.model!r} has no entry in MODEL_CONFIGS – "
             "using experiment values as-is (no size / batch / dtype guidance).")
        return

    opt_w, opt_h = cfg["optimal_size"]
    if (spec.width, spec.height) != (opt_w, opt_h):
        warn(f"Experiment size {spec.width}×{spec.height} differs from "
             f"optimal {opt_w}×{opt_h} for {spec.model}")

    rec_dtype = cfg["recommended_dtype"]
    if spec.dtype.lower() != rec_dtype:
        warn(f"Requested dtype={spec.dtype} but recommended for this model "
             f"on current hardware is {rec_dtype}")

    key = f"max_batch_{spec.dtype.lower()}"
    max_b = cfg.get(key)
    if max_b is not None and spec.batch_size > max_b:
        warn(f"batch_size={spec.batch_size} exceeds configured max {max_b} "
             f"for dtype={spec.dtype} on this model – may OOM")


def parse_experiments_xml(path: Path) -> List[ExperimentSpec]:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag not in ("run", "experiments"):
        raise ValueError("Root element must be <run> or <experiments>")

    specs: List[ExperimentSpec] = []
    for i, el in enumerate(root.findall("experiment"), start=1):
        def txt(tag: str, default: str = "") -> str:
            child = el.find(tag)
            return (child.text or "").strip() if child is not None else default

        def num(tag: str, default: float | int) -> float | int:
            child = el.find(tag)
            if child is None or not (child.text or "").strip():
                return default
            val = (child.text or "").strip()
            return type(default)(val)

        model = txt("model")
        if not model:
            raise ValueError(f"<experiment> #{i} missing required <model>")
        prompt = txt("prompt")
        if not prompt:
            raise ValueError(f"<experiment> #{i} missing required <prompt>")

        seed_raw = txt("seed")
        seed = int(seed_raw) if seed_raw else None

        # Collect any unknown tags into extra
        known = {
            "model", "prompt", "negative_prompt", "width", "height",
            "dtype", "num_images", "batch_size", "num_inference_steps",
            "guidance_scale", "seed",
        }
        extra = {
            child.tag: (child.text or "").strip()
            for child in el
            if child.tag not in known
        }

        specs.append(ExperimentSpec(
            index=i,
            model=model,
            prompt=prompt,
            negative_prompt=txt("negative_prompt"),
            width=int(num("width", 512)),
            height=int(num("height", 512)),
            dtype=txt("dtype", "fp32") or "fp32",
            num_images=int(num("num_images", 1)),
            batch_size=int(num("batch_size", 1)),
            num_inference_steps=int(num("num_inference_steps", 20)),
            guidance_scale=float(num("guidance_scale", 7.5)),
            seed=seed,
            extra=extra,
        ))
    return specs


def create_pipeline(model_id: str, dtype: torch.dtype, device: torch.device):
    print(f"  Loading pipeline {model_id}  dtype={dtype} …")
    t0 = time.perf_counter()
    cfg = MODEL_CONFIGS.get(model_id)
    if cfg is None:
        warn(f"ERR:Model {model_id} has no entry in MODEL_CONFIGS – \n")
        return

    pipeline = cfg["pipeline"]
    warn(f"Requested pipeline {pipeline}")
             
    if (pipeline == "stablediff"):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    if (pipeline == "diff"):
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    #pipe = pipe.to(device)
    # Always enable attention slicing – cheap win on 6 GB cards
    load_s = time.perf_counter() - t0
    print(f"  Pipeline ready in {load_s:.1f}s")
    return pipe, load_s


def write_meta(path: Path, spec: ExperimentSpec, run_info: Dict[str, Any]) -> None:
    meta = {
        "experiment_index": spec.index,
        "model": spec.model,
        "prompt": spec.prompt,
        "negative_prompt": spec.negative_prompt,
        "width": spec.width,
        "height": spec.height,
        "dtype": spec.dtype,
        "num_images": spec.num_images,
        "batch_size": spec.batch_size,
        "num_inference_steps": spec.num_inference_steps,
        "guidance_scale": spec.guidance_scale,
        "seed": spec.seed,
        "extra": spec.extra,
        "run": run_info,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_perf(path: Path, lines: List[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    spec: ExperimentSpec,
    exp_dir: Path,
    device: torch.device,
    run_info: Dict[str, Any],
) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    images_dir = exp_dir / "images"
    images_dir.mkdir(exist_ok=True)

    meta_path = exp_dir / "meta.json"
    perf_path = exp_dir / "performance.log"

    # Header of performance log
    append_perf(perf_path, [
        f"# Performance log – experiment {spec.index:03d}",
        f"# model={spec.model}",
        f"# size={spec.width}x{spec.height}  dtype={spec.dtype}",
        f"# num_images={spec.num_images}  batch_size={spec.batch_size}",
        f"# steps={spec.num_inference_steps}  guidance={spec.guidance_scale}",
        f"# started={datetime.now().isoformat(timespec='seconds')}",
        "",
    ])

    check_against_model_config(spec)

    dtype = dtype_from_string(spec.dtype)
    pipe, load_time = create_pipeline(spec.model, dtype, device)

    append_perf(perf_path, [
        f"pipeline_load_seconds={load_time:.3f}",
        f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        "",
    ])

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    num_batches = math.ceil(spec.num_images / spec.batch_size)
    total_images = 0
    batch_times: List[float] = []

    print(f"  Generating {spec.num_images} image(s) in {num_batches} batch(es) "
          f"(batch_size={spec.batch_size}) …")

    for b in range(num_batches):
        remaining = spec.num_images - total_images
        this_batch = min(spec.batch_size, remaining)

        # Seed handling
        if spec.seed is not None:
            # deterministic: seed + batch index
            generator = torch.Generator(device=device).manual_seed(spec.seed + b)
        else:
            generator = None

        # For a single prompt we can request multiple images in one call
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = pipe(
                prompt=[spec.prompt] * this_batch,
                negative_prompt=[spec.negative_prompt] * this_batch if spec.negative_prompt else None,
                num_inference_steps=spec.num_inference_steps,
                guidance_scale=spec.guidance_scale,
                height=spec.height,
                width=spec.width,
                generator=generator,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_s = time.perf_counter() - t0
        batch_times.append(batch_s)

        for img in result.images:
            out_name = f"{total_images:04d}.png"
            img.save(images_dir / out_name)
            total_images += 1

        peak_mb = 0.0
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        append_perf(perf_path, [
            f"batch={b:03d}  images={this_batch}  "
            f"seconds={batch_s:.3f}  "
            f"sec_per_image={batch_s / this_batch:.3f}  "
            f"peak_vram_mb={peak_mb:.1f}",
        ])
        print(f"    batch {b+1}/{num_batches}: {this_batch} img in {batch_s:.1f}s "
              f"({batch_s/this_batch:.2f}s/img)  peak VRAM {peak_mb:.0f} MB")

    total_gen_s = sum(batch_times)
    append_perf(perf_path, [
        "",
        f"total_generation_seconds={total_gen_s:.3f}",
        f"images_generated={total_images}",
        f"avg_sec_per_image={total_gen_s / max(total_images, 1):.3f}",
        f"finished={datetime.now().isoformat(timespec='seconds')}",
    ])

    # Final meta (include some perf summary)
    run_info = dict(run_info)
    run_info.update({
        "pipeline_load_seconds": round(load_time, 3),
        "total_generation_seconds": round(total_gen_s, 3),
        "images_generated": total_images,
        "avg_sec_per_image": round(total_gen_s / max(total_images, 1), 3),
    })
    write_meta(meta_path, spec, run_info)

    # Free memory for next experiment
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  → results in {exp_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a set of Stable Diffusion experiments described in XML"
    )
    parser.add_argument(
        "xml",
        type=Path,
        help="Path to the experiments XML file (see experiments.example.xml)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory for all run folders (default: ./runs)",
    )
    args = parser.parse_args()

    if not args.xml.is_file():
        print(f"ERROR: XML file not found: {args.xml}", file=sys.stderr)
        return 1

    print("=== Stable Diffusion Experiment Runner ===")
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU            : {props.name}")
    print(f"CC / VRAM      : {props.major}.{props.minor}  /  "
          f"{props.total_memory / 1024**3:.1f} GB")
    print()

    specs = parse_experiments_xml(args.xml)
    if not specs:
        print("No <experiment> elements found in XML.")
        return 1
    print(f"Loaded {len(specs)} experiment(s) from {args.xml}")

    # Create run folder
    run_ts = now_ts()
    run_dir = args.runs_dir / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory  : {run_dir.resolve()}\n")

    run_info = {
        "run_timestamp": run_ts,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "pytorch_version": torch.__version__,
        "xml_source": str(args.xml),
    }

    for spec in specs:
        # Subfolder: 001_YYYYMMDD_HHMMSS  (optionally append a short slug)
        #slug = "".join(c if c.isalnum() else "_" for c in spec.prompt[:40]).strip("_")
        slug = spec.model[:40].replace("/", "_")
        exp_ts = now_ts()
        exp_name = f"{spec.index:03d}_{exp_ts}_{slug}"
        exp_dir = run_dir / exp_name

        print(f"── Experiment {spec.index:03d}  {spec.model}")
        print(f"   prompt: {spec.prompt[:70]}{'…' if len(spec.prompt) > 70 else ''}")

        # 1. run Python-Garbage 
        gc.collect()

        # 2. clean PyTorch-VRAM-Cache completely
        torch.cuda.empty_cache()
        try:
            run_experiment(spec, exp_dir, device, run_info)
        except Exception as exc:
            print(f"  ERROR in experiment {spec.index}: {exc}", file=sys.stderr)
            # still write a minimal failure meta
            (exp_dir / "ERROR.txt").write_text(str(exc), encoding="utf-8")
            continue

        print()

    print(f"All done. Results under: {run_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
