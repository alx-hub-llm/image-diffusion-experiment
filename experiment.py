#!/usr/bin/env python3
"""
experiment.py – Batch experiment runner for Stable Diffusion

Queue mode (default) or single-file mode.
Run folders named after the XML stem.
Images + meta + performance live in the same experiment folder.
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import gc
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": {
        "optimal_size": (512, 512), "pipeline": "stablediff",
        "max_batch_fp32": 1, "max_batch_fp16": 2, "recommended_dtype": "fp32",
        "notes": "Classic SD 1.5 – 6 GB VRAM budget is tight",
    },
    "Kernel/sd-nsfw": {
        "optimal_size": (512, 512), "pipeline": "stablediff",
        "max_batch_fp32": 1, "max_batch_fp16": 2, "recommended_dtype": "fp32",
        "notes": "Classic SD 1.5 – 6 GB VRAM budget is tight",
    },
    "UnfilteredAI/NSFW-gen-v2.1": {
        "optimal_size": (512, 512), "pipeline": "diff",
        "max_batch_fp32": 1, "max_batch_fp16": 2, "recommended_dtype": "fp32",
        "notes": "Classic SD 1.5 – 6 GB VRAM budget is tight",
    },
    "Heartsync/NSFW-Uncensored": {
        "optimal_size": (768, 768), "pipeline": "diff",
        "max_batch_fp32": 1, "max_batch_fp16": 2, "recommended_dtype": "fp16",
        "notes": "Classic SD 1.5 – 6 GB VRAM budget is tight",
    },
    "runwayml/stable-diffusion-v1-5": {
        "optimal_size": (512, 512),
        "max_batch_fp32": 1, "max_batch_fp16": 2, "recommended_dtype": "fp32",
        "notes": "Same weights as stable-diffusion-v1-5/…",
    },
}


@dataclass
class ExperimentSpec:
    index: int
    model: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    dtype: str = "fp32"
    num_images: int = 1
    batch_size: int = 1
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def dtype_from_string(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("fp16", "float16", "half"): return torch.float16
    if s in ("fp32", "float32", "full"): return torch.float32
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    raise ValueError(f"Unsupported scalar type: {s!r}")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def warn(msg: str) -> None:
    print(f"  WARNING: {msg}", file=sys.stderr)


def check_against_model_config(spec: ExperimentSpec) -> None:
    cfg = MODEL_CONFIGS.get(spec.model)
    if cfg is None:
        warn(f"Model {spec.model!r} has no entry in MODEL_CONFIGS")
        return
    opt_w, opt_h = cfg["optimal_size"]
    if (spec.width, spec.height) != (opt_w, opt_h):
        warn(f"size {spec.width}×{spec.height} differs from optimal {opt_w}×{opt_h}")
    if spec.dtype.lower() != cfg["recommended_dtype"]:
        warn(f"dtype={spec.dtype} but recommended is {cfg['recommended_dtype']}")
    key = f"max_batch_{spec.dtype.lower()}"
    max_b = cfg.get(key)
    if max_b is not None and spec.batch_size > max_b:
        warn(f"batch_size={spec.batch_size} exceeds max {max_b} – may OOM")


def parse_experiments_xml(path: Path) -> List[ExperimentSpec]:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag not in ("run", "experiments"):
        raise ValueError("Root element must be <run> or <experiments>")
    specs = []
    for i, el in enumerate(root.findall("experiment"), start=1):
        def txt(tag, default=""):
            child = el.find(tag)
            return (child.text or "").strip() if child is not None else default
        def num(tag, default):
            child = el.find(tag)
            if child is None or not (child.text or "").strip():
                return default
            return type(default)((child.text or "").strip())
        model = txt("model")
        prompt = txt("prompt")
        if not model or not prompt:
            raise ValueError(f"<experiment> #{i} missing model or prompt")
        seed_raw = txt("seed")
        seed = int(seed_raw) if seed_raw else None
        known = {"model","prompt","negative_prompt","width","height","dtype",
                 "num_images","batch_size","num_inference_steps","guidance_scale","seed"}
        extra = {c.tag: (c.text or "").strip() for c in el if c.tag not in known}
        specs.append(ExperimentSpec(
            index=i, model=model, prompt=prompt,
            negative_prompt=txt("negative_prompt"),
            width=int(num("width", 512)), height=int(num("height", 512)),
            dtype=txt("dtype", "fp32") or "fp32",
            num_images=int(num("num_images", 1)), batch_size=int(num("batch_size", 1)),
            num_inference_steps=int(num("num_inference_steps", 20)),
            guidance_scale=float(num("guidance_scale", 7.5)),
            seed=seed, extra=extra,
        ))
    return specs


def create_pipeline(model_id: str, dtype: torch.dtype, device: torch.device):
    print(f"  Loading pipeline {model_id}  dtype={dtype} …")
    t0 = time.perf_counter()
    cfg = MODEL_CONFIGS.get(model_id)
    if cfg is None:
        warn(f"ERR: Model {model_id} missing from MODEL_CONFIGS")
        return None, 0.0
    pipeline = cfg.get("pipeline", "stablediff")
    if pipeline == "stablediff":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    load_s = time.perf_counter() - t0
    print(f"  Pipeline ready in {load_s:.1f}s")
    return pipe, load_s


def write_meta(path: Path, spec: ExperimentSpec, run_info: Dict[str, Any]) -> None:
    meta = {
        "experiment_index": spec.index, "model": spec.model,
        "prompt": spec.prompt, "negative_prompt": spec.negative_prompt,
        "width": spec.width, "height": spec.height, "dtype": spec.dtype,
        "num_images": spec.num_images, "batch_size": spec.batch_size,
        "num_inference_steps": spec.num_inference_steps,
        "guidance_scale": spec.guidance_scale, "seed": spec.seed,
        "extra": spec.extra, "run": run_info,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_perf(path: Path, lines: List[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def run_experiment(spec: ExperimentSpec, exp_dir: Path, device: torch.device, run_info: Dict[str, Any]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    meta_path = exp_dir / "meta.json"
    perf_path = exp_dir / "performance.log"

    append_perf(perf_path, [
        f"# Performance log – experiment {spec.index:03d}",
        f"# model={spec.model}",
        f"# size={spec.width}x{spec.height}  dtype={spec.dtype}",
        f"# num_images={spec.num_images}  batch_size={spec.batch_size}",
        f"# steps={spec.num_inference_steps}  guidance={spec.guidance_scale}",
        f"# started={datetime.now().isoformat(timespec='seconds')}", "",
    ])

    check_against_model_config(spec)
    dtype = dtype_from_string(spec.dtype)
    pipe, load_time = create_pipeline(spec.model, dtype, device)
    if pipe is None:
        raise RuntimeError(f"Failed to create pipeline for {spec.model}")

    append_perf(perf_path, [
        f"pipeline_load_seconds={load_time:.3f}",
        f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", "",
    ])

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
        generator = (torch.Generator(device=device).manual_seed(spec.seed + b)
                     if spec.seed is not None else None)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = pipe(
                prompt=[spec.prompt] * this_batch,
                negative_prompt=[spec.negative_prompt] * this_batch if spec.negative_prompt else None,
                num_inference_steps=spec.num_inference_steps,
                guidance_scale=spec.guidance_scale,
                height=spec.height, width=spec.width, generator=generator,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_s = time.perf_counter() - t0
        batch_times.append(batch_s)

        for img in result.images:
            img.save(exp_dir / f"{total_images:04d}.png")
            total_images += 1

        peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if torch.cuda.is_available() else 0.0)
        append_perf(perf_path, [
            f"batch={b:03d}  images={this_batch}  seconds={batch_s:.3f}  "
            f"sec_per_image={batch_s / this_batch:.3f}  peak_vram_mb={peak_mb:.1f}",
        ])
        print(f"    batch {b+1}/{num_batches}: {this_batch} img in {batch_s:.1f}s "
              f"({batch_s/this_batch:.2f}s/img)  peak VRAM {peak_mb:.0f} MB")

    total_gen_s = sum(batch_times)
    append_perf(perf_path, [
        "", f"total_generation_seconds={total_gen_s:.3f}",
        f"images_generated={total_images}",
        f"avg_sec_per_image={total_gen_s / max(total_images, 1):.3f}",
        f"finished={datetime.now().isoformat(timespec='seconds')}",
    ])

    run_info = dict(run_info)
    run_info.update({
        "pipeline_load_seconds": round(load_time, 3),
        "total_generation_seconds": round(total_gen_s, 3),
        "images_generated": total_images,
        "avg_sec_per_image": round(total_gen_s / max(total_images, 1), 3),
    })
    write_meta(meta_path, spec, run_info)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  → results in {exp_dir}")


def process_xml(xml_path: Path, runs_dir: Path, device: torch.device, props) -> Path:
    specs = parse_experiments_xml(xml_path)
    if not specs:
        print(f"No <experiment> elements found in {xml_path}")
        return Path()

    print(f"Loaded {len(specs)} experiment(s) from {xml_path.name}")
    run_ts = now_ts()
    run_name = f"{xml_path.stem}_{run_ts}"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory  : {run_dir.resolve()}\n")

    run_info = {
        "run_timestamp": run_ts, "run_name": run_name,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "pytorch_version": torch.__version__,
        "xml_source": str(xml_path), "xml_stem": xml_path.stem,
    }

    for spec in specs:
        slug = spec.model[:40].replace("/", "_")
        exp_name = f"{spec.index:03d}_{now_ts()}_{slug}"
        exp_dir = run_dir / exp_name
        print(f"── Experiment {spec.index:03d}  {spec.model}")
        print(f"   prompt: {spec.prompt[:70]}{'…' if len(spec.prompt) > 70 else ''}")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            run_experiment(spec, exp_dir, device, run_info)
        except Exception as exc:
            print(f"  ERROR in experiment {spec.index}: {exc}", file=sys.stderr)
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "ERROR.txt").write_text(str(exc), encoding="utf-8")
            continue
        print()

    # Copy source XML into run folder
    try:
        shutil.copy2(xml_path, run_dir / xml_path.name)
    except Exception as e:
        warn(f"Could not copy XML into run folder: {e}")

    # Write run_summary.json
    summary = {
        "run_name": run_name, "run_timestamp": run_ts,
        "xml_source": str(xml_path), "xml_stem": xml_path.stem,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "pytorch_version": torch.__version__,
        "num_experiments": len(specs), "experiments": [],
    }
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / "meta.json"
        if meta_file.is_file():
            try:
                m = json.loads(meta_file.read_text(encoding="utf-8"))
                summary["experiments"].append({
                    "folder": d.name,
                    "index": m.get("experiment_index"),
                    "model": m.get("model"),
                    "num_images": m.get("num_images"),
                    "images_generated": m.get("run", {}).get("images_generated"),
                    "avg_sec_per_image": m.get("run", {}).get("avg_sec_per_image"),
                    "dtype": m.get("dtype"),
                    "width": m.get("width"), "height": m.get("height"),
                })
            except Exception:
                pass
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"All done for {xml_path.name}. Results under: {run_dir.resolve()}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SD experiments from XML (single file or queue)")
    parser.add_argument("xml", type=Path, nargs="?", default=None,
                        help="Single XML file. Omit to process experiments/pending/")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--pending-dir", type=Path, default=Path("experiments/pending"))
    parser.add_argument("--done-dir", type=Path, default=Path("experiments/done"))
    args = parser.parse_args()

    print("=== Stable Diffusion Experiment Runner ===")
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU            : {props.name}")
    print(f"CC / VRAM      : {props.major}.{props.minor}  /  {props.total_memory / 1024**3:.1f} GB\n")

    if args.xml is not None:
        if not args.xml.is_file():
            print(f"ERROR: XML file not found: {args.xml}", file=sys.stderr)
            return 1
        xml_files = [args.xml]
        move_to_done = False
    else:
        args.pending_dir.mkdir(parents=True, exist_ok=True)
        args.done_dir.mkdir(parents=True, exist_ok=True)
        xml_files = sorted(args.pending_dir.glob("*.xml"))
        if not xml_files:
            print(f"No *.xml files found in {args.pending_dir.resolve()}")
            print("Drop experiment XML files into that folder and re-run.")
            return 0
        move_to_done = True
        print(f"Queue mode: found {len(xml_files)} file(s) in {args.pending_dir}")

    for xml_path in xml_files:
        print(f"\n{'='*60}\nProcessing: {xml_path}\n{'='*60}\n")
        try:
            process_xml(xml_path, args.runs_dir, device, props)
            if move_to_done:
                dest = args.done_dir / xml_path.name
                if dest.exists():
                    dest = args.done_dir / f"{xml_path.stem}_{now_ts()}{xml_path.suffix}"
                shutil.move(str(xml_path), str(dest))
                print(f"Moved {xml_path.name} → {dest}")
        except Exception as exc:
            print(f"FAILED to process {xml_path}: {exc}", file=sys.stderr)
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
