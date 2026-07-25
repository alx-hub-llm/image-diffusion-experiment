#!/usr/bin/env python3
"""
smoke_test.py – minimal Stable Diffusion 1.5 text-to-image generation
Target: GTX 980 Ti (6 GB, Maxwell sm_52) on Fedora

Uses float32 + attention slicing for maximum compatibility and to stay
inside the 6 GB VRAM budget.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def main() -> int:
    print("=== Stable Diffusion 1.5 smoke test ===")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Check that the NVIDIA driver is loaded (nvidia-smi).")
        return 1

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU             : {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Total VRAM      : {props.total_memory / 1024**3:.1f} GB")
    print()

    # ------------------------------------------------------------------
    # Model – public SD 1.5 checkpoint
    # ------------------------------------------------------------------
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"Loading pipeline: {model_id}")
    print("(first run will download ~4 GB of weights …)")

    # float32 is the safest choice on Maxwell; float16 can trigger
    # unsupported kernels or precision issues on older architectures.
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,          # disable for speed / simplicity
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Memory optimisations for 6 GB cards
    pipe.enable_attention_slicing()
    # Optional further savings (slower):
    # pipe.enable_sequential_cpu_offload()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    prompt = "a photo of an astronaut riding a horse on mars, highly detailed, 4k"
    negative_prompt = "blurry, low quality, deformed"

    print(f"\nPrompt          : {prompt}")
    print("Generating (20 steps, 512×512) …")

    generator = torch.Generator(device=device).manual_seed(42)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=generator,
        )

    image: Image.Image = result.images[0]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "smoke_test.png"
    image.save(out_path)

    print(f"\n✓ Image saved to: {out_path.resolve()}")
    print("Smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
