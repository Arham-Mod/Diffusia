# sd_pipeline.py
"""
Stable Diffusion baseline pipeline (memory-aware) — Corrected Version
--------------------------------------------------------------------
Features:
- Loads the model in FP16 (half precision) for lower VRAM use.
- Uses low_cpu_mem_usage to minimize CPU memory during load.
- Enables attention slicing and optionally xFormers for efficient attention.
- Generates one image from a text prompt and saves it.
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# CONFIGURATION

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUT_FILE = "generated_image.png"
PROMPT = "Harshit, high detail"
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 7.5
SEED = 22

# Memory / performance toggles
USE_XFORMERS = False
ENABLE_SLICING = True
LOW_CPU_MEM = True

# DEVICE SETUP

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"✅ Device: {device} | Torch version: {torch.__version__}")


# 1) LOAD PIPELINE

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    safety_checker=None,              feature_extractor=None,           low_cpu_mem_usage=LOW_CPU_MEM,
)

# Switch scheduler to a fast and stable one
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 2) MEMORY OPTIMIZATIONS

if device == "cuda":
    pipe.to(device)

    if ENABLE_SLICING:
        pipe.enable_attention_slicing()
        print("✅ Attention slicing enabled (lower VRAM use).")

    if USE_XFORMERS:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✅ xFormers enabled for memory-efficient attention.")
        except Exception as e:
            print("⚠️ Could not enable xFormers:", e)
else:
    pipe.to("cpu")


# 3) SEED CONTROL

generator = torch.Generator(device=device).manual_seed(SEED)


# 4) INFERENCE

print("🚀 Generating image... (this may take a minute)")

with torch.autocast(device_type=device, dtype=dtype):
    result = pipe(
        PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    )

image = result.images[0]
image.save(OUT_FILE)

print(f"✅ Image generation complete! Saved to '{OUT_FILE}'")
