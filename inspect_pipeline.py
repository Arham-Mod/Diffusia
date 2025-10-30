"""
Inspect Stable Diffusion Pipeline Components
--------------------------------------------
This script loads the Stable Diffusion pipeline and prints its
internal architecture — including the UNet, VAE, and text encoder.
"""

import torch
from diffusers import StableDiffusionPipeline

# CONFIGURATION

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LOW_CPU_MEM = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# LOAD PIPELINE
print("🔹 Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
    feature_extractor=None,
    low_cpu_mem_usage=LOW_CPU_MEM,
)

pipe.to(DEVICE)

# PRINT PIPELINE STRUCTURE

print("\n============================")
print("PIPELINE STRUCTURE OVERVIEW")
print("============================\n")
print(pipe)

# Optionally save to a file for later inspection
with open("pipeline_structure.txt", "w", encoding="utf-8") as f:
    f.write(str(pipe))


# PRINT SUBMODULE STRUCTURES

print("\n============================")
print("UNET STRUCTURE")
print("============================\n")
print(pipe.unet)

with open("unet_structure.txt", "w", encoding="utf-8") as f:
    f.write(str(pipe.unet))

print("\n============================")
print("TEXT ENCODER STRUCTURE")
print("============================\n")
print(pipe.text_encoder)

with open("text_encoder_structure.txt", "w", encoding="utf-8") as f:
    f.write(str(pipe.text_encoder))

print("\n============================")
print("VAE STRUCTURE")
print("============================\n")
print(pipe.vae)

with open("vae_structure.txt", "w", encoding="utf-8") as f:
    f.write(str(pipe.vae))

print("\n✅ All structures printed and saved as text files!")
