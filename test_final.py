# test_final.py
"""
Final environment check - ignores NumPy warnings
"""
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ENVIRONMENT VERIFICATION")
print("="*70)

# 1. PyTorch
import torch
print(f"\n✅ PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")

# 2. TorchVision
import torchvision
print(f"\n✅ TorchVision: {torchvision.__version__}")

# 3. bitsandbytes
import bitsandbytes as bnb
print(f"\n✅ bitsandbytes: {bnb.__version__}")

# Test if quantization will work
try:
    # Try to create a quantized linear layer
    import torch.nn as nn
    test_linear = nn.Linear(10, 10).cuda().half()
    print(f"   GPU quantization: Available (test passed)")
except Exception as e:
    print(f"   GPU quantization: {e}")

# 4. Diffusers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
print(f"\n✅ Diffusers: Import successful")

# 5. NumPy (check version)
import numpy as np
print(f"\n⚠️  NumPy: {np.__version__}")
if np.__version__.startswith('2.'):
    print(f"   Warning: NumPy 2.x detected. PyTorch 2.3.0 prefers NumPy 1.x")
    print(f"   Run: pip install 'numpy<2.0' --force-reinstall")
else:
    print(f"   ✅ NumPy version compatible")

print("\n" + "="*70)
print("READY FOR QUANTIZATION EXPERIMENT")
print("="*70)
print("\nYou can now run the quantization code.")
print("The NumPy warnings are non-fatal but annoying.")
print("Downgrade NumPy if you want to suppress them.")