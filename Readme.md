# Quantization-Aware Training for Diffusion Models

## Project Overview

This repository contains the environment setup and initial experiments for applying **Quantization-Aware Training (QAT)** to diffusion models, specifically targeting efficient deployment on resource-constrained devices. The primary focus is on **Stable Diffusion v1.5** with GPU acceleration and low-bit quantization support.

---

## Environment Setup

We use **Miniconda** to create an isolated environment with **Python 3.10** to ensure reproducibility across systems.

### 1. Miniconda Installation

- **Platform**: Windows
- **Conda Version**: 25.7.0 (Note: Update to 25.9.1 recommended for the latest features and security patches)
- Download and install Miniconda from [miniconda.com](https://docs.conda.io/en/latest/miniconda.html) if not already installed.

### 2. Conda Environment Creation

Run the following commands in your terminal to set up the environment:

```powershell
# Accept terms of service for Conda channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

# Create and activate the environment
conda create -n diffusion python=3.10 -y
conda activate diffusion
```

### 3. Installed Library Versions

The table below lists all libraries installed in the `diffusion` environment, their versions, and installation details. These versions are verified for compatibility with **Python 3.10** and **CUDA 12.1** on **Windows 11**.

| **Library**          | **Version**      | **Installation Source / Description**                              |
|----------------------|------------------|-------------------------------------------------------------------|
| **torch**            | 2.5.1 (CUDA 12.1)| `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y` |
| **torchvision**      | 0.20.1           | Installed via PyTorch                                              |
| **torchaudio**       | 2.5.1            | Installed via PyTorch                                              |
| **numpy**            | 1.26.4           | Downgraded for compatibility                                      |
| **diffusers**        | 0.30.3           | Stable Diffusion base                                             |
| **transformers**     | 4.46.0           | For text encoder support                                          |
| **accelerate**       | 0.34.2           | For efficient hardware utilization                                |
| **safetensors**      | 0.4.4            | Safe and fast tensor loading                                      |
| **bitsandbytes**     | 0.43.3           | 8-bit inference optimization                                      |
| **optimum**          | 1.21.4           | Model optimization toolkit                                        |
| **huggingface_hub**  | 0.25.2           | Access to pretrained models                                       |
| **Pillow**           | 10.3.0           | Image saving and manipulation                                     |
| **matplotlib**       | 3.9.2            | For visualizations                                               |
| **scikit-learn**     | 1.5.1            | General ML utilities                                              |
| **tqdm**             | 4.66.5           | Progress bar utility                                              |
| **torchinfo**        | 1.8.0            | Model architecture summary                                        |
| **xformers** *(opt)* | 0.0.26.post1     | Memory-efficient attention (optional)                             |

---

### ⚠️ Important Notes

- **NumPy Version**: Do **not** upgrade NumPy to 2.x, as it may cause compatibility issues with the current setup.
- **PyTorch CUDA Build**: Ensure the PyTorch CUDA build matches your GPU driver version to avoid runtime errors.
- **Optional Dependencies**: The `xformers` library is optional and can be installed for memory-efficient attention mechanisms if needed.

---

## Next Steps

1. Clone this repository:  
   ```bash
   git clone https://github.com/Arham-Mod/Diffusia.git
   ```
2. Follow the environment setup instructions above.
3. Explore the codebase for quantization-aware training scripts and experiment configurations.

For issues or contributions, please open a GitHub issue or submit a pull request.