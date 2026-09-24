# Can We Use Saliency Maps for the Evaluation of Generative Image Models?

This repository contains the official implementation of the evaluation pipeline proposed in "Can We Use Saliency Maps for the Evaluation of Generative Image Models?" for assessing deep generative image models using saliency maps.

---

# Installation & Usage

## Installation

First, create a Python environment and install the required dependencies:

```bash
conda create --name sas_env python=3.10 pip
conda activate sas_env
```
Install PyTorch appropriate for your machine and CUDA version. For a CPU-only installation:
```bash
pip install torch torchvision
```
Then install the package and its dependencies:
```bash
pip install .
```

## Usage
Compute the SAS score for one generated-image model:
```python
from sas_eval import compute_sas

score = compute_sas(
    real_dir="data/real",
    fake_dir="data/generated/model_a",
    batch_size=32,
    seed=42,
)
```
