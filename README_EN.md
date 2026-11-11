# Lung Lesion V75 — Local Web Inference

Vietnamese version: [README.md](README.md)

This local application accepts a swine lung image, segments the lung lobes and
lesions, and calculates:

```text
lesion ratio (%) = lesion pixels inside the lung / total lung pixels × 100
```

All source code and model assets required for inference are included in this
directory. You do not need to copy `Dataset`, `Legacy_Project`, or any other
research project.

> This is a research tool, not a clinical diagnostic device.

## 1. New-machine requirements

- 64-bit Linux, or Windows 11 with Ubuntu on WSL2. Linux/WSL2 is recommended.
- Python 3.12.
- An NVIDIA CUDA-capable GPU with a working NVIDIA driver.
- About 11 GB for this project, plus space for Python and PyTorch.
- Internet access while installing Python packages. Model inference does not
  require additional model downloads after setup.

Check the GPU first:

```bash
nvidia-smi
```

Do not continue until this command detects the GPU. Fresh, uncached inference
currently requires CUDA; the pipeline does not fully support CPU-only inference.

## 2. What to copy

Copy the complete directory and preserve its name:

```text
Unsupervised_Lesion_Workflow/
├── artifacts/                 # V40, V64, V67, V74 checkpoints and model cache
├── models/
│   ├── sam2.1_hiera_tiny.pt
│   ├── unmarked_lung_unet_best.pt
│   └── dinov2_with_registers_small/
├── src/                       # Python source
├── vendor/sam2/               # Pinned SAM2 source
├── web_app.py
├── requirements-inference.txt
└── README_EN.md
```

Use a `tar` or `zip` archive that preserves symbolic links. Do not rename
`Unsupervised_Lesion_Workflow` after extraction because some checkpoints contain
paths relative to this directory name.

Do not copy a virtual environment from the original computer. Create a clean
environment on the destination machine as described below.

## 3. Install from scratch

Run the following commands in a Linux/WSL2 terminal from the parent directory
that contains `Unsupervised_Lesion_Workflow`.

### 3.1 Install Python and system packages

On Ubuntu 24.04/WSL2:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip libgl1 libglib2.0-0
```

For another Linux distribution, install Python 3.12, its `venv` module, and the
equivalent OpenGL/GLib runtime packages using that distribution's package
manager.

### 3.2 Create a Python environment on the new machine

```bash
cd Unsupervised_Lesion_Workflow
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3.3 Install CUDA-enabled PyTorch

The verified project environment uses PyTorch `2.5.1+cu121`:

```bash
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

If the destination GPU or driver requires another CUDA build, use the matching
command from the official PyTorch installation page and verify the workflow
again before long-term use.

### 3.4 Install inference dependencies

```bash
python -m pip install -r requirements-inference.txt
```

Do not install `vendor/sam2` separately. The workflow loads the bundled SAM2
source directly.

## 4. Verify the installation

Keep the virtual environment activated and run these commands from
`Unsupervised_Lesion_Workflow`:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("NVIDIA GPU/CUDA was not detected")
print("GPU:", torch.cuda.get_device_name(0))
PY
```

The output must contain `GPU available: True`.

Verify the bundled model assets without running prediction:

```bash
PYTHONPATH=src python - <<'PY'
from unsupervised_lesion.color_region.student import _dino_provenance
from unsupervised_lesion.io import LUNG_UNET_CHECKPOINT, SAM2_CHECKPOINT

_, dino = _dino_provenance()
assert SAM2_CHECKPOINT.is_file()
assert LUNG_UNET_CHECKPOINT.is_file()
assert dino.is_dir()
print("Model assets: OK")
PY
```

## 5. Quick start — run the web application

For every new terminal session:

```bash
cd Unsupervised_Lesion_Workflow
source .venv/bin/activate
PYTHONPATH=src python -m uvicorn web_app:app \
  --host 127.0.0.1 --port 8785
```

Open this address in a browser:

```text
http://127.0.0.1:8785
```

Choose an image, select **Phân tích** (Analyze), and wait for the result. The
page displays:

- the original image;
- lung lobes in green;
- lesions in red;
- the lesion-area-to-lung-area pixel ratio.

The first prediction is slower because several models must be loaded onto the
GPU. Byte-identical images processed earlier are read from
`artifacts/web_v75_cache`.

Stop the server with `Ctrl+C` in the terminal.

### Access from another computer on the same network

Only use this on a trusted local network:

```bash
PYTHONPATH=src python -m uvicorn web_app:app \
  --host 0.0.0.0 --port 8785
```

Open `http://HOST_COMPUTER_IP:8785` from the other computer. The firewall must
allow TCP port 8785. The application currently has no authentication, so do not
expose it directly to the public Internet.

## 6. Input and output

- Input must be an image readable by Pillow, such as PNG or JPEG.
- Each image dimension must be at least 64 pixels.
- The image is converted to RGB and letterboxed to 512×512 during inference.
- The lesion mask is always restricted to the predicted lung mask.
- The ratio is calculated after restoring the output to the source dimensions.
- The source image is not modified.

## 7. Run tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest -q -p no:cacheprovider
```

The latest source-machine result is **53/53 tests passed**. If pytest is not
installed on the destination machine:

```bash
python -m pip install pytest
```

Tests do not replace a real GPU prediction smoke test.

## 8. Troubleshooting

### `GPU available: False`

Check `nvidia-smi`, the NVIDIA driver, WSL2 GPU configuration, and the installed
PyTorch CUDA build. Ensure that you did not install a CPU-only PyTorch build.

### `CUDA out of memory`

Close other GPU-heavy applications and restart the web server. Do not submit
multiple predictions concurrently; the application serializes requests within
one server process.

### `ModuleNotFoundError`

Confirm that the virtual environment is active, run `pip install -r
requirements-inference.txt`, launch from the correct directory, and include
`PYTHONPATH=src` in the command.

### Missing checkpoint or model

Do not copy only `src`. Copy the complete directory, especially `artifacts`,
`models`, and `vendor/sam2`. Preserve the root directory name.

### Port 8785 is already in use

```bash
PYTHONPATH=src python -m uvicorn web_app:app \
  --host 127.0.0.1 --port 8786
```

Then open `http://127.0.0.1:8786`.

## 9. Current validation results

V75 on eight regression-validation images:

| Metric | Lung | Lesion |
| --- | ---: | ---: |
| Dice | 99.64% | 80.54% |
| Precision | 99.91% | 80.53% |
| Recall | 99.37% | 80.55% |
| Specificity | 99.91% | 95.62% |

Lesion TP/FP/FN/TN by pixel:
`104,225 / 25,194 / 25,173 / 550,537`.
Lesion-ratio MAE is 2.94 percentage points.

This cohort contributed to blend and threshold selection. These values are
therefore validation/regression evidence, not an independent-test result or an
accuracy guarantee for every new image.

## 10. Technical note

V75 ensembles four branches: V40, V64, V67, and V74. All production
checkpoints, SAM2, DINOv2, and the lung U-Net are bundled in this directory.
Web inference has no project-file dependency outside
`Unsupervised_Lesion_Workflow`; only historical training and validation commands
require external datasets or references.
