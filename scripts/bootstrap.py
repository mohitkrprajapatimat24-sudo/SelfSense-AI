"""One-command setup for ShelfSense AI.

Usage (from the repo root, with Python 3.11):
    Windows:      py -3.11 scripts/bootstrap.py
    Linux/macOS:  python3.11 scripts/bootstrap.py

Options:
    --cpu          force the CPU build of torch
    --skip-models  install libraries only
    --skip-slm     skip llama.cpp and the Qwen model
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
MODELS = ROOT / "models"
TOOLS = ROOT / "tools"

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
DINO_REPO = "facebook/dinov2-small"
SLM_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
SLM_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
LLAMA_CPP_TAG = "b11720"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-{asset}"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"   # runs on NVIDIA drivers >= 525
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def has_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0


def create_venv():
    if venv_python().exists():
        print(f"venv exists: {VENV}")
    else:
        run([sys.executable, "-m", "venv", VENV])
    run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"])


def install_libraries(force_cpu: bool):
    py = venv_python()
    index = TORCH_CPU_INDEX if force_cpu or not has_nvidia_gpu() else TORCH_CUDA_INDEX
    print(f"Installing torch from {index}")
    run([py, "-m", "pip", "install", "torch==2.14.1", "torchvision==0.29.1", "--index-url", index])
    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])


def download_models():
    MODELS.mkdir(exist_ok=True)
    py = venv_python()

    yolo = MODELS / "yolo11n.pt"
    if not yolo.exists():
        print(f"Downloading {yolo.name}")
        urllib.request.urlretrieve(YOLO_URL, yolo)

    dino = MODELS / "dinov2-small"
    if not dino.exists():
        run([py, "-c",
             "from huggingface_hub import snapshot_download; "
             f"snapshot_download('{DINO_REPO}', local_dir=r'{dino}')"])

    # RapidOCR fetches its ONNX weights on first construction into its package folder.
    run([py, "-c", "from rapidocr import RapidOCR; RapidOCR()"])

    # rembg background-removal model, used to cut out product photos
    rembg_home = MODELS / "rembg"
    if not (rembg_home / "u2net.onnx").exists():
        run([py, "-c", "from rembg import new_session; new_session('u2net')"],
            env={**os.environ, "U2NET_HOME": str(rembg_home)})


def llama_cpp_asset() -> str:
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Windows":
        return "win-cpu-arm64.zip" if arm else "win-cpu-x64.zip"
    if system == "Darwin":
        return "macos-arm64.tar.gz" if arm else "macos-x64.tar.gz"
    return "ubuntu-arm64.tar.gz" if arm else "ubuntu-x64.tar.gz"


def install_llama_cpp():
    dest = TOOLS / "llama.cpp"
    if dest.exists():
        print(f"llama.cpp exists: {dest}")
        return
    TOOLS.mkdir(exist_ok=True)
    asset = llama_cpp_asset()
    archive = TOOLS / asset
    url = LLAMA_CPP_URL.format(tag=LLAMA_CPP_TAG, asset=asset)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)
    archive.unlink()


def download_slm():
    MODELS.mkdir(exist_ok=True)
    if (MODELS / SLM_FILE).exists():
        print(f"{SLM_FILE} exists")
        return
    run([venv_python(), "-c",
         "from huggingface_hub import hf_hub_download; "
         f"hf_hub_download('{SLM_REPO}', '{SLM_FILE}', local_dir=r'{MODELS}')"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-slm", action="store_true")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 11):
        sys.exit(f"Run this with Python 3.11 (found {sys.version.split()[0]}).")

    create_venv()
    install_libraries(args.cpu)
    if not args.skip_models:
        download_models()
    if not args.skip_slm:
        install_llama_cpp()
        download_slm()
    print("\nSetup complete. Activate with:",
          r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate")


if __name__ == "__main__":
    main()
