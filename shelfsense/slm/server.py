"""Running Qwen2.5-3B locally with the llama.cpp server (CPU, offline)."""

import os
import platform
import subprocess
import time
from pathlib import Path

import requests

from ..paths import MODELS, ROOT

MODEL = MODELS / "qwen2.5-3b-instruct-q4_k_m.gguf"
PORT = int(os.environ.get("SHELFSENSE_LLM_PORT", "8081"))
URL = os.environ.get("SHELFSENSE_LLM_URL", f"http://127.0.0.1:{PORT}")
CONTEXT = 2048


def find_binary() -> Path | None:
    name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    for p in [Path(os.environ.get("LLAMA_SERVER", "")), *sorted((ROOT / "tools" / "llama.cpp").rglob(name)),
              Path("/usr/local/bin") / name]:
        if p.is_file():
            return p
    return None


def is_up(url: str = URL, timeout: float = 1.0) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


class LlamaServer:
    """Starts llama-server once and keeps it running; reuses one that is already up."""

    def __init__(self, threads: int = 4, url: str = URL):
        self.threads = threads
        self.url = url
        self.proc: subprocess.Popen | None = None

    def ensure(self, wait_s: float = 90) -> bool:
        if is_up(self.url):
            return True
        binary = find_binary()
        if binary is None or not MODEL.exists():
            return False
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(
                [str(binary), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
                 "-t", str(self.threads), "-c", str(CONTEXT), "--no-webui"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if is_up(self.url):
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
