"""Getting one good shelf image out of an upload (photo or short video)."""

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 4000            # bigger phone photos are shrunk; ~3 px/mm on a 1.2 m bay is plenty for OCR
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
FRAMES_PER_SECOND = 4


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _limit(img: Image.Image) -> Image.Image:
    if max(img.size) > MAX_SIDE:
        f = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)
    return img


def load_image(source: str | Path | bytes) -> Image.Image:
    img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    return _limit(ImageOps.exif_transpose(img).convert("RGB"))


def sharpest_frame(video_path: str | Path) -> tuple[Image.Image, dict]:
    """Sample a few frames per second and keep the least blurry one."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps / FRAMES_PER_SECOND)))
    best, best_score, best_idx, idx = None, -1.0, -1, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small = cv2.resize(rgb, (640, int(640 * rgb.shape[0] / rgb.shape[1])))
                score = sharpness(small)
                if score > best_score:
                    best, best_score, best_idx = rgb, score, idx
        idx += 1
    cap.release()
    if best is None:
        raise ValueError(f"No frames could be read from {video_path}")
    return _limit(Image.fromarray(best)), {"frame": best_idx, "frames": idx, "sharpness": round(best_score, 1)}


def load_upload(path: str | Path) -> tuple[Image.Image, dict]:
    path = Path(path)
    if path.suffix.lower() in VIDEO_TYPES:
        return sharpest_frame(path)
    return load_image(path), {}
