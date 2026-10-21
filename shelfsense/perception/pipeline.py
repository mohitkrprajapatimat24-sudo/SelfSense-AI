"""Shelf image -> ShelfMap: detect, identify products, read tags, group into rows."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..data import load_catalogue
from ..paths import MODELS
from ..schemas import BBox, CatalogueItem, DetClass, Gap, PriceTag, ShelfItem, ShelfMap
from .detector import Detector
from .identify import Embedder, ProductIndex
from .ocr import TagReader
from .preprocess import sharpness
from .runtime import limit_threads
from .shelf_map import build_shelf_map

DETECTOR_ONNX = MODELS / "detector" / "shelfsense_yolo11n.onnx"
EMBEDDER_FP32 = MODELS / "dinov2-small.onnx"
EMBEDDER_ONNX = MODELS / "dinov2-small-int8.onnx"
GALLERY = MODELS / "gallery"
CROP_PAD = 0.02


BLURRY = 60.0          # Laplacian variance at 1280 px below which the photo is worth retaking


@dataclass
class PerceptionResult:
    shelf_map: ShelfMap
    timings_ms: dict = field(default_factory=dict)
    detections: int = 0
    sharpness: float = 0.0

    @property
    def blurry(self) -> bool:
        return self.sharpness < BLURRY


def crop(img: Image.Image, b: BBox, pad: float = CROP_PAD) -> Image.Image:
    px, py = b.w * pad, b.h * pad
    return img.crop((max(0, int(b.x1 - px)), max(0, int(b.y1 - py)),
                     min(img.width, int(b.x2 + px)), min(img.height, int(b.y2 + py))))


class Perception:
    def __init__(self, catalogue: dict[str, CatalogueItem] | None = None, threads: int = 4,
                 detector: Path = DETECTOR_ONNX, embedder: Path = EMBEDDER_ONNX, gallery: Path = GALLERY):
        for p, how in [(detector, "python -m shelfsense.perception.train_detector"),
                       (embedder, "python -m shelfsense.perception.build_gallery"),
                       (gallery / "gallery.faiss", "python -m shelfsense.perception.build_gallery")]:
            if not p.exists():
                raise FileNotFoundError(f"{p} is missing. Create it with: {how}")
        limit_threads(threads)
        self.catalogue = catalogue or load_catalogue()
        self.detector = Detector(detector, threads)
        self.embedder = Embedder(embedder, threads)
        self.index = ProductIndex.load(gallery)
        self.reader = TagReader(self.catalogue, threads)

    def __call__(self, img: Image.Image, image_id: str = "upload") -> PerceptionResult:
        t = {}
        t0 = time.perf_counter()
        dets = self.detector(img)
        t["detect"] = (time.perf_counter() - t0) * 1000

        products = [d for d in dets if d.cls == DetClass.PRODUCT]
        t0 = time.perf_counter()
        ids = self.index.identify(self.embedder([crop(img, d.bbox) for d in products]))
        items = [ShelfItem(bbox=d.bbox, sku=sku, match_conf=round(conf, 3))
                 for d, (sku, _, conf) in zip(products, ids)]
        t["identify"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tags = []
        for d in (d for d in dets if d.cls == DetClass.PRICE_TAG):
            read = self.reader.read(crop(img, d.bbox, 0.06))
            tags.append(PriceTag(bbox=d.bbox, sku=read.sku, price=read.price, text=read.text,
                                 ocr_conf=read.conf))
        t["ocr"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        gaps = [Gap(bbox=d.bbox) for d in dets if d.cls == DetClass.EMPTY_GAP]
        shelf_map = build_shelf_map(image_id, img.width, img.height, items, tags, gaps)
        t["shelf_map"] = (time.perf_counter() - t0) * 1000
        t["total"] = sum(t.values())
        small = img.copy()
        small.thumbnail((1280, 1280))
        return PerceptionResult(shelf_map, {k: round(v, 1) for k, v in t.items()}, len(dets),
                                round(sharpness(np.asarray(small)), 1))
