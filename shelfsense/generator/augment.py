"""Camera-like distortions: tilt, lighting, blur, noise. Boxes are moved with the image."""

import cv2
import numpy as np
from PIL import Image

from ..schemas import BBox, ShelfMap

MIN_BOX_PX = 4


def _transform_box(box: BBox, H: np.ndarray, w: int, h: int) -> BBox | None:
    pts = np.array([[box.x1, box.y1], [box.x2, box.y1], [box.x1, box.y2], [box.x2, box.y2]], np.float32)
    out = cv2.perspectiveTransform(pts[None], H)[0]
    x1, y1 = np.clip(out.min(0), 0, [w, h])
    x2, y2 = np.clip(out.max(0), 0, [w, h])
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None
    return BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


def _warp_map(shelf_map: ShelfMap, H: np.ndarray) -> ShelfMap:
    w, h = shelf_map.width, shelf_map.height
    m = shelf_map.model_copy(deep=True)
    for row in m.rows:
        for group in (row.items, row.tags, row.gaps):
            kept = []
            for obj in group:
                b = _transform_box(obj.bbox, H, w, h)
                if b is not None:
                    obj.bbox = b
                    kept.append(obj)
            group[:] = kept
        band = _transform_box(BBox(x1=0, y1=row.y_top, x2=w, y2=row.y_bottom), H, w, h)
        if band is not None:
            row.y_top, row.y_bottom = band.y1, band.y2
    return m


def _perspective(w: int, h: int, rng: np.random.Generator, strength: float) -> np.ndarray:
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    jitter = rng.uniform(-strength, strength, (4, 2)) * [w, h]
    # keep the shelf roughly level: corners on the same edge move together vertically
    angle = np.deg2rad(rng.uniform(-2, 2))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centre = np.array([w / 2, h / 2])
    dst = (src - centre) @ rot.T + centre + jitter
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def augment(img: Image.Image, shelf_map: ShelfMap, rng: np.random.Generator) -> tuple[Image.Image, ShelfMap]:
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]

    # geometry: slight tilt and perspective, as from a hand-held phone
    H = _perspective(w, h, rng, strength=float(rng.uniform(0, 0.03)))
    a = cv2.warpPerspective(a, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    shelf_map = _warp_map(shelf_map, H)

    # lighting: exposure, contrast, colour temperature, vignette
    a = a * rng.uniform(0.75, 1.2) + rng.uniform(-25, 20)
    a = (a - 128) * rng.uniform(0.85, 1.15) + 128
    a *= np.array([rng.uniform(0.92, 1.08), 1.0, rng.uniform(0.92, 1.08)], np.float32)
    if rng.random() < 0.4:
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        a *= (1 - rng.uniform(0.1, 0.3) * np.clip(r - 0.4, 0, None))[..., None]
    if rng.random() < 0.3:  # glare band from ceiling lights
        x0 = rng.uniform(0, w)
        xx = np.arange(w)
        a += (rng.uniform(15, 40) * np.exp(-((xx - x0) / (w * 0.08)) ** 2))[None, :, None]

    # sensor: focus blur and hand shake of a few pixels (about 1-3 mm at phone-photo resolution)
    roll = rng.random()
    if roll < 0.25:
        k = int(rng.choice([3, 5]))
        a = cv2.GaussianBlur(a, (k, k), 0)
    elif roll < 0.4:
        k = int(rng.integers(3, 9))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        if rng.random() < 0.5:
            kernel = kernel.T
        a = cv2.filter2D(a, -1, kernel)
    a += rng.normal(0, rng.uniform(0, 6), a.shape)

    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), shelf_map
