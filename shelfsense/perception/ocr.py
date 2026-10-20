"""Reading price tags: which SKU the tag is for, and its price.

RapidOCR runs the PaddleOCR detection and recognition models on ONNX Runtime.
Each tag crop is upscaled 2x and read on its own; text detection is sized for
small crops (the defaults upscale everything to document size, ~15x slower).
"""

import difflib
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..schemas import CatalogueItem

# ₹ is often recognised as a similar-looking currency sign
PRICE_RE = re.compile(r"(?:₹|RS\.?|INR|MRP|€|£|¥)\s*:?\s*(\d{1,5}(?:[.,]\d{1,2})?)", re.IGNORECASE)
RUPEE_AS_DIGIT_RE = re.compile(r"^[17]\d{2,}")   # "₹25.00" read as "125.00" or "725.00"
NO_CURRENCY_PENALTY = 0.5  # applied when a bare number looks like ₹ misread as its first digit
BARE_NUMBER_RE = re.compile(r"^\s*(\d{1,5}(?:[.,]\d{1,2})?)\s*$")
SKU_RE = re.compile(r"[A-Z]{3}-[A-Z0-9&]+-\d+")
UPSCALE = 2
NAME_MATCH_MIN = 0.6


@dataclass
class TagRead:
    sku: str | None
    price: float | None
    text: str
    conf: float


class TagReader:
    def __init__(self, catalogue: dict[str, CatalogueItem], threads: int = 4):
        from rapidocr import RapidOCR

        self.catalogue = catalogue
        self.ocr = RapidOCR(params={
            "Global.log_level": "error",
            "Global.use_cls": False,                      # tags are upright
            "Det.limit_type": "max",
            "Det.limit_side_len": 320,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        })

    def read(self, crop: Image.Image) -> TagRead:
        up = crop.convert("RGB").resize((crop.width * UPSCALE, crop.height * UPSCALE), Image.BICUBIC)
        out = self.ocr(np.asarray(up))
        lines = []
        if out.txts:
            for text, score, box in zip(out.txts, out.scores, out.boxes):
                box = np.asarray(box)
                lines.append((text.strip(), float(score), float(box[:, 1].max() - box[:, 1].min())))
        return self.parse(lines)

    def parse(self, lines: list[tuple[str, float, float]]) -> TagRead:
        """lines: (text, recognition score, line height in px)."""
        text = " | ".join(t for t, _, _ in lines)

        # price: a line with a currency mark wins, else a bare number; the tallest such line
        price, price_score, best = None, 0.0, (-1, 0.0)
        for t, s, h in lines:
            m = PRICE_RE.search(t)
            rank = 2 if m else 0
            if not m:
                m = BARE_NUMBER_RE.match(t)
                rank = 1 if m else 0
            if m and (rank, h) > best:
                best = (rank, h)
                price = float(m.group(1).replace(",", "."))
                suspicious = rank == 1 and RUPEE_AS_DIGIT_RE.match(m.group(1))
                price_score = s * NO_CURRENCY_PENALTY if suspicious else s

        # SKU: the printed item code if it was read cleanly, else the closest product name
        sku, sku_score = None, 0.0
        for t, s, _ in lines:
            for code in SKU_RE.findall(t.upper().replace(" ", "")):
                if code in self.catalogue:
                    sku, sku_score = code, s
        if sku is None:
            sku, sku_score = self._match_name([t for t, _, _ in lines])

        conf = min(price_score if price is not None else 0.0, sku_score if sku else 0.0)
        return TagRead(sku=sku, price=price, text=text, conf=round(conf, 3))

    def _match_name(self, texts: list[str]) -> tuple[str | None, float]:
        best, best_ratio = None, 0.0
        for t in texts:
            t = t.lower()
            if len(t) < 5 or PRICE_RE.search(t):
                continue
            for sku, item in self.catalogue.items():
                # tags often cut the name short, so compare with the same-length prefix
                ratio = difflib.SequenceMatcher(None, t, item.name.lower()[:len(t)]).ratio()
                if ratio > best_ratio:
                    best, best_ratio = sku, ratio
        return (best, best_ratio) if best_ratio >= NAME_MATCH_MIN else (None, 0.0)
