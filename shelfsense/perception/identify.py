"""Which product is this? DINOv2-small embeddings + colour + FAISS nearest neighbour.

DINOv2 captures pack design and shape but is nearly colour-blind, while
packaging is strongly colour-coded (Maggi yellow vs Yippee orange). Each crop
is described by its DINOv2 embedding blended with a hue/saturation histogram;
the blend keeps cosine similarity, so FAISS inner product still works.

The gallery holds a few reference images per SKU (real cutouts from
data/products when present, otherwise rendered placeholder packs). Adding a
SKU means adding photos and rebuilding the gallery; no model is retrained.
A crop whose best match is below the calibrated threshold is UNKNOWN
(possible competitor or new stock).
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..schemas import UNKNOWN_SKU, CatalogueItem

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PAD = (124, 116, 104)
MARGIN_REF = 0.025         # best-vs-second-best gap that counts as fully confident; half of it is the
                           # review line (on val: flags ~2% of right matches, catches ~half the wrong ones)
COLOR_WEIGHT = 0.35        # share of the similarity that comes from colour (tuned on val crops)
DINO_DIM = 384
COLOR_DIM = 18 * 3 * 2 + 3


def preprocess(crops: list[Image.Image]) -> np.ndarray:
    """Pad to square (keeps pack proportions, which help tell products apart), resize, normalise."""
    out = np.empty((len(crops), 3, SIZE, SIZE), np.float32)
    for i, c in enumerate(crops):
        c = c.convert("RGB")
        side = max(c.size)
        sq = Image.new("RGB", (side, side), PAD)
        sq.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
        a = np.asarray(sq.resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255.0
        out[i] = ((a - MEAN) / STD).transpose(2, 0, 1)
    return out


def color_features(crops: list[Image.Image]) -> np.ndarray:
    """Hue x saturation x dark/bright histogram of the pack (edges dropped), plus dark/mid/bright for greys.

    Brightness matters: dark brown and bright orange share a hue.
    """
    out = np.zeros((len(crops), COLOR_DIM), np.float32)
    for i, c in enumerate(crops):
        a = np.asarray(c.convert("RGB"))
        h, w = a.shape[:2]
        a = a[int(h * .08):max(int(h * .92), 1), int(w * .08):max(int(w * .92), 1)]
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        H, S, V = hsv[:, 0] / 180, hsv[:, 1] / 255, hsv[:, 2] / 255
        chroma = (S > 0.2) & (V > 0.15)
        hist = np.histogramdd(np.column_stack([H[chroma], S[chroma], V[chroma]]), bins=[18, 3, 2],
                              range=[[0, 1], [0.2, 1], [0.15, 1]])[0].ravel()
        grey = np.histogram(V[~chroma], bins=3, range=[0, 1])[0]
        f = np.sqrt(np.concatenate([hist, grey]) / max(1, len(H)))     # Hellinger: tames dominant colours
        out[i] = f / (np.linalg.norm(f) + 1e-9)
    return out


class Embedder:
    def __init__(self, onnx_path: Path, threads: int = 4, color_weight: float = COLOR_WEIGHT):
        import onnxruntime as ort

        from .runtime import session_options

        self.session = ort.InferenceSession(str(onnx_path), session_options(threads),
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.color_weight = color_weight

    def dino(self, crops: list[Image.Image], batch: int = 16) -> np.ndarray:
        embs = [self.session.run(None, {self.input_name: preprocess(crops[i:i + batch])})[0]
                for i in range(0, len(crops), batch)]
        e = np.concatenate(embs).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def __call__(self, crops: list[Image.Image], batch: int = 16) -> np.ndarray:
        """Unit vectors whose inner product = (1-w) * DINOv2 cosine + w * colour cosine."""
        if not crops:
            return np.zeros((0, DINO_DIM + COLOR_DIM), np.float32)
        w = self.color_weight
        return np.hstack([np.sqrt(1 - w) * self.dino(crops, batch), np.sqrt(w) * color_features(crops)])


def export_dinov2(src: Path, out: Path) -> None:
    """One-off: HuggingFace DINOv2 -> ONNX that returns the CLS embedding. Needs torch."""
    import torch
    from transformers import AutoModel

    class ClsOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).pooler_output

    model = ClsOnly(AutoModel.from_pretrained(src)).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (torch.zeros(1, 3, SIZE, SIZE),), str(out), input_names=["pixel_values"],
                      output_names=["embedding"], dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)


def quantize_int8(src: Path, out: Path) -> None:
    """Dynamic int8 weights: ~1.7x faster on CPU, 24 MB instead of 88 MB, same accuracy on val."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(out), weight_type=QuantType.QInt8)


# ---------------------------------------------------------------- reference gallery

def _on_background(pack: Image.Image, rng: np.random.Generator) -> Image.Image:
    shade = int(rng.integers(150, 235))
    bg = Image.new("RGB", pack.size, (shade, shade, shade))
    bg.paste(pack, (0, 0), pack)
    return bg


def _variants(pack: Image.Image, rng: np.random.Generator, n: int) -> list[Image.Image]:
    out = [_on_background(pack, rng)]
    for _ in range(n - 1):
        p = pack.rotate(float(rng.uniform(-3, 3)), resample=Image.BICUBIC, expand=True)
        img = _on_background(p, rng)
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.75, 1.2)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.85, 1.15)))
        # store lighting: warm or cool colour cast
        cast = np.array([rng.uniform(0.92, 1.08), 1.0, rng.uniform(0.92, 1.08)], np.float32)
        img = Image.fromarray(np.clip(np.asarray(img, np.float32) * cast, 0, 255).astype(np.uint8))
        if rng.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
        f = float(rng.uniform(0.35, 1.0))       # shelf crops are often small
        img = img.resize((max(8, int(img.width * f)), max(8, int(img.height * f))), Image.BILINEAR)
        out.append(img)
    return out


def reference_images(catalogue: dict[str, CatalogueItem], packs, seed: int = 0,
                     per_photo: int = 6, per_placeholder: int = 12) -> list[tuple[str, Image.Image]]:
    rng = np.random.default_rng(seed)
    refs = []
    for sku, item in catalogue.items():
        w, h = int(item.width_mm * 2), int(item.height_mm * 2)
        if packs.has_photos(sku):
            for v in range(len(packs.cutouts[sku])):
                refs += [(sku, img) for img in _variants(packs.get(sku, w, h, v), rng, per_photo)]
        else:
            refs += [(sku, img) for img in _variants(packs.get(sku, w, h), rng, per_placeholder)]
    return refs


class ProductIndex:
    def __init__(self, index, labels: list[str], unknown_threshold: float = 0.5):
        self.index = index
        self.labels = labels
        self.unknown_threshold = unknown_threshold

    @classmethod
    def build(cls, embedder: Embedder, refs: list[tuple[str, Image.Image]]) -> "ProductIndex":
        import faiss

        embs = embedder([img for _, img in refs])
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        return cls(index, [sku for sku, _ in refs])

    def save(self, folder: Path) -> None:
        import faiss

        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "gallery.faiss"))
        (folder / "gallery.json").write_text(json.dumps(
            {"labels": self.labels, "unknown_threshold": self.unknown_threshold}, indent=1))

    @classmethod
    def load(cls, folder: Path) -> "ProductIndex":
        import faiss

        meta = json.loads((folder / "gallery.json").read_text())
        return cls(faiss.read_index(str(folder / "gallery.faiss")), meta["labels"], meta["unknown_threshold"])

    def best_matches(self, embs: np.ndarray, k: int = 30) -> list[list[tuple[str, float]]]:
        """Per query: SKUs ranked by their best similarity."""
        if not len(embs):
            return []
        sims, ids = self.index.search(embs, min(k, self.index.ntotal))
        out = []
        for row_s, row_i in zip(sims, ids):
            per_sku: dict[str, float] = {}
            for s, i in zip(row_s, row_i):
                if i >= 0:
                    sku = self.labels[i]
                    per_sku[sku] = max(per_sku.get(sku, -1.0), float(s))
            out.append(sorted(per_sku.items(), key=lambda kv: -kv[1]))
        return out

    def identify(self, embs: np.ndarray) -> list[tuple[str, float, float]]:
        """Per query: (sku or UNKNOWN, best similarity, confidence 0-1)."""
        out = []
        for ranked in self.best_matches(embs):
            best_sku, best = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else best - 1
            if best < self.unknown_threshold:
                conf = min(1.0, (self.unknown_threshold - best) / MARGIN_REF)
                out.append((UNKNOWN_SKU, best, conf))
            else:
                conf = min(1.0, (best - second) / MARGIN_REF, (best - self.unknown_threshold) / MARGIN_REF)
                out.append((best_sku, best, max(0.0, conf)))
        return out

    def calibrate(self, embs: np.ndarray, truth: list[str]) -> dict:
        """Pick the unknown threshold that gets the most crops right (known SKUs and unknowns)."""
        ranked = self.best_matches(embs)
        best_sims = np.array([r[0][1] for r in ranked])
        best_skus = [r[0][0] for r in ranked]
        candidates = np.unique(np.round(best_sims, 3))
        scores = []
        for t in candidates:
            pred = [s if b >= t else UNKNOWN_SKU for s, b in zip(best_skus, best_sims)]
            scores.append(np.mean([p == y for p, y in zip(pred, truth)]))
        i = int(np.argmax(scores))
        self.unknown_threshold = float(candidates[i])
        return {"unknown_threshold": self.unknown_threshold, "val_accuracy": float(scores[i]),
                "crops": len(truth), "unknown_crops": int(sum(y == UNKNOWN_SKU for y in truth))}
