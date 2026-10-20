"""Build the product gallery used to identify SKUs.

    python -m shelfsense.perception.build_gallery

1. Exports DINOv2-small to ONNX (once; needs torch).
2. Embeds reference images for every catalogue SKU (photos from data/products,
   or placeholder packs until photos exist) into a FAISS index.
3. Calibrates the "unknown product" threshold on product crops from the
   synthetic val split.

Rerun after adding product photos or catalogue entries. No retraining.
"""

import argparse
import json

import numpy as np
from PIL import Image

from ..data import load_catalogue
from ..generator.build import load_answer
from ..generator.packs import PackLibrary
from ..paths import MODELS, SYNTHETIC
from .identify import Embedder, ProductIndex, export_dinov2, quantize_int8, reference_images
from .pipeline import EMBEDDER_FP32, EMBEDDER_ONNX, GALLERY, crop


def ground_truth_crops(split: str, limit_images: int) -> tuple[list[Image.Image], list[str]]:
    crops, labels = [], []
    for f in sorted((SYNTHETIC / split / "answers").glob("*.json"))[:limit_images]:
        key = load_answer(f)
        img = Image.open(SYNTHETIC / split / "images" / key.image_file).convert("RGB")
        for row in key.shelf_map.rows:
            for it in row.items:
                crops.append(crop(img, it.bbox))
                labels.append(it.sku)
    return crops, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate-split", default="val")
    ap.add_argument("--calibrate-images", type=int, default=150)
    args = ap.parse_args()

    if not EMBEDDER_ONNX.exists():
        if not EMBEDDER_FP32.exists():
            print(f"Exporting DINOv2-small to {EMBEDDER_FP32}")
            export_dinov2(MODELS / "dinov2-small", EMBEDDER_FP32)
        print(f"Quantizing to {EMBEDDER_ONNX}")
        quantize_int8(EMBEDDER_FP32, EMBEDDER_ONNX)
    embedder = Embedder(EMBEDDER_ONNX)

    catalogue = load_catalogue()
    packs = PackLibrary(catalogue)
    refs = reference_images(catalogue, packs)
    with_photos = sorted({s for s, _ in refs if packs.has_photos(s)})
    print(f"{len(refs)} reference images for {len(catalogue)} SKUs "
          f"({len(with_photos)} from real photos, {len(catalogue) - len(with_photos)} placeholder)")
    index = ProductIndex.build(embedder, refs)

    stats = {}
    if (SYNTHETIC / args.calibrate_split / "answers").exists():
        crops, labels = ground_truth_crops(args.calibrate_split, args.calibrate_images)
        stats = index.calibrate(embedder(crops), labels)
        print(f"Unknown-product threshold {stats['unknown_threshold']:.3f}: "
              f"{stats['val_accuracy']:.1%} of {stats['crops']} val crops identified correctly "
              f"({stats['unknown_crops']} unknown)")
    else:
        print(f"No '{args.calibrate_split}' split to calibrate on; keeping threshold {index.unknown_threshold}")

    index.save(GALLERY)
    (GALLERY / "build.json").write_text(json.dumps(
        {"references": len(refs), "skus": len(catalogue), "with_photos": with_photos, **stats}, indent=1))
    print(f"Saved gallery to {GALLERY}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
