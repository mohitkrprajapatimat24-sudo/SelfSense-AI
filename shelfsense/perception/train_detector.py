"""Fine-tune YOLO11n on the synthetic shelves and export it to ONNX.

    python -m shelfsense.generator --split train --n 2000 --seed 1 --max-side 1280
    python -m shelfsense.generator --split val --n 300 --seed 2 --max-side 1280
    python -m shelfsense.perception.train_detector

Classes: product, price_tag, empty_gap. Uses the GPU when one is available.
The exported model goes to models/detector/shelfsense_yolo11n.onnx and is the
only file the edge runtime needs (no torch).
"""

import argparse
import shutil

from ..paths import MODELS, ROOT, SYNTHETIC
from .pipeline import DETECTOR_ONNX


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", default=str(MODELS / "yolo11n.pt"))
    args = ap.parse_args()

    data = SYNTHETIC / "dataset.yaml"
    for split in ("train", "val"):
        if not (SYNTHETIC / split / "images").exists():
            raise SystemExit(f"Missing {SYNTHETIC / split}. Generate it first (see --help).")

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU (slow)'}")

    model = YOLO(args.base)
    model.train(
        data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=device,
        workers=args.workers, project=str(ROOT / "runs"), name="detector", exist_ok=True,
        patience=15, close_mosaic=10, seed=0, deterministic=False, plots=True,
        fliplr=0.0,          # shelf text and tag layout are not mirror-symmetric
        degrees=2.0, perspective=0.0005, scale=0.4,
    )

    best = ROOT / "runs" / "detector" / "weights" / "best.pt"
    onnx = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=17, simplify=False, dynamic=False)
    DETECTOR_ONNX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, DETECTOR_ONNX)
    shutil.copy(best, DETECTOR_ONNX.with_suffix(".pt"))
    print(f"Saved {DETECTOR_ONNX}")


if __name__ == "__main__":
    main()
