"""Generate synthetic shelf datasets.

    python -m shelfsense.generator --split test --n 500
    python -m shelfsense.generator --split train --n 2000 --seed 1
    python -m shelfsense.generator --split preview --n 24 --planograms P01,P02

Each split folder gets images/, labels/ (YOLO format) and answers/ (JSON answer
keys), plus a summary.json. data/synthetic/dataset.yaml is rewritten so
Ultralytics can train on whichever splits exist.
"""

import argparse
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..paths import SYNTHETIC
from ..schemas import YOLO_CLASSES
from .batch import init_worker, make_one


def write_dataset_yaml(root: Path) -> None:
    lines = [f"path: {root.resolve().as_posix()}"]
    for split in ("train", "val", "test"):
        if (root / split / "images").exists():
            lines.append(f"{split}: {split}/images")
    lines.append("names:")
    lines += [f"  {i}: {c.value}" for i, c in enumerate(YOLO_CLASSES)]
    (root / "dataset.yaml").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="preview", help="train, val, test or preview")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planograms", help="comma-separated IDs, default all")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", type=Path, default=SYNTHETIC)
    args = ap.parse_args()

    split_dir = args.out / args.split
    if split_dir.exists():
        print(f"Replacing existing {split_dir}")
        shutil.rmtree(split_dir)
    for sub in ("images", "labels", "answers"):
        (split_dir / sub).mkdir(parents=True)

    ids = args.planograms.split(",") if args.planograms else None
    jobs = [(str(split_dir), args.split, args.seed, i, not args.no_augment) for i in range(args.n)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(ids,)) as pool:
        results = []
        for i, r in enumerate(pool.map(make_one, jobs, chunksize=4), 1):
            results.append(r)
            if i % 50 == 0 or i == args.n:
                print(f"  {i}/{args.n} images", flush=True)
    elapsed = time.time() - t0

    counts = Counter(v for r in results for v in r["violations"])
    summary = {
        "split": args.split, "images": args.n, "seed": args.seed, "augmented": not args.no_augment,
        "uses_photos": any(r["photos"] for r in results),
        "perfect_shelves": sum(1 for r in results if not r["violations"]),
        "mean_score": round(float(np.mean([r["score"] for r in results])), 1),
        "bands": dict(Counter(r["band"] for r in results)),
        "violations": dict(sorted(counts.items())),
        "seconds": round(elapsed, 1),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_dataset_yaml(args.out)
    print(json.dumps(summary, indent=2))
    if not summary["uses_photos"]:
        print("\nNote: no product photos found in data/products, so placeholder packs were drawn.\n"
              "Add photos to data/raw/<SKU>/ and run: python -m shelfsense.generator.prepare_products")


if __name__ == "__main__":
    main()
