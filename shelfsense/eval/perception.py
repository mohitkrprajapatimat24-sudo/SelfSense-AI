"""Measure perception against the abstract's targets, then the whole pipeline end to end.

    python -m shelfsense.eval.perception --split test
    python -m shelfsense.eval.perception --split test --limit 100 --skip-map

Targets (section 11 of the abstract):
    product detection mAP@0.5   >= 0.85
    SKU identification top-1    >= 90%
    price OCR exact match       >= 90%
    empty-gap recall            >= 90%
    latency per image (4 vCPU)  <= 3 s, excluding the language model
"""

import argparse
import json
import time
from collections import Counter

import numpy as np
from PIL import Image

from ..compliance.engine import audit
from ..data import load_catalogue, load_planograms
from ..generator.build import load_answer
from ..paths import ROOT, SYNTHETIC
from ..perception.pipeline import DETECTOR_ONNX, Perception, crop
from ..schemas import UNKNOWN_SKU, BBox
from .compliance import AuditScorer, print_report

TARGETS = {"detection_map50": 0.85, "sku_top1": 0.90, "price_exact": 0.90, "gap_recall": 0.90, "latency_s": 3.0}


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-9)


def detection_map(split: str, imgsz: int) -> dict:
    """Ultralytics validation of the exported ONNX model on this split (CPU)."""
    from ultralytics import YOLO

    data = SYNTHETIC / f"_eval_{split}.yaml"
    data.write_text((SYNTHETIC / "dataset.yaml").read_text().replace("val: val/images", f"val: {split}/images"))
    m = YOLO(str(DETECTOR_ONNX), task="detect").val(data=str(data), imgsz=imgsz, batch=1, device="cpu",
                                                    plots=False, verbose=False)
    data.unlink()
    names = m.names
    return {"map50": round(float(m.box.map50), 4),
            "per_class_map50": {names[i]: round(float(v), 4) for i, v in enumerate(m.box.ap50)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-map", action="store_true", help="skip the Ultralytics mAP run")
    args = ap.parse_args()

    catalogue = load_catalogue()
    planograms = load_planograms(catalogue)
    perception = Perception(catalogue, threads=args.threads)
    files = sorted((SYNTHETIC / args.split / "answers").glob("*.json"))[:args.limit]

    sku_ok = sku_n = unk_tp = unk_true = unk_pred = review = 0
    confusion = Counter()
    price_ok = tag_sku_ok = tag_n = 0
    gap_hit = gap_n = 0
    latency, audit_ms = [], []
    scorer = AuditScorer()

    for n, f in enumerate(files, 1):
        key = load_answer(f)
        img = Image.open(SYNTHETIC / args.split / "images" / key.image_file).convert("RGB")
        items = [it for r in key.shelf_map.rows for it in r.items]
        tags = [t for r in key.shelf_map.rows for t in r.tags]
        gaps = [g for r in key.shelf_map.rows for g in r.gaps]

        # identification on ground-truth crops (isolates the matcher from the detector)
        ids = perception.index.identify(perception.embedder([crop(img, it.bbox) for it in items]))
        for it, (sku, _, conf) in zip(items, ids):
            sku_n += 1
            sku_ok += sku == it.sku
            review += conf < 0.5
            unk_true += it.sku == UNKNOWN_SKU
            unk_pred += sku == UNKNOWN_SKU
            unk_tp += sku == it.sku == UNKNOWN_SKU
            if sku != it.sku:
                confusion[(it.sku, sku)] += 1

        # OCR on ground-truth tag crops
        for t in tags:
            read = perception.reader.read(crop(img, t.bbox, 0.06))
            tag_n += 1
            price_ok += read.price is not None and abs(read.price - t.price) < 0.01
            tag_sku_ok += read.sku == t.sku

        # full pipeline: detect -> identify -> OCR -> rows -> audit
        res = perception(img, key.image_id)
        t0 = time.perf_counter()
        result = audit(res.shelf_map, planograms[key.planogram_id], catalogue)
        audit_ms.append((time.perf_counter() - t0) * 1000)
        latency.append(res.timings_ms | {"audit": audit_ms[-1]})
        scorer.add(key, result)

        pred_gaps = [g.bbox for r in res.shelf_map.rows for g in r.gaps]
        for g in gaps:
            gap_n += 1
            gap_hit += any(iou(g.bbox, p) >= 0.3 for p in pred_gaps)

        if n % 25 == 0 or n == len(files):
            print(f"  {n}/{len(files)} images, {np.mean([l['total'] for l in latency]):.0f} ms per image", flush=True)

    stages = {k: round(float(np.mean([l[k] for l in latency])), 1) for k in latency[0]}
    total_s = [(l["total"] + l["audit"]) / 1000 for l in latency]
    rep = {
        "split": args.split, "images": len(files), "threads": args.threads,
        "sku_top1": round(sku_ok / sku_n, 4),
        "unknown_recall": round(unk_tp / unk_true, 4) if unk_true else None,
        "unknown_precision": round(unk_tp / unk_pred, 4) if unk_pred else None,
        "sku_low_confidence_rate": round(review / sku_n, 4),
        "top_confusions": [f"{a} -> {b}: {c}" for (a, b), c in confusion.most_common(8)],
        "price_exact": round(price_ok / tag_n, 4),
        "tag_sku_accuracy": round(tag_sku_ok / tag_n, 4),
        "gap_recall": round(gap_hit / gap_n, 4) if gap_n else None,
        "latency_ms_mean": stages,
        "latency_s_p90": round(float(np.percentile(total_s, 90)), 2),
        "end_to_end": scorer.report(),
    }
    if not args.skip_map:
        print("Running detection mAP on the ONNX model ...")
        from onnxruntime import InferenceSession
        imgsz = InferenceSession(str(DETECTOR_ONNX), providers=["CPUExecutionProvider"]).get_inputs()[0].shape[2]
        rep["detection"] = detection_map(args.split, imgsz)

    print("\nPerception vs targets")
    rows = [("product detection mAP@0.5", rep.get("detection", {}).get("map50"), TARGETS["detection_map50"], "{:.3f}"),
            ("SKU identification top-1", rep["sku_top1"], TARGETS["sku_top1"], "{:.1%}"),
            ("price OCR exact match", rep["price_exact"], TARGETS["price_exact"], "{:.1%}"),
            ("empty-gap recall", rep["gap_recall"], TARGETS["gap_recall"], "{:.1%}")]
    for name, val, target, fmt in rows:
        if val is not None:
            print(f"  {name:<28}{fmt.format(val):>8}   target {fmt.format(target)}  {'OK' if val >= target else 'BELOW'}")
    lat = float(np.mean(total_s))
    print(f"  {'latency per image (mean)':<28}{lat:>7.2f}s   target {TARGETS['latency_s']:.1f}s  "
          f"{'OK' if lat <= TARGETS['latency_s'] else 'ABOVE'}   (p90 {rep['latency_s_p90']}s, {args.threads} threads)")
    print(f"  stage ms: {stages}")
    if rep.get("detection"):
        print(f"  per-class mAP@0.5: {rep['detection']['per_class_map50']}")
    print(f"  tag SKU accuracy {rep['tag_sku_accuracy']:.1%}, unknown recall {rep['unknown_recall']}, "
          f"low-confidence matches {rep['sku_low_confidence_rate']:.1%}")
    if rep["top_confusions"]:
        print(f"  top confusions: {rep['top_confusions'][:4]}")
    print_report({"split": args.split, "noise": "real perception pipeline", **rep["end_to_end"]})

    out = ROOT / "reports" / f"perception_{args.split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
