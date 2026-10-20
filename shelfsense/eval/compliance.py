"""Measure the compliance engine against the synthetic answer keys.

The answer key's boxes are flattened (row structure removed) as if they had
just come out of a detector, optionally corrupted to mimic perception errors,
then run through row building, alignment and the rules. The violations found
are compared with the true ones.

    python -m shelfsense.eval.compliance --split test
    python -m shelfsense.eval.compliance --split test --noise realistic
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from ..compliance.engine import audit
from ..data import load_catalogue, load_planograms
from ..generator.build import load_answer
from ..paths import ROOT, SYNTHETIC
from ..perception.shelf_map import build_shelf_map, flatten
from ..schemas import BBox, ViolationType

NOISE_PRESETS = {
    "none": {},
    # roughly what a decent detector + matcher + OCR would get wrong
    "realistic": {"drop_items": 0.02, "misid": 0.02, "low_conf": 0.03, "drop_gaps": 0.3,
                  "drop_tags": 0.02, "misread_price": 0.02, "jitter": 0.04},
    "harsh": {"drop_items": 0.05, "misid": 0.05, "low_conf": 0.05, "drop_gaps": 1.0,
              "drop_tags": 0.05, "misread_price": 0.05, "jitter": 0.08},
}


def corrupt(items, tags, gaps, catalogue, noise: dict, rng: random.Random):
    """Simulate detector / matcher / OCR mistakes."""
    skus = sorted(catalogue)

    def jitter(b: BBox) -> BBox:
        j = noise.get("jitter", 0)
        if not j:
            return b
        dx, dy = b.w * j, b.h * j
        return BBox(x1=b.x1 + rng.gauss(0, dx), y1=b.y1 + rng.gauss(0, dy),
                    x2=b.x2 + rng.gauss(0, dx), y2=b.y2 + rng.gauss(0, dy))

    out_items = []
    for it in items:
        if rng.random() < noise.get("drop_items", 0):
            continue
        it = it.model_copy(update={"bbox": jitter(it.bbox)})
        if rng.random() < noise.get("misid", 0):
            it.sku = rng.choice([s for s in skus if s != it.sku])
        if rng.random() < noise.get("low_conf", 0):
            it.match_conf = 0.3
        out_items.append(it)

    out_tags = []
    for t in tags:
        if rng.random() < noise.get("drop_tags", 0):
            continue
        t = t.model_copy(update={"bbox": jitter(t.bbox)})
        if rng.random() < noise.get("misread_price", 0):
            t.ocr_conf = 0.4 if rng.random() < 0.5 else t.ocr_conf    # half the misreads are flagged by OCR
            t.price = (t.price or 0) + rng.choice([-1, 1, 10, -10])
        out_tags.append(t)

    out_gaps = [g.model_copy(update={"bbox": jitter(g.bbox)}) for g in gaps
                if rng.random() >= noise.get("drop_gaps", 0)]
    return out_items, out_tags, out_gaps


def key_of(v) -> tuple:
    return (v.type.value, v.row, v.position, v.found_sku if v.type in (ViolationType.MISPLACED, ViolationType.EXTRA) else None)


def evaluate(split: str, noise: dict, seed: int = 0, limit: int | None = None) -> dict:
    catalogue = load_catalogue()
    planograms = load_planograms(catalogue)
    files = sorted((SYNTHETIC / split / "answers").glob("*.json"))[:limit]
    rng = random.Random(seed)

    tp, fp, fn = Counter(), Counter(), Counter()
    exact, review_flags, slot_ok, slot_total = 0, 0, 0, 0
    score_err, band_ok = [], 0
    worst = []
    for f in files:
        key = load_answer(f)
        items, tags, gaps = corrupt(*flatten(key.shelf_map), catalogue, noise, rng)
        sm = build_shelf_map(key.image_id, key.shelf_map.width, key.shelf_map.height, items, tags, gaps)
        result = audit(sm, planograms[key.planogram_id], catalogue)

        truth = Counter(key_of(v) for v in key.violations)
        found = Counter(key_of(v) for v in result.violations if v.type != ViolationType.NEEDS_REVIEW)
        review_flags += sum(1 for v in result.violations if v.type == ViolationType.NEEDS_REVIEW)
        for k in truth | found:
            t, p = truth[k], found[k]
            tp[k[0]] += min(t, p)
            fn[k[0]] += max(0, t - p)
            fp[k[0]] += max(0, p - t)
        exact += truth == found

        by_pos = {(s.row, s.position): s for s in key.slots}
        for s in result.slots:
            t = by_pos[(s.row, s.position)]
            slot_total += 1
            slot_ok += (sorted(map(str, s.contents)) == sorted(map(str, t.contents))
                        and s.tag_sku == t.tag_sku and s.tag_price == t.tag_price)

        err = abs(result.score.score - key.score.score)
        score_err.append(err)
        band_ok += result.score.band == key.score.band
        worst.append((err, key.image_id))

    types = sorted(set(tp) | set(fp) | set(fn))
    per_type = {}
    for t in types:
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 1.0
        r = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 1.0
        per_type[t] = {"precision": round(p, 4), "recall": round(r, 4),
                       "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                       "true": tp[t] + fn[t], "found": tp[t] + fp[t]}
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    return {
        "split": split, "images": len(files), "noise": noise,
        "overall": {"precision": round(T / (T + F), 4) if T + F else 1.0,
                    "recall": round(T / (T + N), 4) if T + N else 1.0},
        "per_type": per_type,
        "images_exactly_right": round(exact / len(files), 4),
        "slots_exactly_right": round(slot_ok / slot_total, 4),
        "score_mae": round(float(np.mean(score_err)), 2),
        "band_accuracy": round(band_ok / len(files), 4),
        "needs_review_flags": review_flags,
        "worst_images": [i for _, i in sorted(worst, reverse=True)[:5]],
    }


def print_report(rep: dict) -> None:
    print(f"\n{rep['images']} images from '{rep['split']}', noise: {rep['noise'] or 'none'}\n")
    print(f"{'violation':<14}{'precision':>10}{'recall':>9}{'f1':>8}{'true':>7}{'found':>7}")
    for t, m in rep["per_type"].items():
        print(f"{t:<14}{m['precision']:>10.1%}{m['recall']:>9.1%}{m['f1']:>8.3f}{m['true']:>7}{m['found']:>7}")
    o = rep["overall"]
    print(f"{'ALL':<14}{o['precision']:>10.1%}{o['recall']:>9.1%}")
    print(f"\nimages with exactly the right violations  {rep['images_exactly_right']:.1%}")
    print(f"slots read exactly right                  {rep['slots_exactly_right']:.1%}")
    print(f"score error (mean absolute)               {rep['score_mae']} points")
    print(f"band correct                              {rep['band_accuracy']:.1%}")
    print(f"needs-review flags raised                 {rep['needs_review_flags']}")
    print(f"worst images                              {', '.join(rep['worst_images'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--noise", default="none", choices=list(NOISE_PRESETS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rep = evaluate(args.split, NOISE_PRESETS[args.noise], args.seed, args.limit)
    print_report(rep)
    out = ROOT / "reports" / f"compliance_{args.split}_{args.noise}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
