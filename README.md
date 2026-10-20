# ShelfSense AI

Autonomous retail compliance agent: one shelf photo in, a scored audit out.
Detects products, price tags and empty gaps, checks them against the planogram
and catalogue, and produces a 0-100 compliance score, the exact violations and a
ranked task list for store staff. Runs fully offline on CPU.

## Setup

Needs Python 3.11.

```
py -3.11 scripts/bootstrap.py          # Windows, GPU build of torch if an NVIDIA GPU is present
py -3.11 scripts/bootstrap.py --cpu    # CPU-only torch
.venv\Scripts\activate
```

This creates `.venv`, installs pinned libraries and downloads every model into
`models/` (YOLO11n, DINOv2-small, RapidOCR, rembg, Qwen2.5-3B GGUF) plus the
llama.cpp CPU runtime into `tools/`.

## Project layout

```
shelfsense/
  schemas.py        data contracts shared by every stage
  data.py           catalogue and planogram loading + validation
  perception/       detector, product matcher, tag OCR, video frames, shelf rows
  compliance/       alignment, rules, score, ranked audit
  generator/        synthetic shelf generator
  eval/             accuracy against the answer keys
  viz.py            draw boxes over a shelf image
data/
  catalogue/catalogue.csv   SKU, name, price, size, sales velocity
  planograms/*.json         bay -> rows -> [SKU, facings]
  raw/<SKU>/                product photos you download (not in git)
  products/<SKU>/           background-removed cutouts (not in git)
  synthetic/                generated datasets (not in git)
tests/
```

## Synthetic shelves (stage 1)

Real store photos never come with a list of what is wrong in them, so the
generator builds shelves from the planogram, injects mistakes on purpose and
keeps the answer key.

```
python -m shelfsense.generator --split preview --n 24      # quick look
python -m shelfsense.generator --split test --n 500        # evaluation set
python -m shelfsense.generator --split train --n 2000 --seed 1
streamlit run shelfsense/generator/preview.py              # browse them
```

Each split gets `images/`, `labels/` (YOLO: product, price_tag, empty_gap),
`answers/` (shelf map, per-slot ground truth, violations, score) and a
`summary.json`. `data/synthetic/dataset.yaml` is ready for Ultralytics.

Mistakes injected: out of stock, low stock, misplaced product, unknown
product, product from another bay, wrong price, wrong label. Camera effects:
tilt, perspective, exposure, colour temperature, glare, vignette, blur, noise,
JPEG compression.

### Using real product photos

Until photos are added, packs are drawn as placeholders from the catalogue
colours. To use real packaging:

1. Put 3-5 front-facing photos per product in `data/raw/<SKU>/`
   (folder names must match the `sku` column in `catalogue.csv`).
2. `python -m shelfsense.generator.prepare_products` removes the backgrounds.
3. Regenerate the datasets; the generator picks up the cutouts automatically.

## Compliance engine (stage 2)

`shelfsense.compliance.engine.audit(shelf_map, planogram, catalogue)` turns
detected boxes into a scored, ranked audit:

1. **Rows**: `perception/shelf_map.py` clusters product bottoms into shelf
   rows; each price tag joins the row just above it.
2. **Alignment**: `compliance/align.py` matches detected rows to planned
   rows, then aligns each row's left-to-right products and empty spaces to the
   planned facings (edit distance plus position). Candidate row placements are
   voted by matching SKUs and the cheapest is kept, so a neighbour's product on
   a slot edge reads as "misplaced", not "everything shifted". Scale comes from
   product spacing, which tilt does not inflate.
3. **Tags**: each slot takes the tag under its left edge.
4. **Rules, ranking, score**: stock-outs first (by revenue at risk), then
   misplacements and wrong prices, then low stock and labels. Low-confidence
   product matches or tag reads become NEEDS_REVIEW instead of a guess.

Measure it against the answer keys (boxes are flattened as a detector would
give them, optionally corrupted):

```
python -m shelfsense.eval.compliance --split test                     # perfect detections
python -m shelfsense.eval.compliance --split test --noise realistic   # simulated perception errors
```

On the 500-shelf test set with perfect detections: 99.8% of shelves get
exactly the right violations, 100% recall on every violation type, mean score
error 0.01 points, about 8 ms per shelf. With simulated perception errors the
engine passes them through about one-to-one (a misidentified product really
does look like a misplaced item), so its accuracy is bounded by stage 3.

`streamlit run shelfsense/generator/preview.py` has an "Engine vs answer key"
tab that shows the audit drawn on the shelf next to the truth.

## Perception (stage 3)

`shelfsense.perception.pipeline.Perception` turns a shelf photo into a
`ShelfMap`; everything runs on CPU through ONNX Runtime (no torch at runtime).

| Step | Model | Notes |
|---|---|---|
| Detect | YOLO11n, fine-tuned, ONNX (10 MB) | classes: product, price_tag, empty_gap; 1024 px input |
| Identify | DINOv2-small int8 ONNX (24 MB) + colour histogram + FAISS | 65% DINOv2 / 35% colour similarity; below a calibrated threshold -> UNKNOWN |
| Read tags | RapidOCR (PaddleOCR models on ONNX) | price regex tolerant of ₹ misreads; SKU from the printed item code, else fuzzy product name |
| Rows | `perception/shelf_map.py` | as in stage 2 |

Video uploads are reduced to their sharpest frame (`perception/preprocess.py`).
Low-confidence identifications and tag reads become NEEDS_REVIEW.

Build the models (GPU used for training when available):

```
python -m shelfsense.generator --split train --n 2000 --seed 1 --max-side 1280
python -m shelfsense.generator --split val --n 300 --seed 2 --max-side 1280
python -m shelfsense.perception.train_detector        # ~1 h on an RTX 4060
python -m shelfsense.perception.build_gallery         # DINOv2 -> ONNX int8, gallery, unknown threshold
python -m shelfsense.eval.perception --split test     # targets + end-to-end
```

Results on the 500-shelf synthetic test set (placeholder packs, 4 CPU threads,
i7-13650HX):

| Target (abstract section 11) | Target | Result |
|---|---|---|
| Product detection mAP@0.5 | >= 0.85 | 0.995 (mAP@0.5:0.95 0.975) |
| SKU identification top-1 | >= 90% | 98.1% |
| Price OCR exact match | >= 90% | 97.4% (tag SKU 97.7%) |
| Empty-gap recall | >= 90% | 99.5% |
| Latency per image | <= 3 s | 1.8 s mean, 2.7 s p90 (detect 0.1, identify 0.9, OCR 0.7) |

End to end, photo to audit: 94.8% violation precision, 94.3% recall, 84.6%
of shelves with exactly the right violations, score error 2.6 points, 0.7
human-review flags per shelf.

**Caveat.** These numbers are on synthetic shelves drawn with placeholder
packs, which are easier than real packaging. Add real product photos
(`data/raw/<SKU>/`, then `prepare_products`), regenerate the datasets,
retrain and rebuild the gallery to get representative numbers, and check on
real phone photos. The abstract's SKU-110K subset is not used yet (a ~13 GB
download); it would mainly help the product detector on real shelves.

## Compliance rules

Every planned slot is checked for foreign items (misplaced / extra), stock
level (out of stock / low stock) and its price tag (wrong label / wrong price).

```
Score = 100 x (0.50 P + 0.25 L + 0.25 A)
P = slots holding only the planned SKU / planned slots
L = correct price tags / expected tags
A = facings holding the planned SKU / planned facings
```

Bands: >= 90 Good, 70-89 Needs attention, < 70 Critical.

## Tests

```
pytest
```
