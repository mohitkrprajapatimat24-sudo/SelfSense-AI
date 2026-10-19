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
  compliance/       rules engine and compliance score
  generator/        synthetic shelf generator
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
