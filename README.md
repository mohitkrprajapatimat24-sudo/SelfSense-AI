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
  service.py        one audit end to end + history
  report.py         PDF report
  slm/              Qwen2.5-3B task writer (llama.cpp server) + template fallback
  api/              FastAPI app
  dashboard/        Streamlit store dashboard
  viz.py            draw boxes and audits over a shelf image
assets/
  fonts/            Noto Sans (with ₹), Noto Sans Symbols 2
  samples/          demo shelves and a hand-held video
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

## Dashboard, API and reports (stage 4)

```
streamlit run shelfsense/dashboard/app.py          # store dashboard
uvicorn shelfsense.api.app:app --port 8000         # REST API, docs at /docs
```

**Dashboard.** Pick a sample shelf (`assets/samples/`, including a hand-held
video) or upload a photo or short video of one bay, then *Run audit*. It shows
the annotated shelf (green / amber / red / purple per slot), the compliance
score with its band, issue counts, revenue at risk, and the ranked actions.
Tabs list every violation and every slot, the bay's score history, and how
the photo was read (timings, sharpest video frame, notes). *Download PDF
report* gives a one-page summary for staff.

**API.**

| Method | Path | |
|---|---|---|
| GET | `/health` | models loaded, planograms, SKUs |
| GET | `/planograms` | bays that can be audited |
| POST | `/audit` | multipart `file` (photo or video), `planogram_id`, optional `store_id` |
| GET | `/audits?bay_id=&store_id=` | audit history |
| GET | `/audits/{id}` | full record |
| GET | `/audits/{id}/annotated.jpg` | annotated shelf |
| GET | `/audits/{id}/report.pdf` | PDF report |

```
curl -F file=@assets/samples/P02_test_00009.jpg -F planogram_id=P02 http://localhost:8000/audit
```

Audits are kept in `outputs/` (SQLite index + one folder per audit); set
`SHELFSENSE_OUTPUTS` to store them elsewhere and `SHELFSENSE_THREADS` to
limit CPU threads.

Items sent to human review are left out of the score rather than counted as
failures; the score then says how much of the shelf it covers ("based on 93%
of checks").

## Language model and edge deployment (stage 5)

**Task writer.** After each audit, Qwen2.5-3B-Instruct (4-bit GGUF, run by
the llama.cpp server on CPU) rewrites the ranked violations into short staff
instructions and a one-line summary for the manager. The rules engine still
decides what is wrong and in which order; the model only words it. Guards in
`shelfsense/slm/writer.py`:

- output is constrained to JSON with exactly one task per violation;
- each task must keep its location ("Row 2 Pos 3", and the destination for a
  move), use the right kind of verb (restock / move / remove / fix tag) and
  name the right product and no other;
- the summary may only use numbers and products from the audit;
- anything that fails falls back to the deterministic template, and so does
  everything if the model is missing or slow (`SHELFSENSE_SLM=off` forces it).

```
python -m shelfsense.eval.slm --split test --n 30
```

On 30 test shelves, 88% of task lines and 87% of summaries come from the
model, every one checked; about 13 s per shelf on 4 CPU threads.

**Edge container.** `docker/Dockerfile` builds one image with the dashboard,
the API, the vision models and Qwen. No torch, nothing downloaded at run
time; llama.cpp is compiled from source for the image.

```
docker compose -f docker/compose.yml up --build      # 4 vCPU / 8 GB, dashboard :8501, API :8000
docker run --rm --cpus 4 --memory 8g --network none shelfsense benchmark   # offline benchmark
```

To move it to another computer without internet:
`docker save shelfsense | gzip > shelfsense.tar.gz`, copy, then
`docker load < shelfsense.tar.gz`.

Benchmark inside the container, `--cpus 4 --memory 8g --network none`
(i7-13650HX host, Docker Desktop):

| | Result |
|---|---|
| Internet | none (checked by the benchmark) |
| Audit per image (target <= 3 s, language model excluded) | 1.85 s mean, 2.31 s p90 |
| Stages | detect 0.14 s, identify 0.79 s, read tags 0.75 s, rules 0.01 s |
| Task list by Qwen2.5-3B | ~26 s per audit, once per report |
| Peak memory, all processes incl. Qwen | 4.2 GB of 8 GB |
| Models on disk | detector 11 MB, product embedder 24 MB, gallery 0.3 MB, Qwen 2.1 GB |
| Image size | 5.7 GB |

Inside a CPU quota every library would otherwise start one thread per host
core and ONNX Runtime threads would spin while idle; capping threads and
turning spinning off (`shelfsense/perception/runtime.py`, `SHELFSENSE_SPIN=0`
in the image) took the audit from 5.5 s to 1.85 s.

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
