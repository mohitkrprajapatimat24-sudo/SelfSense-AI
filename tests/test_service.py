import json

import pytest
from PIL import Image

from shelfsense.compliance.engine import audit
from shelfsense.data import load_catalogue, load_planograms
from shelfsense.generator.build import generate_sample
from shelfsense.generator.packs import PackLibrary
from shelfsense.paths import ASSETS
from shelfsense.perception.pipeline import DETECTOR_ONNX, EMBEDDER_ONNX, GALLERY
from shelfsense.perception.shelf_map import build_shelf_map, flatten
from shelfsense.report import build_pdf
from shelfsense.service import AuditRecord, History
from shelfsense.viz import draw_audit

needs_models = pytest.mark.skipif(
    not (DETECTOR_ONNX.exists() and EMBEDDER_ONNX.exists() and (GALLERY / "gallery.faiss").exists()),
    reason="trained models not built (see README)")


@pytest.fixture
def record(tmp_path) -> AuditRecord:
    """A real audit record built from an answer key (no models needed)."""
    catalogue = load_catalogue()
    planograms = load_planograms(catalogue)
    img, key = generate_sample(planograms["P01"], catalogue, PackLibrary(catalogue), [5, 5, 5], "t")
    sm = build_shelf_map(key.image_id, key.shelf_map.width, key.shelf_map.height, *flatten(key.shelf_map))
    result = audit(sm, planograms["P01"], catalogue)
    img.save(tmp_path / "photo.jpg")
    draw_audit(img, result).save(tmp_path / "annotated.jpg")
    return AuditRecord(audit_id="20261020-120000-abc123", created_at="2026-10-20T12:00:00", store_id="Store 1",
                       bay_id="Aisle 3, Bay 2", planogram_id="P01", source="test.jpg", result=result,
                       timings_ms={"total": 1200, "detect": 100, "identify": 600, "ocr": 500},
                       image_path=str(tmp_path / "photo.jpg"), annotated_path=str(tmp_path / "annotated.jpg"))


def test_history_round_trip(tmp_path, record):
    h = History(tmp_path / "out")
    h.add(record)
    [row] = h.list(bay_id="Aisle 3, Bay 2")
    assert row["score"] == record.result.score.score
    assert h.get(record.audit_id).result == record.result
    assert h.list(bay_id="elsewhere") == []
    assert h.get("missing") is None


def test_counts_cover_every_violation(record):
    c = record.counts()
    assert sum(c.values()) == len(record.result.violations)


def test_pdf_report(record):
    pdf = build_pdf(record)
    assert pdf.startswith(b"%PDF") and len(pdf) > 20_000


# ---------------------------------------------------------------- API (needs the trained models)

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    from fastapi.testclient import TestClient

    os.environ["SHELFSENSE_OUTPUTS"] = str(tmp_path_factory.mktemp("api"))
    from shelfsense.api.app import app
    with TestClient(app) as c:
        yield c
    os.environ.pop("SHELFSENSE_OUTPUTS")


@needs_models
def test_api_audit_flow(client):
    assert client.get("/health").json()["status"] == "ok"
    assert {p["planogram_id"] for p in client.get("/planograms").json()} >= {"P01", "P04"}

    sample = json.loads((ASSETS / "samples" / "samples.json").read_text())[0]
    with open(ASSETS / "samples" / sample["file"], "rb") as f:
        r = client.post("/audit", files={"file": (sample["file"], f, "image/jpeg")},
                        data={"planogram_id": sample["planogram_id"], "store_id": "Store 9"})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert abs(rec["result"]["score"]["score"] - sample["expected_score"]) <= 10

    audit_id = rec["audit_id"]
    assert client.get("/audits", params={"store_id": "Store 9"}).json()[0]["audit_id"] == audit_id
    assert client.get(f"/audits/{audit_id}").json()["planogram_id"] == sample["planogram_id"]
    img = client.get(f"/audits/{audit_id}/annotated.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    pdf = client.get(f"/audits/{audit_id}/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


@needs_models
def test_api_rejects_bad_input(client):
    assert client.post("/audit", files={"file": ("notes.txt", b"hi", "text/plain")},
                       data={"planogram_id": "P01"}).status_code == 415
    assert client.post("/audit", files={"file": ("a.jpg", b"not an image", "image/jpeg")},
                       data={"planogram_id": "NOPE"}).status_code == 404
    assert client.post("/audit", files={"file": ("a.jpg", b"not an image", "image/jpeg")},
                       data={"planogram_id": "P01"}).status_code == 422
    assert client.get("/audits/missing").status_code == 404
