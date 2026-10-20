"""ShelfSense REST API.

    uvicorn shelfsense.api.app:app --port 8000
    curl -F file=@shelf.jpg -F planogram_id=P01 http://localhost:8000/audit

Interactive docs at http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from ..perception.preprocess import VIDEO_TYPES
from ..report import build_pdf
from ..service import OUTPUTS, AuditRecord, AuditService

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_MB = 200

service: AuditService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = AuditService(folder=Path(os.environ.get("SHELFSENSE_OUTPUTS", OUTPUTS)),
                           threads=int(os.environ.get("SHELFSENSE_THREADS", "4")))
    if os.environ.get("SHELFSENSE_PRELOAD", "1") == "1":
        service.perception                          # load models now, not on the first request
    yield


app = FastAPI(title="ShelfSense AI", version="1.0",
              description="One shelf photo in, a scored compliance audit out. Runs offline on CPU.",
              lifespan=lifespan)


def _record(audit_id: str) -> AuditRecord:
    rec = service.history.get(audit_id)
    if rec is None:
        raise HTTPException(404, f"No audit {audit_id}")
    return rec


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": service._perception is not None,
            "planograms": len(service.planograms), "skus": len(service.catalogue)}


@app.get("/planograms")
def planograms():
    return [{"planogram_id": p.planogram_id, "bay_id": p.bay_id, "description": p.description,
             "rows": len(p.rows), "slots": sum(len(r.slots) for r in p.rows)} for p in service.planograms.values()]


@app.post("/audit", response_model=AuditRecord)
def run_audit(file: UploadFile = File(..., description="Shelf photo or short video"),
              planogram_id: str = Form(..., description="e.g. P01"),
              store_id: str = Form("Store 214")):
    # a plain def: FastAPI runs it in a worker thread, so a 2 s audit does not block other requests
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_TYPES | VIDEO_TYPES:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Send a photo or a short video.")
    if planogram_id not in service.planograms:
        raise HTTPException(404, f"Unknown planogram '{planogram_id}'. See /planograms.")
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        return service.audit_bytes(data, file.filename, planogram_id, store_id)
    except (ValueError, OSError) as e:           # unreadable image or video
        raise HTTPException(422, f"Could not read the file: {e}")


@app.get("/audits")
def audits(bay_id: str | None = None, store_id: str | None = None, limit: int = 50):
    return service.history.list(bay_id, store_id, limit)


@app.get("/audits/{audit_id}", response_model=AuditRecord)
def get_audit(audit_id: str):
    return _record(audit_id)


@app.get("/audits/{audit_id}/annotated.jpg")
def annotated(audit_id: str):
    return FileResponse(_record(audit_id).annotated_path, media_type="image/jpeg")


@app.get("/audits/{audit_id}/report.pdf")
def report(audit_id: str):
    rec = _record(audit_id)
    return Response(build_pdf(rec), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="shelfsense_{audit_id}.pdf"'})
