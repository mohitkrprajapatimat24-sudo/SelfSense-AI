"""One place that runs an audit end to end and remembers it.

Used by both the API and the dashboard:
    upload -> best frame -> perception -> compliance engine -> annotated image -> history
"""

import json
import os
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from .compliance.engine import AuditConfig, audit
from .data import load_catalogue, load_planograms
from .paths import ROOT
from .perception.preprocess import VIDEO_TYPES, load_image, load_upload
from .schemas import AuditResult, ViolationType as V
from .slm.writer import template
from .viz import draw_audit

OUTPUTS = ROOT / "outputs"


class AuditRecord(BaseModel):
    audit_id: str
    created_at: str                 # ISO local time
    store_id: str
    bay_id: str
    planogram_id: str
    source: str                     # uploaded file name
    result: AuditResult
    timings_ms: dict = {}
    sharpness: float = 0.0
    blurry: bool = False
    frame_info: dict = {}           # for videos: which frame was audited
    summary: str = ""               # plain-language summary for the manager
    tasks: list[str] = []           # one staff instruction per violation, same order
    tasks_source: str = ""          # "slm", "slm+template" or "template"
    tasks_ms: float = 0.0
    image_path: str = ""
    annotated_path: str = ""

    def counts(self) -> dict[str, int]:
        c = {t: 0 for t in V}
        for v in self.result.violations:
            c[v.type] += 1
        return {
            "out_of_stock": c[V.OUT_OF_STOCK],
            "misplaced": c[V.MISPLACED],
            "low_stock": c[V.LOW_STOCK],
            "price_label": c[V.WRONG_PRICE] + c[V.WRONG_LABEL],
            "extra": c[V.EXTRA],
            "review": c[V.NEEDS_REVIEW],
        }


class History:
    """Audit index in SQLite; full records as JSON next to their images."""

    def __init__(self, folder: Path = OUTPUTS):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.db = folder / "shelfsense.db"
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS audits (
                audit_id TEXT PRIMARY KEY, created_at TEXT, store_id TEXT, bay_id TEXT, planogram_id TEXT,
                score REAL, band TEXT, revenue_at_risk REAL, violations INTEGER, record_path TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.db)

    def audit_dir(self, audit_id: str) -> Path:
        d = self.folder / "audits" / audit_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add(self, record: AuditRecord) -> None:
        path = self.audit_dir(record.audit_id) / "record.json"
        path.write_text(record.model_dump_json(indent=1), encoding="utf-8")
        r = record.result
        actionable = sum(1 for v in r.violations if v.type != V.NEEDS_REVIEW)
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO audits VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (record.audit_id, record.created_at, record.store_id, record.bay_id, record.planogram_id,
                       r.score.score, r.score.band, r.revenue_at_risk, actionable, str(path)))

    def list(self, bay_id: str | None = None, store_id: str | None = None, limit: int = 100) -> list[dict]:
        q, args = "SELECT * FROM audits WHERE 1=1", []
        if bay_id:
            q, args = q + " AND bay_id = ?", args + [bay_id]
        if store_id:
            q, args = q + " AND store_id = ?", args + [store_id]
        q += " ORDER BY created_at DESC LIMIT ?"
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(row) for row in c.execute(q, args + [limit])]

    def get(self, audit_id: str) -> AuditRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT record_path FROM audits WHERE audit_id = ?", (audit_id,)).fetchone()
        if not row or not Path(row[0]).exists():
            return None
        return AuditRecord(**json.loads(Path(row[0]).read_text(encoding="utf-8")))


class AuditService:
    def __init__(self, folder: Path = OUTPUTS, threads: int = 4, config: AuditConfig | None = None,
                 use_slm: bool | None = None):
        self.catalogue = load_catalogue()
        self.planograms = load_planograms(self.catalogue)
        self.history = History(folder)
        self.threads = threads
        self.config = config or AuditConfig()
        self.use_slm = os.environ.get("SHELFSENSE_SLM", "on") != "off" if use_slm is None else use_slm
        self._perception = None
        self._writer = None

    @property
    def perception(self):
        if self._perception is None:            # models load on first use (a few seconds)
            from .perception.pipeline import Perception
            self._perception = Perception(self.catalogue, threads=self.threads)
        return self._perception

    @property
    def writer(self):
        if self._writer is None:
            from .slm.writer import ActionWriter
            self._writer = ActionWriter(self.catalogue, threads=self.threads)
        return self._writer

    def write_tasks(self, record: AuditRecord) -> AuditRecord:
        """Have the language model write the task list (tens of seconds on CPU); template if unavailable."""
        from .slm.writer import template

        if self.use_slm:
            tl = self.writer.write(record.result, record.bay_id, record.store_id)
        else:
            tl = template(record.result, record.bay_id)
        record.summary, record.tasks, record.tasks_source, record.tasks_ms = tl.summary, tl.tasks, tl.source, tl.ms
        self.history.add(record)
        return record

    def audit_image(self, img: Image.Image, planogram_id: str, store_id: str = "Store 214",
                    source: str = "upload", frame_info: dict | None = None) -> AuditRecord:
        if planogram_id not in self.planograms:
            raise KeyError(f"Unknown planogram {planogram_id}; known: {sorted(self.planograms)}")
        planogram = self.planograms[planogram_id]
        now = datetime.now()
        audit_id = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

        perceived = self.perception(img, audit_id)
        t0 = time.perf_counter()
        result = audit(perceived.shelf_map, planogram, self.catalogue, self.config)
        perceived.timings_ms["audit"] = round((time.perf_counter() - t0) * 1000, 1)
        if perceived.blurry:
            result.notes.append("The photo looks blurry; retake it if any result seems wrong.")

        folder = self.history.audit_dir(audit_id)
        img.save(folder / "photo.jpg", quality=90)
        draw_audit(img, result).save(folder / "annotated.jpg", quality=88)
        quick = template(result, planogram.bay_id)       # instant; the language model can rewrite it after
        record = AuditRecord(
            summary=quick.summary, tasks=quick.tasks, tasks_source=quick.source,
            audit_id=audit_id, created_at=now.isoformat(timespec="seconds"), store_id=store_id,
            bay_id=planogram.bay_id, planogram_id=planogram_id, source=source, result=result,
            timings_ms=perceived.timings_ms, sharpness=perceived.sharpness, blurry=perceived.blurry,
            frame_info=frame_info or {}, image_path=str(folder / "photo.jpg"),
            annotated_path=str(folder / "annotated.jpg"))
        self.history.add(record)
        return record

    def audit_file(self, path: Path, planogram_id: str, store_id: str = "Store 214") -> AuditRecord:
        img, info = load_upload(path)
        return self.audit_image(img, planogram_id, store_id, Path(path).name, info)

    def audit_bytes(self, data: bytes, filename: str, planogram_id: str, store_id: str = "Store 214") -> AuditRecord:
        suffix = Path(filename).suffix.lower()
        if suffix in VIDEO_TYPES:                # OpenCV reads videos from disk
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / f"upload{suffix}"
                p.write_bytes(data)
                img, info = load_upload(p)
            return self.audit_image(img, planogram_id, store_id, filename, info)
        return self.audit_image(load_image(data), planogram_id, store_id, filename)
