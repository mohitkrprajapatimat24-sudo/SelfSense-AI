"""Loading and validating the catalogue and planograms."""

import csv
import json
from pathlib import Path

from .paths import CATALOGUE_CSV, PLANOGRAMS
from .schemas import CatalogueItem, Planogram


def load_catalogue(path: Path = CATALOGUE_CSV) -> dict[str, CatalogueItem]:
    with open(path, newline="", encoding="utf-8") as f:
        items = [CatalogueItem(**row) for row in csv.DictReader(f)]
    catalogue = {item.sku: item for item in items}
    if len(catalogue) != len(items):
        raise ValueError(f"{path}: duplicate SKUs")
    return catalogue


def load_planogram(path: Path, catalogue: dict[str, CatalogueItem]) -> Planogram:
    planogram = Planogram(**json.loads(Path(path).read_text(encoding="utf-8")))
    validate_planogram(planogram, catalogue)
    return planogram


def load_planograms(catalogue: dict[str, CatalogueItem], folder: Path = PLANOGRAMS) -> dict[str, Planogram]:
    planograms = [load_planogram(p, catalogue) for p in sorted(folder.glob("*.json"))]
    return {p.planogram_id: p for p in planograms}


def row_width_mm(row, planogram: Planogram, catalogue: dict[str, CatalogueItem]) -> float:
    facings = sum(s.facings for s in row.slots)
    product_width = sum(catalogue[s.sku].width_mm * s.facings for s in row.slots)
    return product_width + planogram.facing_gap_mm * (facings + 1)


def validate_planogram(planogram: Planogram, catalogue: dict[str, CatalogueItem]) -> None:
    for row in planogram.rows:
        for slot in row.slots:
            if slot.sku not in catalogue:
                raise ValueError(f"{planogram.planogram_id} row {row.row}: unknown SKU {slot.sku}")
        width = row_width_mm(row, planogram, catalogue)
        if width > planogram.shelf_width_mm:
            raise ValueError(
                f"{planogram.planogram_id} row {row.row}: needs {width:.0f} mm, "
                f"shelf is {planogram.shelf_width_mm:.0f} mm")
