"""Data contracts shared by every stage of the pipeline.

Coordinates are image pixels. Rows are numbered from the top shelf (1) down,
and positions are 1-based slot indices from the left of a row.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

UNKNOWN_SKU = "UNKNOWN"


# ---------------------------------------------------------------- reference data

class CatalogueItem(BaseModel):
    sku: str
    name: str
    brand: str
    category: str
    price: float                      # catalogue selling price in INR
    width_mm: float                   # front-face size on the shelf
    height_mm: float
    velocity_per_day: float           # assumed units sold per day, for revenue at risk
    color: str = "#888888"            # placeholder pack art, used until real photos exist
    accent: str = "#ffffff"


class PlanogramSlot(BaseModel):
    sku: str
    facings: int = Field(ge=1)


class PlanogramRow(BaseModel):
    row: int = Field(ge=1)
    slots: list[PlanogramSlot]


class Planogram(BaseModel):
    planogram_id: str
    bay_id: str
    description: str = ""
    shelf_width_mm: float
    facing_gap_mm: float = 5.0
    rows: list[PlanogramRow]

    @model_validator(mode="after")
    def _rows_numbered_in_order(self):
        numbers = [r.row for r in self.rows]
        if numbers != list(range(1, len(self.rows) + 1)):
            raise ValueError(f"{self.planogram_id}: rows must be numbered 1..n top to bottom, got {numbers}")
        return self

    def skus(self) -> set[str]:
        return {s.sku for r in self.rows for s in r.slots}

    def home_of(self, sku: str) -> list[tuple[int, int]]:
        """(row, position) of every slot planned for this SKU."""
        return [(r.row, i + 1) for r in self.rows for i, s in enumerate(r.slots) if s.sku == sku]


# ---------------------------------------------------------------- perception output

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class DetClass(str, Enum):
    PRODUCT = "product"
    PRICE_TAG = "price_tag"
    EMPTY_GAP = "empty_gap"


YOLO_CLASSES = [DetClass.PRODUCT, DetClass.PRICE_TAG, DetClass.EMPTY_GAP]


class Detection(BaseModel):
    bbox: BBox
    cls: DetClass
    conf: float = 1.0


class ShelfItem(BaseModel):
    """One facing: a single product front visible on the shelf."""
    bbox: BBox
    sku: str                          # catalogue SKU or UNKNOWN_SKU
    match_conf: float = 1.0


class PriceTag(BaseModel):
    bbox: BBox
    sku: Optional[str] = None         # SKU code printed on the tag
    price: Optional[float] = None
    text: Optional[str] = None
    ocr_conf: float = 1.0


class Gap(BaseModel):
    bbox: BBox


class ShelfRow(BaseModel):
    row: int
    y_top: float
    y_bottom: float
    items: list[ShelfItem] = []       # left to right
    tags: list[PriceTag] = []         # left to right
    gaps: list[Gap] = []


class ShelfMap(BaseModel):
    image_id: str
    width: int
    height: int
    rows: list[ShelfRow]


# ---------------------------------------------------------------- slot-level state

class SlotState(BaseModel):
    """What is actually in one planned slot.

    The generator writes this as ground truth. In stage 2 the compliance
    engine builds the same structure from a ShelfMap by aligning it to the
    planogram, then both go through the same rules.
    """
    row: int
    position: int
    planned_sku: str
    planned_facings: int
    contents: list[Optional[str]]     # one entry per facing position, None = empty
    tag_sku: Optional[str] = None
    tag_price: Optional[float] = None


# ---------------------------------------------------------------- compliance output

class ViolationType(str, Enum):
    OUT_OF_STOCK = "OUT_OF_STOCK"
    LOW_STOCK = "LOW_STOCK"
    MISPLACED = "MISPLACED"
    EXTRA = "EXTRA"                   # unknown, competitor or not planned for this bay
    WRONG_PRICE = "WRONG_PRICE"
    WRONG_LABEL = "WRONG_LABEL"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Violation(BaseModel):
    type: ViolationType
    row: int
    position: int
    expected_sku: Optional[str] = None
    found_sku: Optional[str] = None
    expected_facings: Optional[int] = None
    found_facings: Optional[int] = None
    count: int = 1                    # units affected (facings missing, items misplaced)
    expected_price: Optional[float] = None
    found_price: Optional[float] = None
    target_row: Optional[int] = None  # where a misplaced item belongs
    target_position: Optional[int] = None
    revenue_at_risk: float = 0.0      # INR per day
    action: str = ""


class ScoreBreakdown(BaseModel):
    placement: float                  # P: slots holding only the planned SKU / planned slots
    labels: float                     # L: correct price tags / expected tags
    availability: float               # A: facings holding the planned SKU / planned facings
    score: float                      # 0-100
    band: str


class AuditResult(BaseModel):
    image_id: str
    planogram_id: str
    score: ScoreBreakdown
    violations: list[Violation]
    revenue_at_risk: float
    actions: list[str] = []
