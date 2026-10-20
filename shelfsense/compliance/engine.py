"""Shelf map + planogram -> scored, ranked audit."""

from dataclasses import dataclass, field

from ..schemas import AuditResult, CatalogueItem, Planogram, ShelfMap, Violation, ViolationType as V
from .align import align
from .rules import DEFAULT_WEIGHTS, compute_score, evaluate_slots

# what staff should do first: lost sales and legal exposure, then fixable drift, then housekeeping
PRIORITY = {
    V.OUT_OF_STOCK: 1,
    V.MISPLACED: 2,
    V.WRONG_PRICE: 2,
    V.LOW_STOCK: 3,
    V.WRONG_LABEL: 3,
    V.EXTRA: 4,
    V.NEEDS_REVIEW: 5,
}
PRODUCT_CHECKS = {V.OUT_OF_STOCK, V.LOW_STOCK, V.MISPLACED, V.EXTRA}
TAG_CHECKS = {V.WRONG_PRICE, V.WRONG_LABEL}


@dataclass
class AuditConfig:
    min_match_conf: float = 0.5     # product identifications below this go to human review
    min_ocr_conf: float = 0.6       # price tag reads below this go to human review
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def rank(violations: list[Violation]) -> list[Violation]:
    return sorted(violations, key=lambda v: (PRIORITY[v.type], -v.revenue_at_risk, v.row, v.position))


def audit(shelf_map: ShelfMap, planogram: Planogram, catalogue: dict[str, CatalogueItem],
          config: AuditConfig = AuditConfig()) -> AuditResult:
    al = align(shelf_map, planogram, catalogue, config.min_match_conf, config.min_ocr_conf)
    violations = evaluate_slots(al.slots, planogram, catalogue)

    # uncertain reads are never guessed: drop the checks they would feed, ask a human instead
    for review in al.reviews:
        checks = PRODUCT_CHECKS if review.what == "product" else TAG_CHECKS
        violations = [v for v in violations
                      if not (v.row == review.row and v.position == review.position and v.type in checks)]
        violations.append(Violation(
            type=V.NEEDS_REVIEW, row=review.row, position=review.position,
            action=f"Check Row {review.row}, Pos {review.position} by hand: {review.reason}"))

    violations = rank(violations)
    unsure_products = {(r.row, r.position) for r in al.reviews if r.what == "product"}
    unsure_tags = {(r.row, r.position) for r in al.reviews if r.what == "tag"}
    return AuditResult(
        image_id=shelf_map.image_id, planogram_id=planogram.planogram_id,
        score=compute_score(al.slots, catalogue, config.weights, unsure_products, unsure_tags),
        violations=violations,
        revenue_at_risk=round(sum(v.revenue_at_risk for v in violations), 2),
        actions=[v.action for v in violations],
        slots=al.slots, slot_boxes=al.slot_boxes, notes=al.notes)
