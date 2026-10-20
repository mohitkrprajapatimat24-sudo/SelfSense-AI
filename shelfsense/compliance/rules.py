"""Deterministic compliance rules applied to every planned slot.

Per slot, in order:
  1. foreign items      -> MISPLACED (planned elsewhere in this bay) or EXTRA (anything else)
  2. planned SKU count  -> OUT_OF_STOCK (none left) or LOW_STOCK (fewer than planned)
  3. price tag          -> WRONG_LABEL (tag is for another SKU or missing) or WRONG_PRICE

Unlike a strict "first failed check" flow, a slot can raise several
violations, e.g. a foreign item that pushed out the planned product is both
MISPLACED and OUT_OF_STOCK, and each needs its own staff action.
"""

from ..schemas import (UNKNOWN_SKU, CatalogueItem, Planogram, ScoreBreakdown,
                       SlotState, Violation, ViolationType)

DEFAULT_WEIGHTS = {"placement": 0.50, "labels": 0.25, "availability": 0.25}
PRICE_TOLERANCE = 0.01


def _name(sku: str, catalogue: dict[str, CatalogueItem]) -> str:
    return catalogue[sku].name if sku in catalogue else "unknown product"


def _target_slot(sku: str, slots: list[SlotState], planogram: Planogram) -> tuple[int, int]:
    """Home slot for a misplaced SKU: the first one with room, else the first one."""
    homes = planogram.home_of(sku)
    by_pos = {(s.row, s.position): s for s in slots}
    for row, pos in homes:
        slot = by_pos.get((row, pos))
        if slot and slot.contents.count(sku) < slot.planned_facings:
            return row, pos
    return homes[0]


def evaluate_slot(slot: SlotState, slots: list[SlotState], planogram: Planogram,
                  catalogue: dict[str, CatalogueItem]) -> list[Violation]:
    out: list[Violation] = []
    r, p, sku = slot.row, slot.position, slot.planned_sku
    item = catalogue[sku]
    planned = planogram.skus()

    # 1. foreign items, one violation per foreign SKU in the slot
    foreign: dict[str, int] = {}
    for c in slot.contents:
        if c is not None and c != sku:
            foreign[c] = foreign.get(c, 0) + 1
    for other, count in foreign.items():
        if other in planned:
            tr, tp = _target_slot(other, slots, planogram)
            out.append(Violation(
                type=ViolationType.MISPLACED, row=r, position=p, expected_sku=sku, found_sku=other,
                count=count, target_row=tr, target_position=tp,
                action=f"Move {_name(other, catalogue)} from R{r}-P{p} to Row {tr}, Pos {tp}"))
        else:
            why = ("possible competitor or new stock" if other == UNKNOWN_SKU
                   else "not planned for this bay")
            out.append(Violation(
                type=ViolationType.EXTRA, row=r, position=p, expected_sku=sku, found_sku=other,
                count=count,
                action=f"Remove {_name(other, catalogue)} from Row {r}, Pos {p} ({why})"))

    # 2. availability of the planned SKU
    have = min(slot.contents.count(sku), slot.planned_facings)
    missing = slot.planned_facings - have
    daily_value = item.velocity_per_day * item.price
    if have == 0:
        out.append(Violation(
            type=ViolationType.OUT_OF_STOCK, row=r, position=p, expected_sku=sku,
            expected_facings=slot.planned_facings, found_facings=0, count=missing,
            revenue_at_risk=round(daily_value, 2),
            action=f"Restock {item.name} at Row {r}, Pos {p} ({missing} units)"))
    elif missing > 0:
        out.append(Violation(
            type=ViolationType.LOW_STOCK, row=r, position=p, expected_sku=sku,
            expected_facings=slot.planned_facings, found_facings=have, count=missing,
            revenue_at_risk=round(daily_value * missing / slot.planned_facings, 2),
            action=f"Replenish {item.name} at Row {r}, Pos {p}: {missing} of "
                   f"{slot.planned_facings} facings empty"))

    # 3. price tag
    if slot.tag_sku is None and slot.tag_price is None:
        out.append(Violation(
            type=ViolationType.WRONG_LABEL, row=r, position=p, expected_sku=sku,
            expected_price=item.price,
            action=f"Add missing tag at Row {r}, Pos {p}: {item.name} ₹{item.price:.2f}"))
    elif slot.tag_sku != sku:
        out.append(Violation(
            type=ViolationType.WRONG_LABEL, row=r, position=p, expected_sku=sku,
            found_sku=slot.tag_sku, expected_price=item.price, found_price=slot.tag_price,
            action=f"Replace tag at Row {r}, Pos {p}: shows {_name(slot.tag_sku, catalogue)}, "
                   f"should be {item.name} ₹{item.price:.2f}"))
    elif slot.tag_price is None or abs(slot.tag_price - item.price) > PRICE_TOLERANCE:
        shown = "no price" if slot.tag_price is None else f"₹{slot.tag_price:.2f}"
        out.append(Violation(
            type=ViolationType.WRONG_PRICE, row=r, position=p, expected_sku=sku, found_sku=sku,
            expected_price=item.price, found_price=slot.tag_price,
            action=f"Reprint tag: {item.name} shows {shown}, catalogue ₹{item.price:.2f}"))
    return out


def evaluate_slots(slots: list[SlotState], planogram: Planogram,
                   catalogue: dict[str, CatalogueItem]) -> list[Violation]:
    return [v for s in slots for v in evaluate_slot(s, slots, planogram, catalogue)]


def band(score: float) -> str:
    if score >= 90:
        return "Good"
    if score >= 70:
        return "Needs attention"
    return "Critical"


def compute_score(slots: list[SlotState], catalogue: dict[str, CatalogueItem],
                  weights: dict[str, float] = DEFAULT_WEIGHTS,
                  unsure_products: set[tuple[int, int]] = frozenset(),
                  unsure_tags: set[tuple[int, int]] = frozenset()) -> ScoreBreakdown:
    """Score = 100 x (wP*P + wL*L + wA*A), as defined in the abstract.

    Slots whose products or tag go to human review are left out of the parts
    they would affect, so an unreadable tag neither passes nor fails the shelf.
    """
    def ratio(hits, total):
        return hits / total if total else 1.0

    sure_p = [s for s in slots if (s.row, s.position) not in unsure_products]
    sure_t = [s for s in slots if (s.row, s.position) not in unsure_tags]
    placement = ratio(sum(
        1 for s in sure_p
        if s.planned_sku in s.contents and all(c in (None, s.planned_sku) for c in s.contents)), len(sure_p))
    labels = ratio(sum(
        1 for s in sure_t
        if s.tag_sku == s.planned_sku and s.tag_price is not None
        and abs(s.tag_price - catalogue[s.planned_sku].price) <= PRICE_TOLERANCE), len(sure_t))
    availability = ratio(sum(min(s.contents.count(s.planned_sku), s.planned_facings) for s in sure_p),
                         sum(s.planned_facings for s in sure_p))
    score = 100 * (weights["placement"] * placement + weights["labels"] * labels
                   + weights["availability"] * availability)
    score = round(score, 1)
    verified = (len(sure_p) + len(sure_t)) / (2 * len(slots)) if slots else 1.0
    return ScoreBreakdown(placement=round(placement, 4), labels=round(labels, 4),
                          availability=round(availability, 4), score=score, band=band(score),
                          verified=round(verified, 4))
