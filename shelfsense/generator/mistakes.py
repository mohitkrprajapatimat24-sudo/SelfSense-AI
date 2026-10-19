"""Turning a perfect planogram into a realistic, imperfect shelf state."""

import numpy as np
from pydantic import BaseModel

from ..schemas import UNKNOWN_SKU, CatalogueItem, Planogram, SlotState

# relative frequency of each mistake type
MISTAKE_WEIGHTS = {
    "out_of_stock": 0.20,
    "low_stock": 0.25,
    "misplaced": 0.18,
    "unknown_product": 0.07,
    "stray_product": 0.05,
    "wrong_price": 0.14,
    "wrong_label": 0.11,
}
PERFECT_SHELF_RATE = 0.12
PRICE_ERRORS = [-5, -2, 2, 5, 10]


class Mistake(BaseModel):
    kind: str
    row: int
    position: int
    detail: str


def perfect_state(planogram: Planogram, catalogue: dict[str, CatalogueItem]) -> list[SlotState]:
    return [
        SlotState(row=r.row, position=i + 1, planned_sku=s.sku, planned_facings=s.facings,
                  contents=[s.sku] * s.facings, tag_sku=s.sku, tag_price=catalogue[s.sku].price)
        for r in planogram.rows for i, s in enumerate(r.slots)
    ]


def _intact(slot: SlotState) -> bool:
    return all(c == slot.planned_sku for c in slot.contents)


def _apply(kind: str, slots: list[SlotState], planogram: Planogram,
           catalogue: dict[str, CatalogueItem], rng: np.random.Generator) -> Mistake | None:
    def pick(candidates):
        return candidates[int(rng.integers(len(candidates)))] if candidates else None

    if kind == "out_of_stock":
        s = pick([s for s in slots if _intact(s)])
        if s is None:
            return None
        s.contents = [None] * len(s.contents)
        return Mistake(kind=kind, row=s.row, position=s.position, detail=f"all {s.planned_facings} facings removed")

    if kind == "low_stock":
        s = pick([s for s in slots if _intact(s) and s.planned_facings >= 2])
        if s is None:
            return None
        k = int(rng.integers(1, s.planned_facings))
        idx = range(len(s.contents) - k, len(s.contents)) if rng.random() < 0.6 else range(k)
        for i in idx:
            s.contents[i] = None
        return Mistake(kind=kind, row=s.row, position=s.position, detail=f"{k} of {s.planned_facings} facings removed")

    if kind in ("misplaced", "unknown_product", "stray_product"):
        s = pick([s for s in slots if s.planned_sku in s.contents])
        if s is None:
            return None
        if kind == "misplaced":
            other = pick(sorted(planogram.skus() - {s.planned_sku}))
        elif kind == "stray_product":
            other = pick(sorted(set(catalogue) - planogram.skus()))
        else:
            other = UNKNOWN_SKU
        if other is None:
            return None
        own = [i for i, c in enumerate(s.contents) if c == s.planned_sku]
        k = 1 if kind != "misplaced" else int(rng.integers(1, min(2, len(own)) + 1))
        for i in rng.choice(own, size=k, replace=False):
            s.contents[int(i)] = other
        return Mistake(kind=kind, row=s.row, position=s.position, detail=f"{k} facing(s) replaced by {other}")

    if kind == "wrong_price":
        s = pick([s for s in slots if s.tag_sku == s.planned_sku and s.tag_price == catalogue[s.planned_sku].price])
        if s is None:
            return None
        price = catalogue[s.planned_sku].price
        delta = pick([d for d in PRICE_ERRORS if price + d > 0])
        s.tag_price = price + delta
        return Mistake(kind=kind, row=s.row, position=s.position, detail=f"tag shows {s.tag_price:g} instead of {price:g}")

    if kind == "wrong_label":
        s = pick([s for s in slots if s.tag_sku == s.planned_sku])
        if s is None:
            return None
        other = pick(sorted(planogram.skus() - {s.planned_sku}))
        s.tag_sku, s.tag_price = other, catalogue[other].price
        return Mistake(kind=kind, row=s.row, position=s.position, detail=f"tag is for {other}")

    raise ValueError(kind)


def inject_mistakes(slots: list[SlotState], planogram: Planogram, catalogue: dict[str, CatalogueItem],
                    rng: np.random.Generator, max_mistakes: int = 8) -> list[Mistake]:
    if rng.random() < PERFECT_SHELF_RATE:
        return []
    n = min(max_mistakes, 1 + int(rng.poisson(2.0)))
    kinds = list(MISTAKE_WEIGHTS)
    p = np.array(list(MISTAKE_WEIGHTS.values()))
    mistakes = []
    for kind in rng.choice(kinds, size=n, p=p / p.sum()):
        m = _apply(str(kind), slots, planogram, catalogue, rng)
        if m is not None:
            mistakes.append(m)
    return mistakes
