"""Aligning what was detected with what was planned.

1. Rows: detected rows are matched to planogram rows top to bottom, allowing
   for a detected row that matches nothing (stray boxes) or a planned row that
   was not detected at all.
2. Facings: within a row, the detected left-to-right sequence (products and
   empty spaces) is aligned to the planned facing sequence with an edit
   distance (Needleman-Wunsch). Substituted facings are foreign products,
   deleted facings are empty, inserted products are extras.
3. Tags: the planned layout is mapped onto image x coordinates using the
   aligned facings, and each tag is given to the slot it sits under.
"""

from dataclasses import dataclass, field

import numpy as np

from ..schemas import BBox, CatalogueItem, Planogram, PlanogramRow, ShelfMap, ShelfRow, SlotState

# edit costs: substituting a product is cheaper than deleting + inserting,
# so a foreign product in a planned position is read as "replaced" not "shifted"
COST_SUB = 1.0
COST_DEL = 0.6
COST_INS_ITEM = 0.6
COST_INS_EMPTY = 0.2
COST_EMPTY = 0.1
IMPLICIT_GAP = 0.6     # space between products wider than this x a facing is empty
POSITION_WEIGHT = 0.8  # second pass: extra cost per facing-width of misplacement
OFF_IMAGE = 1.0        # per planned facing that a candidate placement puts outside the image


@dataclass
class Token:
    sku: str | None    # None = empty space
    x1: float
    x2: float
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2


@dataclass
class Review:
    row: int
    position: int
    what: str          # "product" or "tag"
    reason: str


@dataclass
class Alignment:
    slots: list[SlotState]
    slot_boxes: dict[str, BBox]
    reviews: list[Review] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- geometry

def estimate_scale(shelf_map: ShelfMap, catalogue: dict[str, CatalogueItem], planogram: Planogram) -> float:
    """Pixels per millimetre.

    Measured from the spacing of neighbouring products (centre to centre vs
    planned pitch), which is not inflated by tilt the way box widths are.
    Falls back to box widths, then to the shelf filling the image.
    """
    pitch, widths = [], []
    for r in shelf_map.rows:
        known = [it for it in r.items if it.sku in catalogue]
        widths += [it.bbox.w / catalogue[it.sku].width_mm for it in known]
        for a, b in zip(known, known[1:]):
            planned = (catalogue[a.sku].width_mm + catalogue[b.sku].width_mm) / 2 + planogram.facing_gap_mm
            pitch.append((b.bbox.cx - a.bbox.cx) / planned)
    if widths:
        rough = float(np.median(widths))
        # neighbours only: spacing across an empty facing is not a pitch
        pitch = [p for p in pitch if 0.6 * rough < p < 1.3 * rough]
        return float(np.median(pitch)) if len(pitch) >= 3 else rough
    return shelf_map.width * 0.9 / planogram.shelf_width_mm


def planned_layout(row: PlanogramRow, planogram: Planogram, catalogue) -> tuple[list[str], list[float], list[float], list[int]]:
    """Per planned facing: SKU, left and right edge in mm from the row start, and slot index."""
    skus, x1s, x2s, slot_idx = [], [], [], []
    x = planogram.facing_gap_mm
    for i, slot in enumerate(row.slots):
        w = catalogue[slot.sku].width_mm
        for _ in range(slot.facings):
            skus.append(slot.sku)
            x1s.append(x)
            x2s.append(x + w)
            slot_idx.append(i)
            x += w + planogram.facing_gap_mm
    return skus, x1s, x2s, slot_idx


def row_tokens(row: ShelfRow, facing_px: float) -> list[Token]:
    """Left-to-right products and empty facings, splitting wide gaps into facings."""
    raw = [Token(it.sku, it.bbox.x1, it.bbox.x2, it.match_conf) for it in row.items]
    for g in row.gaps:
        n = max(1, int(round(g.bbox.w / facing_px)))
        step = g.bbox.w / n
        raw += [Token(None, g.bbox.x1 + k * step, g.bbox.x1 + (k + 1) * step) for k in range(n)]
    raw.sort(key=lambda t: t.cx)

    tokens: list[Token] = []
    for t in raw:
        if tokens:
            space = t.x1 - tokens[-1].x2
            n = int(round(space / facing_px)) if space > IMPLICIT_GAP * facing_px else 0
            step = space / n if n else 0
            tokens += [Token(None, tokens[-1].x2 + k * step, tokens[-1].x2 + (k + 1) * step) for k in range(n)]
        tokens.append(t)
    return tokens


# ---------------------------------------------------------------- sequence alignment

def align_sequence(planned: list[str], tokens: list[Token], expected_cx: list[float] | None = None,
                   facing_px: float = 1.0) -> list[tuple[int | None, int | None]]:
    return _align(planned, tokens, expected_cx, facing_px)[1]


def _align(planned: list[str], tokens: list[Token], expected_cx: list[float] | None = None,
           facing_px: float = 1.0) -> tuple[float, list[tuple[int | None, int | None]]]:
    """Needleman-Wunsch alignment. Returns the total cost and (planned index, token index) pairs in order.

    With expected_cx (where each planned facing should appear in the image),
    pairing a token with a facing far from it costs extra, which settles cases
    the sequence alone cannot, like a neighbour's product sitting on a slot edge.
    """
    n, m = len(planned), len(tokens)
    cost = np.zeros((n + 1, m + 1))
    cost[1:, 0] = np.arange(1, n + 1) * COST_DEL
    for j in range(1, m + 1):
        cost[0, j] = cost[0, j - 1] + (COST_INS_EMPTY if tokens[j - 1].sku is None else COST_INS_ITEM)

    def pair_cost(i, j):
        t = tokens[j].sku
        c = COST_EMPTY if t is None else (0.0 if t == planned[i] else COST_SUB)
        if expected_cx is not None:
            c += POSITION_WEIGHT * min(abs(tokens[j].cx - expected_cx[i]) / facing_px, 2.0)
        return c

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ins = COST_INS_EMPTY if tokens[j - 1].sku is None else COST_INS_ITEM
            cost[i, j] = min(cost[i - 1, j - 1] + pair_cost(i - 1, j - 1),
                             cost[i - 1, j] + COST_DEL,
                             cost[i, j - 1] + ins)

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isclose(cost[i, j], cost[i - 1, j - 1] + pair_cost(i - 1, j - 1)):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and np.isclose(cost[i, j], cost[i - 1, j] + COST_DEL):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return float(cost[n, m]), pairs[::-1]


def candidate_offsets(planned: list[str], centres_mm: list[float], tokens: list[Token],
                      scale: float, facing_px: float, top: int = 6) -> list[float]:
    """Where the row could start in the image (px offset of planned mm 0).

    Every detected product that has the same SKU as some planned facing votes
    for the offset that would put it there; the most-voted offsets are kept.
    """
    votes = [t.cx - scale * centres_mm[p] for t in tokens if t.sku is not None
             for p, sku in enumerate(planned) if sku == t.sku]
    if not votes:
        return []
    votes = np.array(votes)
    tol = 0.25 * facing_px
    support = [(int(np.sum(np.abs(votes - v) < tol)), float(np.median(votes[np.abs(votes - v) < tol]))) for v in votes]
    support.sort(key=lambda s: -s[0])
    picked: list[float] = []
    for _, v in support:
        if all(abs(v - q) >= tol for q in picked):
            picked.append(v)
        if len(picked) == top:
            break
    return picked


def _row_similarity(det: ShelfRow, plan: PlanogramRow) -> float:
    planned = {}
    for s in plan.slots:
        planned[s.sku] = planned.get(s.sku, 0) + s.facings
    seen = {}
    for sku in [it.sku for it in det.items] + [t.sku for t in det.tags]:
        seen[sku] = seen.get(sku, 0) + 1
    return sum(min(c, planned.get(k, 0)) for k, c in seen.items())


def match_rows(shelf_map: ShelfMap, planogram: Planogram) -> dict[int, int]:
    """Planogram row number -> index into shelf_map.rows (monotonic, best total overlap)."""
    det, plan = shelf_map.rows, planogram.rows
    n, m = len(plan), len(det)
    score = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # small bonus per match so that equal counts map in order even with empty rows
            score[i, j] = max(score[i - 1, j - 1] + _row_similarity(det[j - 1], plan[i - 1]) + 0.01,
                              score[i - 1, j], score[i, j - 1])
    out, i, j = {}, n, m
    while i > 0 and j > 0:
        if np.isclose(score[i, j], score[i - 1, j - 1] + _row_similarity(det[j - 1], plan[i - 1]) + 0.01):
            out[plan[i - 1].row] = j - 1
            i, j = i - 1, j - 1
        elif np.isclose(score[i, j], score[i - 1, j]):
            i -= 1
        else:
            j -= 1
    return out


# ---------------------------------------------------------------- full alignment

def align(shelf_map: ShelfMap, planogram: Planogram, catalogue: dict[str, CatalogueItem],
          min_match_conf: float = 0.5, min_ocr_conf: float = 0.6) -> Alignment:
    scale = estimate_scale(shelf_map, catalogue, planogram)
    row_map = match_rows(shelf_map, planogram)
    result = Alignment(slots=[], slot_boxes={})

    for j, det in enumerate(shelf_map.rows):
        if j not in row_map.values() and (det.items or det.tags):
            result.notes.append(f"Detected row {j + 1} ({len(det.items)} products) does not match any planned row")

    for prow in planogram.rows:
        skus, x1s, x2s, slot_of = planned_layout(prow, planogram, catalogue)
        n_slots = len(prow.slots)
        contents: list[list[str | None]] = [[None] * s.facings for s in prow.slots]
        facing_no = [k - slot_of.index(slot_of[k]) for k in range(len(skus))]
        low_conf = [False] * n_slots
        det = shelf_map.rows[row_map[prow.row]] if prow.row in row_map else None

        offset, row_scale = None, scale
        if det is None:
            result.notes.append(f"Row {prow.row} not found in the image; treated as empty")
        else:
            mean_w = float(np.mean([x2 - x1 for x1, x2 in zip(x1s, x2s)]))
            facing_px = (mean_w + planogram.facing_gap_mm) * scale
            tokens = row_tokens(det, facing_px)
            centres_mm = [(a + b) / 2 for a, b in zip(x1s, x2s)]

            # where does the row sit in the image? try the best-supported offsets,
            # align with position cost for each, keep the cheapest
            pairs = align_sequence(skus, tokens)
            best = None
            for off in candidate_offsets(skus, centres_mm, tokens, scale, facing_px):
                expected = [off + scale * c for c in centres_mm]
                cost, p = _align(skus, tokens, expected, facing_px)
                # one bay per photo: a placement that pushes planned facings out of the image is unlikely
                cost += OFF_IMAGE * sum(1 for x in expected if x < 0 or x > shelf_map.width)
                if best is None or cost < best[0]:
                    best = (cost, p, off)
            if best is not None:
                _, pairs, offset = best
                matched = [(centres_mm[p], tokens[t].cx) for p, t in pairs
                           if p is not None and t is not None and tokens[t].sku == skus[p]]
                mms = np.array([mm for mm, _ in matched])
                if len(matched) >= 4 and np.ptp(mms) > 0.4 * (x2s[-1] - x1s[0]):
                    # refine with this row's own px/mm, fitted on the facings that matched
                    fit_a, fit_b = np.polyfit(mms, [px for _, px in matched], 1)
                    if 0.85 * scale < fit_a < 1.15 * scale:
                        row_scale, offset = float(fit_a), float(fit_b)
                        pairs = align_sequence(skus, tokens, [offset + row_scale * c for c in centres_mm], facing_px)
            elif any(t.sku is None for t in tokens):
                # only empty spaces seen: place the row by its first gap
                offset = tokens[0].x1 - scale * x1s[0]
            elif det.tags:
                # nothing on the shelf at all: place the row by its first tag (tags sit at slot starts)
                offset = det.tags[0].bbox.x1 - scale * x1s[0]

            last_slot = 0
            for p, t in pairs:
                if p is not None:
                    last_slot = slot_of[p]
                if t is None:
                    continue
                tok = tokens[t]
                if p is not None:
                    slot = slot_of[p]
                    contents[slot][facing_no[p]] = tok.sku
                elif tok.sku is not None:
                    # extra product squeezed in: belongs to the slot it sits in
                    slot = last_slot
                    if offset is not None:
                        mm = (tok.cx - offset) / row_scale
                        slot = slot_of[min(range(len(skus)), key=lambda k: abs((x1s[k] + x2s[k]) / 2 - mm))]
                    contents[slot].append(tok.sku)
                else:
                    continue
                if tok.sku is not None and tok.conf < min_match_conf:
                    low_conf[slot] = True

        # slot extents in the image
        slot_x = []
        for i in range(n_slots):
            ks = [k for k in range(len(skus)) if slot_of[k] == i]
            slot_x.append((x1s[ks[0]], x2s[ks[-1]]))
        if det is not None and offset is not None:
            y1, y2 = det.y_top, det.y_bottom
            for i, (a, b) in enumerate(slot_x):
                result.slot_boxes[f"{prow.row}-{i + 1}"] = BBox(x1=offset + a * row_scale, y1=y1,
                                                                x2=offset + b * row_scale, y2=y2)

        # tags: each slot takes the tag sitting under it, closest to its left edge
        tag_for: dict[int, int] = {}
        if det is not None and det.tags:
            for k, tag in enumerate(det.tags):
                if offset is None:
                    continue
                mm = (tag.bbox.cx - offset) / row_scale
                inside = [i for i, (a, b) in enumerate(slot_x) if a - planogram.facing_gap_mm <= mm <= b + planogram.facing_gap_mm]
                i = inside[0] if inside else min(range(n_slots), key=lambda s: min(abs(mm - slot_x[s][0]), abs(mm - slot_x[s][1])))
                prev = tag_for.get(i)
                left = offset + slot_x[i][0] * row_scale
                if prev is None or abs(tag.bbox.x1 - left) < abs(det.tags[prev].bbox.x1 - left):
                    tag_for[i] = k

        for i, slot in enumerate(prow.slots):
            tag = det.tags[tag_for[i]] if det is not None and i in tag_for else None
            result.slots.append(SlotState(
                row=prow.row, position=i + 1, planned_sku=slot.sku, planned_facings=slot.facings,
                contents=contents[i], tag_sku=tag.sku if tag else None, tag_price=tag.price if tag else None))
            if low_conf[i]:
                result.reviews.append(Review(prow.row, i + 1, "product", "low-confidence product match"))
            if tag is not None and tag.ocr_conf < min_ocr_conf:
                result.reviews.append(Review(prow.row, i + 1, "tag", "price tag could not be read reliably"))
    return result
