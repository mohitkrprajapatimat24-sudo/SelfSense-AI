"""Grouping loose detections into shelf rows.

Products stand on the shelf, so the bottoms of products (and empty gaps) on
one shelf line up. Rows are found by clustering those bottom edges. Price tags
sit on the shelf edge just below their row.
"""

import numpy as np

from ..schemas import Gap, PriceTag, ShelfItem, ShelfMap, ShelfRow

ROW_SPLIT = 0.5        # new row when bottoms differ by more than this x median product height
TAG_ROW_REACH = 0.8    # a tag belongs to the row above if within this x median product height


def _cluster(values: list[float], threshold: float) -> list[list[int]]:
    order = np.argsort(values)
    groups: list[list[int]] = []
    last = None
    for i in order:
        if last is None or values[i] - last > threshold:
            groups.append([])
        groups[-1].append(int(i))
        last = values[i]
    return groups


def build_shelf_map(image_id: str, width: int, height: int, items: list[ShelfItem],
                    tags: list[PriceTag], gaps: list[Gap]) -> ShelfMap:
    standing = [("item", o) for o in items] + [("gap", o) for o in gaps]
    heights = [o.bbox.h for _, o in standing] or [t.bbox.h * 4 for t in tags] or [height / 4]
    unit = float(np.median(heights))

    rows: list[dict] = []
    for group in _cluster([o.bbox.y2 for _, o in standing], ROW_SPLIT * unit):
        members = [standing[i] for i in group]
        rows.append({
            "bottom": float(np.median([o.bbox.y2 for _, o in members])),
            "top": min(o.bbox.y1 for _, o in members),
            "items": [o for k, o in members if k == "item"],
            "gaps": [o for k, o in members if k == "gap"],
            "tags": [],
        })

    # tags go to the nearest row whose shelf edge is just above them;
    # tags with no row above (a completely empty shelf) start rows of their own
    orphans = []
    for tag in tags:
        above = [r for r in rows if r["bottom"] <= tag.bbox.cy + 0.25 * unit]
        best = max(above, key=lambda r: r["bottom"], default=None)
        if best is not None and tag.bbox.cy - best["bottom"] <= TAG_ROW_REACH * unit:
            best["tags"].append(tag)
        else:
            orphans.append(tag)
    for group in _cluster([t.bbox.cy for t in orphans], ROW_SPLIT * unit):
        members = [orphans[i] for i in group]
        top = min(t.bbox.y1 for t in members)
        rows.append({"bottom": top, "top": top - unit, "items": [], "gaps": [], "tags": members})

    rows.sort(key=lambda r: r["bottom"])
    out = []
    for n, r in enumerate(rows, 1):
        y_bottom = max([r["bottom"]] + [t.bbox.y2 for t in r["tags"]])
        out.append(ShelfRow(row=n, y_top=r["top"], y_bottom=y_bottom,
                            items=sorted(r["items"], key=lambda o: o.bbox.cx),
                            tags=sorted(r["tags"], key=lambda o: o.bbox.cx),
                            gaps=sorted(r["gaps"], key=lambda o: o.bbox.cx)))
    return ShelfMap(image_id=image_id, width=width, height=height, rows=out)


def flatten(shelf_map: ShelfMap) -> tuple[list[ShelfItem], list[PriceTag], list[Gap]]:
    """Drop the row structure, as if the boxes had just come out of a detector."""
    items = [o for r in shelf_map.rows for o in r.items]
    tags = [o for r in shelf_map.rows for o in r.tags]
    gaps = [o for r in shelf_map.rows for o in r.gaps]
    return items, tags, gaps
