"""Drawing a ShelfMap over its image."""

from PIL import Image, ImageDraw

from .generator.packs import font
from .schemas import UNKNOWN_SKU, AuditResult, ShelfMap, ViolationType as V

COLORS = {
    "product": (34, 197, 94),
    "unknown": (249, 115, 22),
    "price_tag": (59, 130, 246),
    "empty_gap": (239, 68, 68),
    "row": (168, 85, 247),
}


def draw_shelf_map(img: Image.Image, shelf_map: ShelfMap, products: bool = True, tags: bool = True,
                   gaps: bool = True, rows: bool = False, labels: bool = True) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, out.width // 500)
    f = font(max(10, out.width // 110), bold=True)

    def box(b, color, text=None):
        d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=lw)
        if text and labels:
            l, t, r, bt = d.textbbox((b.x1, b.y1), text, font=f)
            d.rectangle([l, t - (bt - t) - 2, r + 4, b.y1], fill=color)
            d.text((b.x1 + 2, b.y1 - (bt - t) - 2), text, font=f, fill=(255, 255, 255))

    for row in shelf_map.rows:
        if rows:
            d.rectangle([0, row.y_top, out.width - 1, row.y_bottom], outline=COLORS["row"], width=lw)
            d.text((6, row.y_top + 4), f"Row {row.row}", font=f, fill=COLORS["row"])
        if products:
            for it in row.items:
                color = COLORS["unknown"] if it.sku == UNKNOWN_SKU else COLORS["product"]
                box(it.bbox, color, it.sku)
        if tags:
            for tg in row.tags:
                text = f"{tg.sku} {tg.price:g}" if tg.price is not None else tg.sku
                box(tg.bbox, COLORS["price_tag"], text)
        if gaps:
            for g in row.gaps:
                box(g.bbox, COLORS["empty_gap"], "EMPTY")
    return out


STATUS_COLORS = {
    "ok": (34, 197, 94),
    "violation": (220, 38, 38),
    "low": (217, 119, 6),
    "review": (124, 58, 237),
}
SHORT = {
    V.OUT_OF_STOCK: "EMPTY", V.LOW_STOCK: "LOW", V.MISPLACED: "WRONG", V.EXTRA: "EXTRA",
    V.WRONG_PRICE: "PRICE", V.WRONG_LABEL: "LABEL", V.NEEDS_REVIEW: "CHECK",
}


def draw_audit(img: Image.Image, result: AuditResult) -> Image.Image:
    """Every planned slot outlined green (OK), amber (low stock), red (violation) or purple (review)."""
    out = img.convert("RGB").copy()
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    lw = max(3, out.width // 400)
    f = font(max(11, out.width // 95), bold=True)

    by_slot: dict[str, list] = {}
    for v in result.violations:
        by_slot.setdefault(f"{v.row}-{v.position}", []).append(v)

    for key, b in result.slot_boxes.items():
        vs = by_slot.get(key, [])
        kinds = {v.type for v in vs}
        if not vs:
            status = "ok"
        elif kinds == {V.NEEDS_REVIEW}:
            status = "review"
        elif kinds == {V.LOW_STOCK}:
            status = "low"
        else:
            status = "violation"
        color = STATUS_COLORS[status]
        d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color + (255,), width=lw,
                    fill=color + (40,) if status != "ok" else None)
        if vs:
            text = " ".join(dict.fromkeys(SHORT[v.type] for v in vs))
            l, t, r, bt = d.textbbox((0, 0), text, font=f)
            d.rectangle([b.x1, b.y1, b.x1 + (r - l) + 8, b.y1 + (bt - t) + 8], fill=color + (230,))
            d.text((b.x1 + 4, b.y1 + 4 - t), text, font=f, fill=(255, 255, 255, 255))
    out = out.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB")
