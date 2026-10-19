"""Drawing a ShelfMap over its image."""

from PIL import Image, ImageDraw

from .generator.packs import font
from .schemas import UNKNOWN_SKU, ShelfMap

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
