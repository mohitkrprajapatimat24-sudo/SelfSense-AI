"""Drawing a shelf state into an image, with exact boxes for everything drawn."""

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from ..schemas import (UNKNOWN_SKU, BBox, CatalogueItem, Gap, Planogram, PriceTag,
                       ShelfItem, ShelfMap, ShelfRow, SlotState)
from .packs import PackLibrary, fit_font, font, random_unknown_style

SIDE_MM = 35          # bay uprights
TOP_MM = 50
BOTTOM_MM = 40
LIP_MM = 40           # shelf edge strip that holds the price tags
TAG_H_MM = 30
TAG_W_MM = 70
HEADROOM = 1.12

BACK_PANELS = [(232, 228, 220), (214, 218, 224), (242, 242, 240), (205, 195, 178), (70, 72, 78)]
LIP_COLORS = [(244, 244, 244), (190, 196, 202), (28, 86, 156), (196, 38, 40), (40, 40, 44)]
TAG_COLORS = [(255, 255, 255), (255, 255, 255), (255, 236, 96)]


def _background(w: int, h: int, rng: np.random.Generator) -> Image.Image:
    base = np.array(BACK_PANELS[int(rng.integers(len(BACK_PANELS)))], dtype=np.float32)
    grad = np.linspace(1.04, 0.90, h, dtype=np.float32)[:, None, None]
    img = np.ones((h, w, 3), np.float32) * base * grad
    img += rng.normal(0, 3, (h, w, 1))
    bg = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    if rng.random() < 0.4:  # pegboard
        d = ImageDraw.Draw(bg)
        step = int(rng.integers(22, 34))
        hole = tuple(int(c * 0.7) for c in base)
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=hole)
    return bg


def _draw_tag(img: Image.Image, box: BBox, sku: str, price: float, catalogue, rng) -> str:
    d = ImageDraw.Draw(img)
    color = TAG_COLORS[int(rng.integers(len(TAG_COLORS)))]
    d.rectangle([box.x1, box.y1, box.x2, box.y2], fill=color, outline=(120, 120, 120))
    w, h = box.w, box.h
    name = catalogue[sku].name if sku in catalogue else sku
    price_text = f"₹{price:.2f}" if rng.random() < 0.7 else f"₹ {price:g}"
    nf = font(max(6, int(h * 0.17)))
    while len(name) > 4 and nf.getlength(name) > w * 0.92:
        name = name[:-1]
    pf = fit_font(price_text, w * 0.9, h * 0.46, bold=True)
    sf = fit_font(sku, w * 0.92, h * 0.17, bold=False)
    d.text((box.cx, box.y1 + h * 0.15), name, font=nf, anchor="mm", fill=(30, 30, 30))
    d.text((box.cx, box.y1 + h * 0.52), price_text, font=pf, anchor="mm", fill=(0, 0, 0))
    d.text((box.cx, box.y1 + h * 0.87), sku, font=sf, anchor="mm", fill=(60, 60, 60))
    return f"{name} {price_text} {sku}"


def _jitter_pack(pack: Image.Image, rng: np.random.Generator) -> Image.Image:
    if rng.random() < 0.5:
        pack = pack.rotate(float(rng.uniform(-1.5, 1.5)), resample=Image.BICUBIC, expand=True)
    factor = float(rng.uniform(0.93, 1.07))
    rgb = ImageEnhance.Brightness(pack.convert("RGB")).enhance(factor)
    rgb.putalpha(pack.getchannel("A"))
    return rgb


def render_shelf(planogram: Planogram, slots: list[SlotState], catalogue: dict[str, CatalogueItem],
                 packs: PackLibrary, rng: np.random.Generator, image_id: str,
                 scale: float | None = None) -> tuple[Image.Image, ShelfMap]:
    s = scale or float(rng.uniform(0.75, 1.1))       # pixels per mm
    px = lambda mm: int(round(mm * s))  # noqa: E731

    zone_mm = [max(catalogue[sl.sku].height_mm for sl in r.slots) * HEADROOM + 15 for r in planogram.rows]
    W = px(planogram.shelf_width_mm + 2 * SIDE_MM)
    H = px(TOP_MM + sum(z + LIP_MM for z in zone_mm) + BOTTOM_MM)

    bg = _background(W, H, rng)
    products = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))   # boards and tags, drawn over products
    od = ImageDraw.Draw(overlay)
    bgd = ImageDraw.Draw(bg)

    lip_color = LIP_COLORS[int(rng.integers(len(LIP_COLORS)))]
    upright = (55, 57, 62)
    variants = {sku: packs.pick_variant(sku, rng) for sku in catalogue}
    by_pos = {(sl.row, sl.position): sl for sl in slots}
    gap_mm = planogram.facing_gap_mm

    rows_out: list[ShelfRow] = []
    y_mm = TOP_MM
    for r, zone in zip(planogram.rows, zone_mm):
        y_top, y_board = px(y_mm), px(y_mm + zone)
        y_lip = px(y_mm + zone + LIP_MM)
        # shelf surface behind the products
        bgd.rectangle([0, y_board - px(8), W, y_board], fill=tuple(int(c * 0.82) for c in bg.getpixel((W // 2, y_board))))

        used = sum(catalogue[sl.sku].width_mm * sl.facings for sl in r.slots) + gap_mm * (sum(sl.facings for sl in r.slots) + 1)
        x_mm = SIDE_MM + gap_mm + float(rng.uniform(0, (planogram.shelf_width_mm - used) * 0.5))

        items, tags, cells = [], [], []   # cells: (x1_mm, x2_mm, planned height mm, filled?)
        for i, plan in enumerate(r.slots):
            state = by_pos[(r.row, i + 1)]
            planned = catalogue[plan.sku]
            slot_x1 = x_mm
            for c in state.contents:
                cell_w = planned.width_mm
                if c is not None:
                    if c == UNKNOWN_SKU:
                        style = random_unknown_style(rng, planned.category)
                        w_mm, h_mm = planned.width_mm, planned.height_mm
                        size = float(rng.uniform(0.97, 1.03))
                        pack = packs.unknown(style, px(w_mm * size), px(h_mm * size))
                    else:
                        it = catalogue[c]
                        fit = min(1.0, cell_w * 1.08 / it.width_mm, zone * 0.95 / it.height_mm)
                        size = fit * float(rng.uniform(0.97, 1.03))
                        pack = packs.get(c, px(it.width_mm * size), px(it.height_mm * size), variants[c])
                    pack = _jitter_pack(pack, rng)
                    x = px(x_mm + cell_w / 2 + rng.uniform(-2, 2)) - pack.width // 2
                    y = y_board - pack.height + px(rng.uniform(0, 1.5))
                    products.alpha_composite(pack, (max(0, x), max(0, y)))
                    l, t, rr, b = pack.getchannel("A").point(lambda a: 255 if a > 20 else 0).getbbox()
                    items.append(ShelfItem(bbox=BBox(x1=max(0, x) + l, y1=max(0, y) + t,
                                                     x2=max(0, x) + rr, y2=max(0, y) + b), sku=c))
                cells.append((x_mm, x_mm + cell_w, planned.height_mm, c is not None))
                x_mm += cell_w + gap_mm
            # price tag at the left of the slot, on the shelf lip
            if state.tag_sku is not None:
                slot_w = x_mm - gap_mm - slot_x1
                tw = min(TAG_W_MM, slot_w - 4)
                ty = y_mm + zone + (LIP_MM - TAG_H_MM) / 2
                box = BBox(x1=px(slot_x1 + 2), y1=px(ty), x2=px(slot_x1 + 2 + tw), y2=px(ty + TAG_H_MM))
                tags.append((box, state.tag_sku, state.tag_price))

        # shelf lip with tags
        od.rectangle([px(SIDE_MM) - 2, y_board, W - px(SIDE_MM) + 2, y_lip], fill=lip_color + (255,))
        od.line([px(SIDE_MM), y_board, W - px(SIDE_MM), y_board], fill=tuple(min(255, c + 40) for c in lip_color) + (255,), width=2)
        tag_out = []
        for box, sku, price in tags:
            text = _draw_tag(overlay, box, sku, price, catalogue, rng)
            tag_out.append(PriceTag(bbox=box, sku=sku, price=price, text=text))

        # empty gaps: runs of consecutive empty cells
        gaps, run = [], []
        for cell in cells + [(0, 0, 0, True)]:
            if not cell[3]:
                run.append(cell)
            elif run:
                gaps.append(Gap(bbox=BBox(x1=px(run[0][0]), y1=y_board - px(max(c[2] for c in run)),
                                          x2=px(run[-1][1]), y2=y_board)))
                run = []

        rows_out.append(ShelfRow(row=r.row, y_top=y_top, y_bottom=y_lip,
                                 items=sorted(items, key=lambda it: it.bbox.cx), tags=tag_out, gaps=gaps))
        y_mm += zone + LIP_MM

    # uprights and base
    od.rectangle([0, 0, px(SIDE_MM), H], fill=upright + (255,))
    od.rectangle([W - px(SIDE_MM), 0, W, H], fill=upright + (255,))
    od.rectangle([0, H - px(BOTTOM_MM), W, H], fill=tuple(int(c * 0.8) for c in upright) + (255,))

    # soft contact shadows under products
    alpha = products.getchannel("A")
    shadow = Image.new("L", (W, H), 0)
    shadow.paste(alpha, (px(4), px(3)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(px(5)))
    dark = Image.new("RGB", (W, H), (0, 0, 0))
    bg = Image.composite(dark, bg, shadow.point(lambda a: int(a * 0.45)))

    img = bg.convert("RGBA")
    img.alpha_composite(products)
    img.alpha_composite(overlay)
    return img.convert("RGB"), ShelfMap(image_id=image_id, width=W, height=H, rows=rows_out)
