"""Product pack images for the synthetic shelves.

If background-removed photos exist in data/products/<sku>/*.png they are used.
Otherwise a placeholder pack is drawn from the catalogue colours, so the whole
pipeline can run before any photos are collected.
"""

import colorsys
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..paths import FONTS, PRODUCT_CUTOUTS
from ..schemas import CatalogueItem

FAKE_BRANDS = ["CRUNCHO", "SNACKZ", "NOODLA", "BISKO", "CHIPPY", "MUNCHO", "TASTEE", "KRISPA"]
CATEGORIES = ["noodles", "biscuits", "chips"]


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"
    return ImageFont.truetype(str(FONTS / name), max(6, int(size)))


def fit_font(text: str, max_w: float, max_h: float, bold: bool = True) -> ImageFont.FreeTypeFont:
    size = max(6, int(max_h))
    while size > 6:
        f = font(size, bold)
        l, t, r, b = f.getbbox(text)
        if r - l <= max_w and b - t <= max_h:
            return f
        size -= 1
    return font(6, bold)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def shade(rgb, factor: float):
    return tuple(int(max(0, min(255, c * factor))) for c in rgb)


def contrast_text(rgb) -> tuple[int, int, int]:
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (20, 20, 20) if luminance > 150 else (250, 250, 250)


def _variant_and_weight(item_name: str, brand: str) -> tuple[str, str]:
    m = re.search(r"(\d+\s?g)$", item_name)
    weight = m.group(1) if m else ""
    variant = item_name.replace(brand, "").replace(weight, "").strip(" -")
    return variant, weight


def _crimp(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, horizontal: bool, color):
    """Zig-zag sealed edge of a pillow pack."""
    teeth = 10
    if horizontal:
        step = (x2 - x1) / teeth
        pts = [(x1 + i * step, y1 if i % 2 == 0 else y2) for i in range(teeth + 1)]
    else:
        step = (y2 - y1) / teeth
        pts = [(x1 if i % 2 == 0 else x2, y1 + i * step) for i in range(teeth + 1)]
    draw.line(pts, fill=color, width=1)


def draw_placeholder(brand: str, variant: str, weight: str, category: str,
                     color: tuple, accent: tuple, w: int, h: int) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    text_rgb = contrast_text(color)
    accent_text = contrast_text(accent)

    if category == "chips":
        crimp = max(3, int(h * 0.06))
        d.rounded_rectangle([0, crimp, w - 1, h - crimp - 1], radius=int(w * 0.12), fill=color)
        d.rectangle([int(w * 0.04), 0, int(w * 0.96), crimp], fill=shade(color, 0.85))
        d.rectangle([int(w * 0.04), h - crimp - 1, int(w * 0.96), h - 1], fill=shade(color, 0.85))
        _crimp(d, int(w * 0.04), 0, int(w * 0.96), crimp, True, shade(color, 0.6))
        _crimp(d, int(w * 0.04), h - crimp - 1, int(w * 0.96), h - 1, True, shade(color, 0.6))
        # chip "photo" and a soft vertical highlight
        cx, cy, r = w * 0.5, h * 0.62, w * 0.28
        d.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8], fill=accent)
        d.ellipse([cx - r * 0.55, cy - r * 0.45, cx + r * 0.55, cy + r * 0.45], fill=shade(accent, 1.25))
        hl = Image.new("L", (w, h), 0)
        ImageDraw.Draw(hl).rectangle([int(w * 0.15), crimp, int(w * 0.25), h - crimp], fill=45)
        img.paste((255, 255, 255, 255), (0, 0), Image.composite(hl, Image.new("L", (w, h), 0), img.getchannel("A")))
        brand_box = (w * 0.86, h * 0.18, h * 0.12)
        variant_y, weight_y = h * 0.34, h * 0.86
    elif category == "noodles":
        crimp = max(3, int(w * 0.05))
        d.rectangle([crimp, 0, w - crimp - 1, h - 1], fill=color)
        d.rectangle([0, int(h * 0.04), crimp, int(h * 0.96)], fill=shade(color, 0.85))
        d.rectangle([w - crimp - 1, int(h * 0.04), w - 1, int(h * 0.96)], fill=shade(color, 0.85))
        _crimp(d, 0, int(h * 0.04), crimp, int(h * 0.96), False, shade(color, 0.6))
        _crimp(d, w - crimp - 1, int(h * 0.04), w - 1, int(h * 0.96), False, shade(color, 0.6))
        d.rectangle([crimp, int(h * 0.62), w - crimp - 1, h - 1], fill=accent)
        # bowl of noodles
        bx, by, br = w * 0.72, h * 0.62, w * 0.17
        d.pieslice([bx - br, by - br, bx + br, by + br], 0, 180, fill=(245, 245, 240))
        d.ellipse([bx - br, by - br * 0.35, bx + br, by + br * 0.35], fill=(236, 190, 90))
        brand_box = (w * 0.55, h * 0.24, h * 0.24)
        variant_y, weight_y = h * 0.47, h * 0.84
    else:  # biscuits: upright box
        d.rectangle([0, 0, w - 1, h - 1], fill=color, outline=shade(color, 0.7))
        d.rectangle([0, 0, w - 1, int(h * 0.2)], fill=accent)
        for i in range(3):  # stack of biscuits
            y = h * (0.56 + i * 0.09)
            d.ellipse([w * 0.18, y, w * 0.82, y + h * 0.07], fill=(201, 145, 72), outline=(150, 100, 45))
        brand_box = (w * 0.9, h * 0.1, h * 0.1)
        variant_y, weight_y = h * 0.32, h * 0.92

    max_w, cy, max_h = brand_box
    bf = fit_font(brand.upper(), max_w, max_h, bold=True)
    on_accent = category == "biscuits"
    d.text((w / 2, cy), brand.upper(), font=bf, anchor="mm", fill=accent_text if on_accent else text_rgb)
    if variant:
        vf = fit_font(variant, w * 0.86, h * 0.07, bold=False)
        d.text((w / 2, variant_y), variant, font=vf, anchor="mm", fill=text_rgb)
    if weight:
        wf = fit_font(weight, w * 0.4, h * 0.07, bold=True)
        fill = accent_text if category == "noodles" else text_rgb
        d.text((w * 0.22, weight_y), weight, font=wf, anchor="mm", fill=fill)
    return img


def random_unknown_style(rng: np.random.Generator, category: str) -> dict:
    hue = rng.random()
    color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.55 + 0.4 * rng.random(), 0.6 + 0.4 * rng.random()))
    accent = tuple(int(c * 255) for c in colorsys.hsv_to_rgb((hue + 0.5) % 1, 0.7, 0.9))
    return {"brand": str(rng.choice(FAKE_BRANDS)), "variant": "New Flavour", "weight": f"{int(rng.integers(40, 150))}g",
            "category": category, "color": color, "accent": accent}


class PackLibrary:
    def __init__(self, catalogue: dict[str, CatalogueItem], cutouts_dir: Path = PRODUCT_CUTOUTS):
        self.catalogue = catalogue
        self.cutouts: dict[str, list[Path]] = {}
        if cutouts_dir.exists():
            for sku in catalogue:
                files = sorted((cutouts_dir / sku).glob("*.png"))
                if files:
                    self.cutouts[sku] = files

    def has_photos(self, sku: str) -> bool:
        return sku in self.cutouts

    def pick_variant(self, sku: str, rng: np.random.Generator) -> int:
        """Which photo to use for this SKU in one image (all its facings look the same)."""
        return int(rng.integers(len(self.cutouts[sku]))) if sku in self.cutouts else 0

    def get(self, sku: str, w: int, h: int, variant: int = 0) -> Image.Image:
        """Pack image that fits inside w x h (keeping aspect for photos)."""
        if sku in self.cutouts:
            return _load_cutout(str(self.cutouts[sku][variant % len(self.cutouts[sku])]), w, h)
        item = self.catalogue[sku]
        v, wt = _variant_and_weight(item.name, item.brand)
        return _placeholder(item.brand, v, wt, item.category, item.color, item.accent, w, h)

    @staticmethod
    def unknown(style: dict, w: int, h: int) -> Image.Image:
        return draw_placeholder(style["brand"], style["variant"], style["weight"], style["category"],
                                style["color"], style["accent"], w, h)


@lru_cache(maxsize=512)
def _placeholder(brand, variant, weight, category, color, accent, w, h) -> Image.Image:
    return draw_placeholder(brand, variant, weight, category, hex_rgb(color), hex_rgb(accent), w, h)


@lru_cache(maxsize=256)
def _load_cutout(path: str, w: int, h: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    s = min(w / img.width, h / img.height)
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
