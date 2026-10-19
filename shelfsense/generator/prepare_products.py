"""Cut product photos out of their backgrounds for the generator.

Put 3-5 front-facing photos per product in data/raw/<SKU>/ (jpg, png or webp),
then run:

    python -m shelfsense.generator.prepare_products

Cutouts are written to data/products/<SKU>/<name>.png. Existing cutouts are kept,
so it is safe to rerun after adding more photos.
"""

import os

from ..data import load_catalogue
from ..paths import MODELS, PRODUCT_CUTOUTS, RAW_PRODUCTS

os.environ.setdefault("U2NET_HOME", str(MODELS / "rembg"))

from PIL import Image  # noqa: E402
from rembg import new_session, remove  # noqa: E402

PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
ALPHA_THRESHOLD = 20


def cut_out(path, session) -> Image.Image:
    img = Image.open(path).convert("RGB")
    out = remove(img, session=session)
    bbox = out.getchannel("A").point(lambda a: 255 if a > ALPHA_THRESHOLD else 0).getbbox()
    return out.crop(bbox) if bbox else out


def main():
    catalogue = load_catalogue()
    for sku in catalogue:  # one folder per SKU, ready for photos
        (RAW_PRODUCTS / sku).mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in RAW_PRODUCTS.iterdir() if p.is_dir()):
        if folder.name not in catalogue:
            print(f"Skipping {folder}: not a SKU in the catalogue")

    session = None
    made = 0
    for sku in catalogue:
        photos = sorted(p for p in (RAW_PRODUCTS / sku).glob("*") if p.suffix.lower() in PHOTO_TYPES)
        out_dir = PRODUCT_CUTOUTS / sku
        for photo in photos:
            target = out_dir / f"{photo.stem}.png"
            if target.exists():
                continue
            session = session or new_session("u2net")
            out_dir.mkdir(parents=True, exist_ok=True)
            cut_out(photo, session).save(target)
            made += 1
            print(f"  {sku}/{target.name}")

    print(f"\n{made} new cutouts in {PRODUCT_CUTOUTS}")
    print("\nSKU                 photos  cutouts")
    for sku in catalogue:
        n_raw = len([p for p in (RAW_PRODUCTS / sku).glob("*") if p.suffix.lower() in PHOTO_TYPES])
        n_cut = len(list((PRODUCT_CUTOUTS / sku).glob("*.png")))
        flag = "" if n_cut >= 3 else "  <- needs at least 3"
        print(f"{sku:<20}{n_raw:>6}{n_cut:>9}{flag}")


if __name__ == "__main__":
    main()
