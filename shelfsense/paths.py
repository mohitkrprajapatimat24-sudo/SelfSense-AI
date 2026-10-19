from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ASSETS = ROOT / "assets"
FONTS = ASSETS / "fonts"
MODELS = ROOT / "models"

DATA = ROOT / "data"
CATALOGUE_CSV = DATA / "catalogue" / "catalogue.csv"
PLANOGRAMS = DATA / "planograms"
RAW_PRODUCTS = DATA / "raw"            # photos you download, one folder per SKU
PRODUCT_CUTOUTS = DATA / "products"    # background-removed PNGs made from data/raw
SYNTHETIC = DATA / "synthetic"
