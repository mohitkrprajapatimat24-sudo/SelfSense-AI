import cv2
import numpy as np
import pytest

from shelfsense.data import load_catalogue
from shelfsense.perception.ocr import TagReader
from shelfsense.perception.preprocess import load_upload


@pytest.fixture(scope="module")
def reader():
    r = TagReader.__new__(TagReader)          # parsing only, no OCR model needed
    r.catalogue = load_catalogue()
    return r


@pytest.mark.parametrize("lines, sku, price", [
    ([("Maggi Masala Noodles", 0.99, 20), ("₹15.00", 0.98, 40), ("NDL-MAGGI-70", 0.99, 14)], "NDL-MAGGI-70", 15.0),
    ([("Yippee Magic Masala N", 0.99, 20), ("₹ 15", 0.9, 40)], "NDL-YIPPEE-70", 15.0),          # no code: name match
    ([("Parle-G Glucose", 0.97, 20), ("Rs. 25.00", 0.95, 38)], "BIS-PARLEG-250", 25.0),
    ([("BIS - OREO - 120", 0.9, 14), ("₹35,00", 0.9, 40)], "BIS-OREO-120", 35.0),            # spaced code, comma
    ([("Oreo Original Cream", 0.98, 20), ("€35.00", 0.97, 40)], "BIS-OREO-120", 35.0),       # ₹ read as €
    ([("Oreo Original Cream", 0.98, 20), ("35.00", 0.97, 40), ("70g", 0.9, 12)], "BIS-OREO-120", 35.0),  # ₹ dropped
])
def test_parse_tag(reader, lines, sku, price):
    read = reader.parse(lines)
    assert (read.sku, read.price) == (sku, price)
    assert read.conf > 0.8


def test_price_without_currency_mark_is_less_certain(reader):
    # "₹25.00" misread as "125.00" looks like a bare number: read it, but send it to review
    read = reader.parse([("Parle-G Glucose", 0.97, 20), ("125.00", 0.95, 40)])
    assert read.price == 125.0 and read.conf < 0.6


def test_unreadable_tag_has_zero_confidence(reader):
    read = reader.parse([("~~", 0.3, 10)])
    assert read.sku is None and read.price is None and read.conf == 0


def test_video_gives_sharpest_frame(tmp_path):
    path = tmp_path / "shelf.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    rng = np.random.default_rng(0)
    sharp = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    for i in range(30):
        frame = sharp if 16 <= i <= 18 else cv2.GaussianBlur(sharp, (15, 15), 0)   # a moment of steady hands
        out.write(frame)
    out.release()
    img, info = load_upload(path)
    assert img.size == (320, 240)
    assert abs(info["frame"] - 17) <= 2      # sampled a few frames per second
