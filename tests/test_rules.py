import pytest

from shelfsense.compliance.rules import compute_score, evaluate_slots
from shelfsense.data import load_catalogue, load_planograms
from shelfsense.generator.mistakes import perfect_state
from shelfsense.schemas import UNKNOWN_SKU, ViolationType as V


@pytest.fixture(scope="module")
def ref():
    catalogue = load_catalogue()
    return catalogue, load_planograms(catalogue)["P01"]


def slot(slots, row, pos):
    return next(s for s in slots if s.row == row and s.position == pos)


def types(violations):
    return sorted(v.type for v in violations)


def test_perfect_shelf_scores_100(ref):
    catalogue, p = ref
    slots = perfect_state(p, catalogue)
    assert evaluate_slots(slots, p, catalogue) == []
    s = compute_score(slots, catalogue)
    assert (s.score, s.band) == (100.0, "Good")


def test_out_of_stock(ref):
    catalogue, p = ref
    slots = perfect_state(p, catalogue)
    slot(slots, 1, 1).contents = [None, None, None]          # Maggi x3
    [v] = evaluate_slots(slots, p, catalogue)
    assert v.type == V.OUT_OF_STOCK and v.count == 3
    assert v.revenue_at_risk == 40 * 15                       # velocity x price
    s = compute_score(slots, catalogue)
    assert s.placement == round(9 / 10, 4)
    assert s.availability == round(26 / 29, 4)


def test_low_stock(ref):
    catalogue, p = ref
    slots = perfect_state(p, catalogue)
    slot(slots, 2, 1).contents = ["BIS-PARLEG-250", None, None, "BIS-PARLEG-250"]
    [v] = evaluate_slots(slots, p, catalogue)
    assert (v.type, v.found_facings, v.count) == (V.LOW_STOCK, 2, 2)


def test_misplaced_points_to_home_slot(ref):
    catalogue, p = ref
    slots = perfect_state(p, catalogue)
    slot(slots, 2, 3).contents = ["BIS-OREO-120", "NDL-KNORR-70", "BIS-OREO-120"]
    got = evaluate_slots(slots, p, catalogue)
    assert types(got) == [V.LOW_STOCK, V.MISPLACED]
    m = next(v for v in got if v.type == V.MISPLACED)
    assert (m.found_sku, m.target_row, m.target_position) == ("NDL-KNORR-70", 1, 3)


def test_fully_replaced_slot_is_misplaced_and_out_of_stock(ref):
    catalogue, p = ref
    slots = perfect_state(p, catalogue)
    slot(slots, 1, 3).contents = ["NDL-MAGGI-70", "NDL-MAGGI-70"]
    assert types(evaluate_slots(slots, p, catalogue)) == [V.MISPLACED, V.OUT_OF_STOCK]


def test_unknown_and_stray_products_are_extra(ref):
    catalogue, p = ref
    slots = perfect_state(p, catalogue)
    slot(slots, 1, 1).contents = ["NDL-MAGGI-70", UNKNOWN_SKU, "NDL-MAGGI-70"]
    slot(slots, 3, 1).contents = ["NDL-MAGGI-70", "NDL-MAGGI-70", "NDL-MAGGI-70", "CHP-BINGO-66"]
    got = [v for v in evaluate_slots(slots, p, catalogue) if v.type == V.EXTRA]
    assert {v.found_sku for v in got} == {UNKNOWN_SKU, "CHP-BINGO-66"}


def test_tag_checks(ref):
    catalogue, p = ref
    slots = perfect_state(p, catalogue)
    slot(slots, 1, 1).tag_price = 20                          # Maggi is 15
    slot(slots, 1, 2).tag_sku, slot(slots, 1, 2).tag_price = "NDL-KNORR-70", 20
    slot(slots, 1, 3).tag_sku, slot(slots, 1, 3).tag_price = None, None
    got = {(v.row, v.position): v.type for v in evaluate_slots(slots, p, catalogue)}
    assert got == {(1, 1): V.WRONG_PRICE, (1, 2): V.WRONG_LABEL, (1, 3): V.WRONG_LABEL}
    assert compute_score(slots, catalogue).labels == round(7 / 10, 4)


def test_bands(ref):
    catalogue, p = ref
    slots = perfect_state(p, catalogue)
    for s in slots[:5]:
        s.contents = [None] * s.planned_facings
    s = compute_score(slots, catalogue)
    assert s.score < 70 and s.band == "Critical"
