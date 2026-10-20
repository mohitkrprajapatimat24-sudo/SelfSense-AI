from collections import Counter

import pytest

from shelfsense.compliance.engine import audit
from shelfsense.data import load_catalogue, load_planograms
from shelfsense.eval.compliance import key_of
from shelfsense.generator.build import generate_sample
from shelfsense.generator.packs import PackLibrary
from shelfsense.perception.shelf_map import build_shelf_map, flatten
from shelfsense.schemas import ViolationType as V


@pytest.fixture(scope="module")
def ref():
    catalogue = load_catalogue()
    return catalogue, load_planograms(catalogue), PackLibrary(catalogue)


def run(key, planogram, catalogue, items=None, tags=None, gaps=None):
    i, t, g = flatten(key.shelf_map)
    sm = build_shelf_map(key.image_id, key.shelf_map.width, key.shelf_map.height,
                         i if items is None else items, t if tags is None else tags, g if gaps is None else gaps)
    return audit(sm, planogram, catalogue)


@pytest.mark.parametrize("pid", ["P01", "P02", "P03", "P04"])
def test_matches_answer_key_with_perfect_detections(ref, pid):
    catalogue, planograms, packs = ref
    exact = 0
    for seed in range(15):
        _, key = generate_sample(planograms[pid], catalogue, packs, [seed, 8, 8], "t")
        result = run(key, planograms[pid], catalogue)
        exact += Counter(map(key_of, result.violations)) == Counter(map(key_of, key.violations))
    assert exact >= 14


def test_out_of_stock_found_without_gap_boxes(ref):
    catalogue, planograms, packs = ref
    found = total = 0
    for seed in range(20):
        _, key = generate_sample(planograms["P01"], catalogue, packs, [seed, 9, 9], "t")
        truth = {key_of(v) for v in key.violations if v.type == V.OUT_OF_STOCK}
        got = {key_of(v) for v in run(key, planograms["P01"], catalogue, gaps=[]).violations}
        found += len(truth & got)
        total += len(truth)
    assert total and found / total >= 0.95


def test_empty_shelf_with_only_tags(ref):
    catalogue, planograms, packs = ref
    p = planograms["P02"]
    _, key = generate_sample(p, catalogue, packs, [1, 1, 1], "t", inject=False)
    _, tags, _ = flatten(key.shelf_map)
    result = run(key, p, catalogue, items=[], gaps=[])
    types = Counter(v.type for v in result.violations)
    n_slots = sum(len(r.slots) for r in p.rows)
    assert types == {V.OUT_OF_STOCK: n_slots}
    assert result.score.availability == 0 and result.score.labels == 1


def test_row_missing_from_image(ref):
    catalogue, planograms, packs = ref
    p = planograms["P03"]
    _, key = generate_sample(p, catalogue, packs, [2, 2, 2], "t", inject=False)
    bottom = key.shelf_map.rows[-1]
    keep = lambda objs: [o for o in objs if o not in bottom.items + bottom.tags + bottom.gaps]  # noqa: E731
    i, t, g = flatten(key.shelf_map)
    result = run(key, p, catalogue, keep(i), keep(t), keep(g))
    last = p.rows[-1].row
    assert {v.type for v in result.violations if v.row == last} == {V.OUT_OF_STOCK, V.WRONG_LABEL}
    assert all(v.row == last for v in result.violations)
    assert any("not found" in n for n in result.notes)


def test_low_confidence_goes_to_review(ref):
    catalogue, planograms, packs = ref
    p = planograms["P01"]
    _, key = generate_sample(p, catalogue, packs, [3, 3, 3], "t", inject=False)
    items, tags, gaps = flatten(key.shelf_map)
    items[0] = items[0].model_copy(update={"sku": "CHP-BINGO-66", "match_conf": 0.2})   # uncertain wrong guess
    tags[-1] = tags[-1].model_copy(update={"price": 99.0, "ocr_conf": 0.3})             # unreadable price
    result = run(key, p, catalogue, items, tags, gaps)
    assert {v.type for v in result.violations} == {V.NEEDS_REVIEW}
    assert len(result.violations) == 2


def test_ranking_puts_stock_outs_first(ref):
    catalogue, planograms, packs = ref
    for seed in range(30):
        _, key = generate_sample(planograms["P03"], catalogue, packs, [seed, 4, 4], "t")
        result = run(key, planograms["P03"], catalogue)
        types = [v.type for v in result.violations]
        if V.OUT_OF_STOCK in types and len(set(types)) > 1:
            assert types[0] == V.OUT_OF_STOCK
            oos = [v.revenue_at_risk for v in result.violations if v.type == V.OUT_OF_STOCK]
            assert oos == sorted(oos, reverse=True)
            assert result.actions == [v.action for v in result.violations]
            return
    pytest.fail("no shelf with mixed violations generated")
