import pytest

from shelfsense.data import load_catalogue, load_planograms
from shelfsense.generator.build import generate_sample, yolo_lines
from shelfsense.generator.packs import PackLibrary


@pytest.fixture(scope="module")
def ref():
    catalogue = load_catalogue()
    return catalogue, load_planograms(catalogue), PackLibrary(catalogue)


def all_boxes(key):
    for row in key.shelf_map.rows:
        for obj in row.items + row.tags + row.gaps:
            yield obj.bbox


@pytest.mark.parametrize("pid", ["P01", "P02", "P03", "P04"])
def test_shelf_contents_match_answer_key(ref, pid):
    catalogue, planograms, packs = ref
    for seed in range(5):
        img, key = generate_sample(planograms[pid], catalogue, packs, [seed, 0, 0], "t", do_augment=False)
        filled = [c for s in key.slots for c in s.contents if c is not None]
        drawn = [it.sku for row in key.shelf_map.rows for it in row.items]
        assert sorted(drawn) == sorted(filled)
        assert sum(len(r.tags) for r in key.shelf_map.rows) == len(key.slots)
        empty_cells = sum(c is None for s in key.slots for c in s.contents)
        assert (empty_cells == 0) == (sum(len(r.gaps) for r in key.shelf_map.rows) == 0)
        assert img.size == (key.shelf_map.width, key.shelf_map.height)


def test_no_mistakes_means_no_violations(ref):
    catalogue, planograms, packs = ref
    _, key = generate_sample(planograms["P01"], catalogue, packs, [1, 2, 3], "t", inject=False)
    assert key.violations == [] and key.score.score == 100.0


def test_same_seed_same_sample(ref):
    catalogue, planograms, packs = ref
    a_img, a = generate_sample(planograms["P03"], catalogue, packs, [7, 1, 42], "t")
    b_img, b = generate_sample(planograms["P03"], catalogue, packs, [7, 1, 42], "t")
    assert a.model_dump() == b.model_dump()
    assert a_img.tobytes() == b_img.tobytes()


def test_augmented_boxes_stay_inside_image(ref):
    catalogue, planograms, packs = ref
    for seed in range(10):
        _, key = generate_sample(planograms["P02"], catalogue, packs, [seed, 5, 5], "t", do_augment=True)
        W, H = key.shelf_map.width, key.shelf_map.height
        for b in all_boxes(key):
            assert 0 <= b.x1 < b.x2 <= W and 0 <= b.y1 < b.y2 <= H


def test_yolo_labels_are_normalised(ref):
    catalogue, planograms, packs = ref
    _, key = generate_sample(planograms["P01"], catalogue, packs, [3, 3, 3], "t")
    lines = yolo_lines(key.shelf_map)
    assert len(lines) == len(list(all_boxes(key)))
    for line in lines:
        cls, *vals = line.split()
        assert cls in {"0", "1", "2"}
        assert all(0 <= float(v) <= 1 for v in vals)
