import json

import pytest

from shelfsense.compliance.engine import audit
from shelfsense.data import load_catalogue, load_planograms
from shelfsense.generator.build import generate_sample
from shelfsense.generator.packs import PackLibrary
from shelfsense.perception.shelf_map import build_shelf_map, flatten
from shelfsense.schemas import Violation, ViolationType as V
from shelfsense.slm import writer as W

CAT = load_catalogue()


def v(**kw) -> Violation:
    return Violation(**kw)


@pytest.mark.parametrize("task, violation, ok", [
    ("Restock Maggi Masala Noodles 70g at Row 1 Pos 4", v(type=V.OUT_OF_STOCK, row=1, position=4, expected_sku="NDL-MAGGI-70"), True),
    ("Restock Maggi Masala Noodles 70g", v(type=V.OUT_OF_STOCK, row=1, position=4, expected_sku="NDL-MAGGI-70"), False),        # no location
    ("Restock Maggi at Row 4 Pos 1", v(type=V.OUT_OF_STOCK, row=1, position=4, expected_sku="NDL-MAGGI-70"), False),            # swapped
    ("Restock Yippee at Row 1 Pos 4", v(type=V.OUT_OF_STOCK, row=1, position=4, expected_sku="NDL-MAGGI-70"), False),           # wrong product
    ("Move Lay's Classic from R1-P3 to Row 2, Pos 6",
     v(type=V.MISPLACED, row=1, position=3, expected_sku="CHP-KURKURE-75", found_sku="CHP-LAYSCL-52", target_row=2, target_position=6), True),
    ("Move Lay's Classic from R1-P3 to Row 2",
     v(type=V.MISPLACED, row=1, position=3, expected_sku="CHP-KURKURE-75", found_sku="CHP-LAYSCL-52", target_row=2, target_position=6), False),
    ("Add 1 facing of Bingo in Row 2 Pos 2", v(type=V.EXTRA, row=2, position=2, expected_sku="CHP-BINGO-66", found_sku="UNKNOWN"), False),
    ("Remove the unknown product from Row 2 Pos 2", v(type=V.EXTRA, row=2, position=2, expected_sku="CHP-BINGO-66", found_sku="UNKNOWN"), True),
    ("Fix price tag for Parle-G at Row 3 Pos 1: ₹25", v(type=V.WRONG_PRICE, row=3, position=1, expected_sku="BIS-PARLEG-250"), True),
])
def test_task_checks(task, violation, ok):
    assert W.task_ok(task, violation, CAT) is ok


def test_tidy():
    assert W.tidy("[Top up] Maggi 70g, Row 1 Pos 1: add 2 facings, ₹150/day at risk") == \
        "Top up Maggi 70g, Row 1 Pos 1: add 2 facings"


@pytest.fixture
def result():
    planograms = load_planograms(CAT)
    _, key = generate_sample(planograms["P01"], CAT, PackLibrary(CAT), [4, 4, 4], "t")
    sm = build_shelf_map(key.image_id, key.shelf_map.width, key.shelf_map.height, *flatten(key.shelf_map))
    r = audit(sm, planograms["P01"], CAT)
    assert r.violations
    return r


def test_template_has_one_task_per_violation(result):
    tl = W.template(result, "Aisle 3, Bay 2")
    assert tl.tasks == [v.action for v in result.violations] and tl.source == "template"
    assert f"{result.score.score:.0f}/100" in tl.summary


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self.payload)}}]}


def test_writer_checks_each_line(result, monkeypatch):
    """Good lines are kept, a line with a wrong location falls back to the template."""
    writer = W.ActionWriter(CAT)
    monkeypatch.setattr(writer, "available", lambda: True)
    vs = result.violations[:W.MAX_TASKS]
    lines = [W.template(result, "x").tasks[i] for i in range(len(vs))]
    lines[0] = "Do something at Row 9 Pos 9"
    monkeypatch.setattr(W.requests, "post", lambda *a, **k: FakeResponse({"summary": "Fine.", "tasks": lines}))
    tl = writer.write(result, "Aisle 3, Bay 2")
    assert tl.tasks[0] == vs[0].action and tl.source.startswith("slm")
    assert tl.summary == "Fine."


def test_writer_falls_back_when_model_missing(result, monkeypatch):
    writer = W.ActionWriter(CAT)
    monkeypatch.setattr(writer, "available", lambda: False)
    assert writer.write(result, "Aisle 3, Bay 2").source == "template"


def test_summary_with_invented_numbers_is_replaced(result, monkeypatch):
    writer = W.ActionWriter(CAT)
    monkeypatch.setattr(writer, "available", lambda: True)
    lines = [v.action for v in result.violations[:W.MAX_TASKS]]
    monkeypatch.setattr(W.requests, "post",
                        lambda *a, **k: FakeResponse({"summary": "Losing ₹987654 a day!", "tasks": lines}))
    assert "987654" not in writer.write(result, "Aisle 3, Bay 2").summary
