"""Browse generated shelves, their answer keys, and what the compliance engine makes of them.

    streamlit run shelfsense/generator/preview.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from shelfsense.compliance.engine import audit  # noqa: E402
from shelfsense.data import load_catalogue, load_planograms  # noqa: E402
from shelfsense.eval.compliance import key_of  # noqa: E402
from shelfsense.generator.build import load_answer  # noqa: E402
from shelfsense.paths import SYNTHETIC  # noqa: E402
from shelfsense.perception.shelf_map import build_shelf_map, flatten  # noqa: E402
from shelfsense.viz import draw_audit, draw_shelf_map  # noqa: E402

st.set_page_config(page_title="ShelfSense · synthetic data", layout="wide")


@st.cache_resource
def reference():
    catalogue = load_catalogue()
    return catalogue, load_planograms(catalogue)


splits = sorted(p.name for p in SYNTHETIC.glob("*") if (p / "answers").exists()) if SYNTHETIC.exists() else []
if not splits:
    st.warning("No synthetic data yet. Run: python -m shelfsense.generator --split preview --n 24")
    st.stop()

with st.sidebar:
    split = st.selectbox("Split", splits)
    split_dir = SYNTHETIC / split
    answers = sorted((split_dir / "answers").glob("*.json"))
    only_bad = st.checkbox("Only shelves with violations")
    if only_bad:
        answers = [a for a in answers if json.loads(a.read_text(encoding="utf-8"))["violations"]]
    if not answers:
        st.info("Nothing to show.")
        st.stop()
    idx = st.number_input(f"Image (0-{len(answers) - 1})", 0, len(answers) - 1, 0)
    st.divider()
    show_products = st.checkbox("Products", True)
    show_tags = st.checkbox("Price tags", True)
    show_gaps = st.checkbox("Empty gaps", True)
    show_rows = st.checkbox("Rows", False)
    show_labels = st.checkbox("Labels", True)
    summary_file = split_dir / "summary.json"
    if summary_file.exists():
        st.divider()
        st.caption("Split summary")
        st.json(json.loads(summary_file.read_text()), expanded=False)

catalogue, planograms = reference()
key = load_answer(answers[idx])
img = Image.open(split_dir / "images" / key.image_file)

# the engine sees the boxes without their row structure, as a detector would produce them
shelf_map = build_shelf_map(key.image_id, key.shelf_map.width, key.shelf_map.height, *flatten(key.shelf_map))
result = audit(shelf_map, planograms[key.planogram_id], catalogue)


def score_metrics(s, revenue):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score", f"{s.score:.1f}", s.band, delta_color="off")
    c2.metric("Placement", f"{s.placement:.0%}")
    c3.metric("Labels", f"{s.labels:.0%}")
    c4.metric("Availability", f"{s.availability:.0%}")
    c5.metric("At risk / day", f"₹{revenue:,.0f}")


st.subheader(f"{key.image_id} · planogram {key.planogram_id}")
truth_tab, engine_tab = st.tabs(["Answer key", "Engine vs answer key"])

with truth_tab:
    left, right = st.columns([3, 2])
    with left:
        st.image(draw_shelf_map(img, key.shelf_map, show_products, show_tags, show_gaps, show_rows, show_labels),
                 width="stretch")
        with st.expander("Original image"):
            st.image(img, width="stretch")
    with right:
        score_metrics(key.score, key.revenue_at_risk)
        st.markdown("**Violations**")
        if key.violations:
            st.dataframe(pd.DataFrame([{"type": v.type.value, "row": v.row, "pos": v.position, "action": v.action}
                                       for v in key.violations]), hide_index=True, width="stretch")
        else:
            st.success("Fully compliant shelf")
        st.markdown("**Mistakes injected**")
        st.dataframe(pd.DataFrame([m.model_dump() for m in key.mistakes]) if key.mistakes else pd.DataFrame(),
                     hide_index=True, width="stretch")
        with st.expander("Slot ground truth"):
            st.dataframe(pd.DataFrame([{"row": sl.row, "pos": sl.position, "planned": sl.planned_sku,
                                        "facings": sl.planned_facings,
                                        "contents": ", ".join(c or "-" for c in sl.contents),
                                        "tag": f"{sl.tag_sku} {sl.tag_price:g}"} for sl in key.slots]),
                         hide_index=True, width="stretch")

with engine_tab:
    truth = Counter(map(key_of, key.violations))
    found = Counter(map(key_of, result.violations))
    if truth == found:
        st.success("Engine found exactly the answer-key violations")
    else:
        st.error(f"Engine missed {sum((truth - found).values())} and added {sum((found - truth).values())} violation(s)")
    left, right = st.columns([3, 2])
    with left:
        st.image(draw_audit(img, result), width="stretch")
        st.caption("Green: compliant · amber: low stock · red: violation · purple: needs human review")
    with right:
        score_metrics(result.score, result.revenue_at_risk)
        st.caption(f"Answer key score: {key.score.score:.1f}")
        st.markdown("**Ranked actions**")
        rows = []
        for v in result.violations:
            k = key_of(v)
            rows.append({"": "✓" if truth[k] else "✗ extra", "type": v.type.value, "action": v.action})
        for k in (truth - found).elements():
            rows.append({"": "✗ missed", "type": k[0], "action": f"Row {k[1]}, Pos {k[2]}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.success("Nothing to do: shelf is compliant")
        for note in result.notes:
            st.warning(note)
