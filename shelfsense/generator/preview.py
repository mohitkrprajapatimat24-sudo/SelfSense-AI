"""Browse generated shelves with their boxes and answer keys.

    streamlit run shelfsense/generator/preview.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from shelfsense.generator.build import load_answer  # noqa: E402
from shelfsense.paths import SYNTHETIC  # noqa: E402
from shelfsense.viz import draw_shelf_map  # noqa: E402

st.set_page_config(page_title="ShelfSense · synthetic data", layout="wide")

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

key = load_answer(answers[idx])
img = Image.open(split_dir / "images" / key.image_file)

st.subheader(f"{key.image_id} · planogram {key.planogram_id}")
left, right = st.columns([3, 2])
with left:
    st.image(draw_shelf_map(img, key.shelf_map, show_products, show_tags, show_gaps, show_rows, show_labels),
             width="stretch")
    with st.expander("Original image"):
        st.image(img, width="stretch")
with right:
    s = key.score
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{s.score:.1f}", s.band, delta_color="off")
    c2.metric("Placement", f"{s.placement:.0%}")
    c3.metric("Labels", f"{s.labels:.0%}")
    c4.metric("Availability", f"{s.availability:.0%}")
    st.metric("Revenue at risk / day", f"₹{key.revenue_at_risk:,.0f}")

    st.markdown("**Violations (answer key)**")
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
