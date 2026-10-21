"""ShelfSense store dashboard.

    streamlit run shelfsense/dashboard/app.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from shelfsense.paths import ASSETS  # noqa: E402
from shelfsense.report import LABELS, build_pdf  # noqa: E402
from shelfsense.schemas import UNKNOWN_SKU  # noqa: E402
from shelfsense.service import OUTPUTS, AuditService  # noqa: E402

SAMPLES = ASSETS / "samples"
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv", "webm"]
BAND_STYLE = {   # status colours, always with an icon and a label
    "Good": ("#0ca30c", "#ffffff", "✓"),
    "Needs attention": ("#fab219", "#1a1a19", "!"),
    "Critical": ("#d03b3b", "#ffffff", "✕"),
}
SCORE_LINE = "#2a78d6"

st.set_page_config(page_title="ShelfSense AI", page_icon="🛒", layout="wide")
st.markdown("""
<style>
.ss-hero {display:flex; align-items:center; gap:18px; padding:14px 18px; border:1px solid rgba(128,128,128,.3);
          border-radius:10px; margin-bottom:10px}
.ss-score {font-size:64px; font-weight:700; line-height:1; font-variant-numeric:tabular-nums}
.ss-chip {display:inline-block; padding:3px 10px; border-radius:999px; font-weight:600; font-size:14px}
.ss-sub {opacity:.7; font-size:13px; margin-top:6px}
.ss-act {padding:6px 0; border-bottom:1px solid rgba(128,128,128,.2); font-size:15px}
.ss-act .n {display:inline-block; width:1.6em; font-weight:700}
.ss-act .r {opacity:.7; font-size:13px; white-space:nowrap}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ShelfSense models ...")
def get_service() -> AuditService:
    svc = AuditService(folder=Path(os.environ.get("SHELFSENSE_OUTPUTS", OUTPUTS)),
                       threads=int(os.environ.get("SHELFSENSE_THREADS", "4")))
    svc.perception
    return svc


@st.cache_data(show_spinner=False)
def pdf_bytes(audit_id: str, tasks_source: str) -> bytes:     # tasks_source: rebuild once Qwen has written
    return build_pdf(get_service().history.get(audit_id))


def samples() -> list[dict]:
    f = SAMPLES / "samples.json"
    return json.loads(f.read_text()) if f.exists() else []


def name_of(svc: AuditService, sku: str | None) -> str:
    if sku is None:
        return "–"
    if sku == UNKNOWN_SKU:
        return "Unknown product"
    return svc.catalogue[sku].name if sku in svc.catalogue else sku


svc = get_service()
planograms = svc.planograms

# ---------------------------------------------------------------- sidebar: what to audit
with st.sidebar:
    st.title("ShelfSense AI")
    st.caption("One shelf photo in, a scored audit out. Runs offline on CPU.")
    store = st.text_input("Store", "Store 214")
    source = st.radio("Shelf image", ["Sample shelf", "Upload photo or video"], horizontal=False)

    upload, sample = None, None
    if source == "Sample shelf" and samples():
        options = samples()
        sample = st.selectbox(
            "Sample", options,
            format_func=lambda s: f"{planograms[s['planogram_id']].bay_id} · {s['file']}"
                                  + (" (video)" if s.get("video") else ""))
        planogram_id = sample["planogram_id"]
        st.caption(f"Planogram {planogram_id}: {planograms[planogram_id].description}")
        if not sample.get("video"):
            st.image(str(SAMPLES / sample["file"]), width="stretch")
    else:
        planogram_id = st.selectbox(
            "Bay", list(planograms),
            format_func=lambda p: f"{planograms[p].bay_id} · {planograms[p].description} ({p})")
        upload = st.file_uploader("Photo or short video of one bay", type=UPLOAD_TYPES)

    run = st.button("Run audit", type="primary", width="stretch",
                    disabled=source != "Sample shelf" and upload is None)

    recent = svc.history.list(limit=15)
    if recent:
        st.divider()
        pick = st.selectbox("Open a previous audit", [None] + [r["audit_id"] for r in recent],
                            format_func=lambda a: "–" if a is None else next(
                                f"{datetime.fromisoformat(r['created_at']):%d %b %H:%M} · {r['bay_id']} · {r['score']:.0f}"
                                for r in recent if r["audit_id"] == a))
        if pick:
            st.session_state["audit_id"] = pick

if run:
    with st.spinner("Auditing the shelf ..."):
        try:
            if upload is not None:
                rec = svc.audit_bytes(upload.getvalue(), upload.name, planogram_id, store)
            else:
                rec = svc.audit_file(SAMPLES / sample["file"], planogram_id, store)
            st.session_state["audit_id"] = rec.audit_id
            st.session_state["write_tasks"] = rec.audit_id if rec.result.violations else None
        except Exception as e:      # unreadable upload, etc.
            st.error(f"Could not audit that file: {e}")

rec = svc.history.get(st.session_state["audit_id"]) if "audit_id" in st.session_state else None

# ---------------------------------------------------------------- nothing yet
if rec is None:
    st.header("Shelf compliance audit")
    st.markdown(
        "Pick a **sample shelf** or **upload a photo or short video** of one bay, then press **Run audit**.\n\n"
        "ShelfSense finds every product, price tag and empty gap, checks them against the planogram and price "
        "list, and gives you a 0-100 compliance score with a ranked list of fixes.")
    st.stop()

# ---------------------------------------------------------------- audit result
r = rec.result
when = datetime.fromisoformat(rec.created_at)
st.subheader(f"{rec.store_id} · {rec.bay_id} · Audit {when:%d %b, %H:%M}")
if rec.blurry:
    st.warning("This photo looks blurry. Results may be less reliable; consider retaking it.")

left, right = st.columns([3, 2], gap="large")
with left:
    st.image(rec.annotated_path, width="stretch")
    st.caption("Green: compliant · amber: low stock · red: violation · purple: check by hand")

with right:
    bg, fg, icon = BAND_STYLE[r.score.band]
    st.markdown(f"""
<div class="ss-hero">
  <div class="ss-score">{r.score.score:.0f}</div>
  <div><span class="ss-chip" style="background:{bg};color:{fg}">{icon}&nbsp; {r.score.band}</span>
       <div class="ss-sub">compliance score out of 100 · placement {r.score.placement:.0%} ·
       labels {r.score.labels:.0%} · availability {r.score.availability:.0%}</div>
       {f'<div class="ss-sub">based on {r.score.verified:.0%} of checks; the rest need a check by hand</div>'
        if r.score.verified < 1 else ''}</div>
</div>""", unsafe_allow_html=True)

    c = rec.counts()
    t1, t2, t3 = st.columns(3)
    t1.metric("Out of stock", c["out_of_stock"])
    t2.metric("Misplaced", c["misplaced"])
    t3.metric("Low stock", c["low_stock"])
    t4, t5, t6 = st.columns(3)
    t4.metric("Price / label", c["price_label"])
    t5.metric("Extra items", c["extra"])
    t6.metric("Check by hand", c["review"])
    st.metric("Revenue at risk", f"₹{r.revenue_at_risk:,.0f} / day")

    actions_box = st.container()
    if st.session_state.get("write_tasks") == rec.audit_id and svc.use_slm:
        with st.spinner("Qwen2.5-3B is writing the task list (on CPU, offline) ..."):
            rec = svc.write_tasks(rec)
        st.session_state["write_tasks"] = None

    with actions_box:
        source = {"slm": "written by Qwen2.5-3B, checked against the audit",
                  "slm+template": "written by Qwen2.5-3B; some lines kept from the template after checking",
                  "template": "template"}.get(rec.tasks_source, "")
        st.markdown(f"**Recommended actions, most urgent first** <span class='r'>· {source}</span>",
                    unsafe_allow_html=True)
        if rec.summary:
            st.write(rec.summary)
        tasks = rec.tasks or [v.action for v in r.violations]
        if tasks:
            rows = []
            for i, (task, v) in enumerate(zip(tasks[:8], r.violations), 1):
                risk = f" <span class='r'>· ₹{v.revenue_at_risk:,.0f}/day at risk</span>" if v.revenue_at_risk else ""
                rows.append(f"<div class='ss-act'><span class='n'>{i}.</span>{task}{risk}</div>")
            st.markdown("".join(rows), unsafe_allow_html=True)
            if len(tasks) > 8:
                st.caption(f"+ {len(tasks) - 8} more in the Violations tab and the PDF")
        else:
            st.success("Nothing to do: the shelf matches the planogram.")

    st.download_button("Download PDF report", pdf_bytes(rec.audit_id, rec.tasks_source),
                       file_name=f"shelfsense_{rec.audit_id}.pdf",
                       mime="application/pdf", width="stretch")

# ---------------------------------------------------------------- details
tab_v, tab_s, tab_h, tab_d = st.tabs(["Violations", "Every slot", "Score history", "How it was read"])

with tab_v:
    if r.violations:
        st.dataframe(pd.DataFrame([{
            "#": i, "Issue": LABELS[v.type], "Row": v.row, "Pos": v.position,
            "Planned": name_of(svc, v.expected_sku), "Found": name_of(svc, v.found_sku),
            "Facings": f"{v.found_facings}/{v.expected_facings}" if v.expected_facings else "",
            "Tag price": f"₹{v.found_price:g}" if v.found_price is not None else "",
            "Catalogue": f"₹{v.expected_price:g}" if v.expected_price is not None else "",
            "₹/day at risk": round(v.revenue_at_risk) if v.revenue_at_risk else None,
            "Action": v.action} for i, v in enumerate(r.violations, 1)]), hide_index=True, width="stretch")
    else:
        st.info("No violations.")

with tab_s:
    issues = {}
    for v in r.violations:
        issues.setdefault((v.row, v.position), []).append(LABELS[v.type])
    st.dataframe(pd.DataFrame([{
        "Row": s.row, "Pos": s.position, "Planned": name_of(svc, s.planned_sku),
        "Facings found": f"{sum(c == s.planned_sku for c in s.contents)}/{s.planned_facings}",
        "Other products": ", ".join(name_of(svc, c) for c in s.contents if c not in (None, s.planned_sku)) or "",
        "Tag": f"{name_of(svc, s.tag_sku)} ₹{s.tag_price:g}" if s.tag_price is not None else name_of(svc, s.tag_sku),
        "Status": ", ".join(issues.get((s.row, s.position), [])) or "OK"} for s in r.slots]),
        hide_index=True, width="stretch")

with tab_h:
    hist = pd.DataFrame(svc.history.list(bay_id=rec.bay_id, limit=200))
    hist["when"] = pd.to_datetime(hist["created_at"])
    st.markdown(f"**Compliance score over time · {rec.bay_id}**")
    if len(hist) < 2:
        st.caption("Run more audits of this bay to see the trend.")
    base = alt.Chart(hist).encode(x=alt.X("when:T", title=None))
    bands = alt.Chart(pd.DataFrame({"y": [70, 90]})).mark_rule(
        strokeDash=[4, 4], color="#9a9890", strokeWidth=1).encode(y="y:Q")
    band_text = alt.Chart(pd.DataFrame({"y": [70, 90], "label": ["70 · critical below", "90 · good"]})).mark_text(
        align="left", dx=4, dy=-6, color="#7a7974", fontSize=11).encode(y="y:Q", text="label:N", x=alt.value(0))
    line = base.mark_line(color=SCORE_LINE, strokeWidth=2).encode(
        y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]), title="Score"))
    points = base.mark_circle(color=SCORE_LINE, size=70, opacity=1).encode(
        y="score:Q", tooltip=[alt.Tooltip("when:T", title="When", format="%d %b %H:%M"),
                              alt.Tooltip("score:Q", title="Score", format=".1f"),
                              alt.Tooltip("band:N", title="Band"),
                              alt.Tooltip("revenue_at_risk:Q", title="₹/day at risk", format=",.0f"),
                              alt.Tooltip("audit_id:N", title="Audit")])
    st.altair_chart((bands + band_text + line + points).properties(height=260), width="stretch")
    st.dataframe(hist[["created_at", "store_id", "score", "band", "violations", "revenue_at_risk", "audit_id"]]
                 .rename(columns={"created_at": "When", "store_id": "Store", "score": "Score", "band": "Band",
                                  "violations": "Violations", "revenue_at_risk": "₹/day at risk",
                                  "audit_id": "Audit"}), hide_index=True, width="stretch")

with tab_d:
    t = rec.timings_ms
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total time", f"{(t.get('total', 0) + t.get('audit', 0)) / 1000:.2f} s")
    d2.metric("Detect", f"{t.get('detect', 0):.0f} ms")
    d3.metric("Identify products", f"{t.get('identify', 0):.0f} ms")
    d4.metric("Read price tags", f"{t.get('ocr', 0):.0f} ms")
    if rec.tasks_ms:
        st.caption(f"Task list written by Qwen2.5-3B in {rec.tasks_ms / 1000:.1f} s (separate from the audit time).")
    st.caption(f"Offline on CPU · photo sharpness {rec.sharpness:.0f} · source {rec.source} · audit {rec.audit_id}")
    if rec.frame_info:
        st.info(f"Video: frame {rec.frame_info['frame']} of {rec.frame_info['frames']} was the sharpest and was audited.")
    for note in r.notes:
        st.warning(note)
    with st.expander("Original photo"):
        st.image(rec.image_path, width="stretch")
