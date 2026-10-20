"""One-page (or two) PDF audit report for store staff."""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from .paths import FONTS
from .schemas import ViolationType as V

pdfmetrics.registerFont(TTFont("Noto", str(FONTS / "NotoSans-Regular.ttf")))      # has the ₹ glyph
pdfmetrics.registerFont(TTFont("Noto-Bold", str(FONTS / "NotoSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("NotoSym", str(FONTS / "NotoSansSymbols2-Regular.ttf")))   # ✓ ✕

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d3")
HEADER = colors.HexColor("#0f3a52")
BAND = {"Good": ("#0ca30c", "✓"), "Needs attention": ("#fab219", "!"), "Critical": ("#d03b3b", "✕")}

LABELS = {
    V.OUT_OF_STOCK: "Out of stock", V.LOW_STOCK: "Low stock", V.MISPLACED: "Misplaced", V.EXTRA: "Extra item",
    V.WRONG_PRICE: "Wrong price", V.WRONG_LABEL: "Wrong label", V.NEEDS_REVIEW: "Check by hand",
}


def _style(name, size=9.5, font="Noto", color=INK, leading=None, **kw):
    return ParagraphStyle(name, fontName=font, fontSize=size, textColor=color, leading=leading or size * 1.35,
                          alignment=TA_LEFT, **kw)


S = {
    "title": _style("title", 18, "Noto-Bold", colors.white),
    "sub": _style("sub", 9.5, color=colors.HexColor("#cfe3ee")),
    "h": _style("h", 12, "Noto-Bold", spaceBefore=8, spaceAfter=4),
    "body": _style("body"),
    "small": _style("small", 8, color=MUTED),
    "cell": _style("cell", 8.5),
    "cellb": _style("cellb", 8.5, "Noto-Bold"),
    "big": _style("big", 30, "Noto-Bold", leading=34),
}


def _rupees(x: float) -> str:
    return f"₹{x:,.0f}"


def build_pdf(record) -> bytes:
    r = record.result
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"ShelfSense audit {record.audit_id}", author="ShelfSense AI")
    width = A4[0] - 32 * mm
    when = datetime.fromisoformat(record.created_at).strftime("%d %b %Y, %H:%M")
    story = []

    head = Table([[Paragraph("ShelfSense audit", S["title"])],
                  [Paragraph(f"{record.store_id} · {record.bay_id} · planogram {record.planogram_id} · {when}",
                             S["sub"])]], colWidths=[width])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), HEADER), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                              ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, -1), (-1, -1), 9)]))
    story += [head, Spacer(1, 6 * mm)]

    # score block: big number, band chip with icon + label, components
    color, icon = BAND[r.score.band]
    icon_font = "Noto-Bold" if icon == "!" else "NotoSym"
    chip = Table([[Paragraph(f'<font name="{icon_font}">{icon}</font>  {r.score.band}',
                             _style("chip", 10, "Noto-Bold", colors.white))]])
    chip.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)),
                              ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
    counts = record.counts()
    comp = [[Paragraph("Placement", S["small"]), Paragraph("Labels", S["small"]), Paragraph("Availability", S["small"]),
             Paragraph("Revenue at risk", S["small"])],
            [Paragraph(f"{r.score.placement:.0%}", S["cellb"]), Paragraph(f"{r.score.labels:.0%}", S["cellb"]),
             Paragraph(f"{r.score.availability:.0%}", S["cellb"]),
             Paragraph(f"{_rupees(r.revenue_at_risk)} / day", S["cellb"])]]
    comp_t = Table(comp, colWidths=[(width * 0.62) / 4] * 4)
    score_block = Table([[Paragraph(f"{r.score.score:.0f}", S["big"]), chip, comp_t]],
                        colWidths=[width * 0.16, width * 0.22, width * 0.62])
    score_block.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [score_block, Spacer(1, 3 * mm)]
    if r.score.verified < 1:
        story.append(Paragraph(f"Score based on {r.score.verified:.0%} of checks; "
                               f"{counts['review']} item(s) need a check by hand.", S["small"]))

    tiles = [("Out of stock", counts["out_of_stock"]), ("Misplaced", counts["misplaced"]),
             ("Low stock", counts["low_stock"]), ("Price / label", counts["price_label"]),
             ("Extra items", counts["extra"]), ("Check by hand", counts["review"])]
    tile_t = Table([[Paragraph(f"<b>{n}</b>", _style("tn", 14, "Noto-Bold")) for _, n in tiles],
                    [Paragraph(label, S["small"]) for label, _ in tiles]], colWidths=[width / 6] * 6)
    tile_t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, RULE), ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
                                ("TOPPADDING", (0, 0), (-1, 0), 5)]))
    story += [tile_t, Spacer(1, 5 * mm)]

    # annotated shelf
    from PIL import Image as PILImage
    with PILImage.open(record.annotated_path) as im:
        w, h = im.size
    img_w = width
    img_h = min(img_w * h / w, 105 * mm)
    img_w = img_h * w / h
    story += [RLImage(record.annotated_path, width=img_w, height=img_h),
              Paragraph("Green: compliant · amber: low stock · red: violation · purple: check by hand", S["small"])]

    # ranked actions
    story.append(Paragraph("What to do, in order", S["h"]))
    if record.summary:
        story.append(Paragraph(record.summary, S["body"]))
        story.append(Spacer(1, 2 * mm))
    if r.violations:
        rows = [[Paragraph("#", S["cellb"]), Paragraph("Action", S["cellb"]), Paragraph("Type", S["cellb"]),
                 Paragraph("At risk / day", S["cellb"])]]
        for i, v in enumerate(r.violations, 1):
            rows.append([Paragraph(str(i), S["cell"]), Paragraph(v.action, S["cell"]),
                         Paragraph(LABELS[v.type], S["cell"]),
                         Paragraph(_rupees(v.revenue_at_risk) if v.revenue_at_risk else "–", S["cell"])])
        t = Table(rows, colWidths=[8 * mm, width - 8 * mm - 30 * mm - 26 * mm, 30 * mm, 26 * mm], repeatRows=1)
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t)
    else:
        story.append(Paragraph("Nothing to do: the shelf matches the planogram.", S["body"]))

    # notes and how the photo was read
    notes = list(r.notes)
    if record.frame_info:
        notes.append(f"Video: frame {record.frame_info.get('frame')} of {record.frame_info.get('frames')} "
                     f"was the sharpest and was audited.")
    t = record.timings_ms
    notes.append(f"Processed offline on CPU in {t.get('total', 0) / 1000:.1f} s "
                 f"(detect {t.get('detect', 0):.0f} ms, identify {t.get('identify', 0):.0f} ms, "
                 f"read tags {t.get('ocr', 0):.0f} ms). Audit {record.audit_id}, source {record.source}.")
    story.append(KeepTogether([Paragraph("Notes", S["h"])] + [Paragraph(n, S["small"]) for n in notes]))

    doc.build(story)
    return buf.getvalue()
