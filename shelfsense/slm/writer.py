"""Turning the ranked violations into a staff task list and a short summary.

The rules engine has already decided what is wrong and in which order; the
language model only rewrites that into plain, short instructions. Its output
is constrained to JSON with exactly one task per violation, and every task is
checked against the facts (row, position, destination). A task that drops or
changes a location falls back to the template wording, and if the model is
missing, slow or fails, the whole list comes from the template.
"""

import json
import re
import time
from dataclasses import dataclass

import requests

from ..schemas import UNKNOWN_SKU, AuditResult, CatalogueItem, Violation, ViolationType as V
from .server import URL, LlamaServer

MAX_TASKS = 10            # longer lists: the rest keep their template wording
TIMEOUT_S = 90

SYSTEM = ("You write short, clear task lists for supermarket shelf staff. Use only the facts given. "
          "Never add products, prices, quantities or locations that are not in the facts.")


@dataclass
class TaskList:
    summary: str
    tasks: list[str]
    source: str               # "slm", "slm+template" or "template"
    ms: float = 0.0


def _name(sku: str | None, catalogue: dict[str, CatalogueItem]) -> str:
    if sku is None:
        return "an unreadable item"
    if sku == UNKNOWN_SKU:
        return "an unknown product"
    return catalogue[sku].name if sku in catalogue else sku


def fact(v: Violation, catalogue: dict[str, CatalogueItem]) -> str:
    """One plain sentence per violation: everything the model may use, nothing more."""
    where = f"Row {v.row} Pos {v.position}"
    risk = f", ₹{v.revenue_at_risk:,.0f}/day of sales at risk" if v.revenue_at_risk else ""
    name = _name(v.expected_sku, catalogue)
    if v.type == V.OUT_OF_STOCK:
        return f"{name}, {where}: none left on the shelf, {v.count} facings to refill{risk}"
    if v.type == V.LOW_STOCK:
        return f"{name}, {where}: {v.found_facings} of {v.expected_facings} facings filled, {v.count} to add{risk}"
    if v.type == V.MISPLACED:
        return (f"{_name(v.found_sku, catalogue)} is in {where}, which is planned for {name}; "
                f"it belongs in Row {v.target_row} Pos {v.target_position}")
    if v.type == V.EXTRA:
        return f"{_name(v.found_sku, catalogue)} is in {where} but is not planned for this bay; remove it"
    if v.type == V.WRONG_PRICE:
        shown = f"₹{v.found_price:g}" if v.found_price is not None else "no price"
        return f"Price tag for {name} at {where} shows {shown}, the catalogue price is ₹{v.expected_price:g}"
    if v.type == V.WRONG_LABEL:
        return (f"Price tag at {where} is for {_name(v.found_sku, catalogue)}, "
                f"but the slot holds {name} (₹{v.expected_price:g})")
    return f"{where} could not be read reliably; a person should check it"


# what each kind of problem asks staff to do: a hint in the prompt, and a check on the answer
VERB_HINT = {V.OUT_OF_STOCK: "Restock", V.LOW_STOCK: "Top up", V.MISPLACED: "Move", V.EXTRA: "Remove",
             V.WRONG_PRICE: "Fix price tag", V.WRONG_LABEL: "Fix price tag", V.NEEDS_REVIEW: "Check by hand"}
ALLOWED = {
    V.OUT_OF_STOCK: r"restock|refill|replenish|stock|fill|bring",
    V.LOW_STOCK: r"top up|top-up|replenish|refill|restock|add|fill|stock",
    V.MISPLACED: r"move|relocate|return|put back|shift",
    V.EXTRA: r"remove|take (?:out|off|away)|clear|pull",
    V.WRONG_PRICE: r"tag|label|price",
    V.WRONG_LABEL: r"tag|label",
    V.NEEDS_REVIEW: r"check|inspect|verify|look",
}
FORBIDDEN = {
    V.OUT_OF_STOCK: r"\bremove|\bmove\b", V.LOW_STOCK: r"\bremove|\bmove\b",
    V.MISPLACED: r"\bremove|restock|refill", V.EXTRA: r"\badd\b|restock|refill|replenish|\bmove\b",
    V.WRONG_PRICE: r"restock|refill|\bmove\b", V.WRONG_LABEL: r"restock|refill|\bmove\b",
    V.NEEDS_REVIEW: r"restock|refill|\bmove\b|\bremove",
}


def _at(row: int, pos: int) -> str:
    """'Row 2 Pos 3', 'Row 2, Position 3' or 'R2-P3'."""
    return rf"(?:row\s*{row}\b[,\s]*(?:pos(?:ition)?\.?\s*{pos}\b)|\br{row}\s*-\s*p{pos}\b)"


def tidy(task: str) -> str:
    """'[Top up] Maggi ..., ₹150/day at risk' -> 'Top up Maggi ...' (the money is shown next to each task)."""
    task = re.sub(r"^\[([^\]]+)\]\s*:?\s*", r"\1 ", task.strip())
    task = re.sub(r"[,;]?\s*(?:₹|Rs\.?\s?)[\d,]+(?:\.\d+)?\s*(?:/|per\s+)day(?:\s+of\s+sales)?(?:\s+at\s+risk)?", "", task)
    task = re.sub(r"\s{2,}", " ", task).strip(" ,;")
    return task[:1].upper() + task[1:]


def task_ok(task: str, v: Violation, catalogue: dict[str, CatalogueItem]) -> bool:
    """A task must keep the location(s), use the right kind of verb, and name the right product only."""
    t = task.lower()
    if not re.search(_at(v.row, v.position), t):
        return False
    if v.type == V.MISPLACED and v.target_row is not None and not re.search(_at(v.target_row, v.target_position), t):
        return False
    if not re.search(ALLOWED[v.type], t) or re.search(FORBIDDEN[v.type], t):
        return False
    subject = v.found_sku if v.type in (V.MISPLACED, V.EXTRA) else v.expected_sku
    allowed = {catalogue[s].brand.lower() for s in (v.expected_sku, v.found_sku) if s in catalogue}
    if subject in catalogue and catalogue[subject].brand.lower() not in t and v.type != V.WRONG_LABEL:
        return False
    mentioned = {item.brand.lower() for item in catalogue.values() if item.brand.lower() in t}
    return mentioned <= allowed


def template(result: AuditResult, bay: str) -> TaskList:
    s = result.score
    by_type: dict[V, int] = {}
    for v in result.violations:
        by_type[v.type] = by_type.get(v.type, 0) + 1
    parts = []
    for t, label in [(V.OUT_OF_STOCK, "out of stock"), (V.MISPLACED, "misplaced"), (V.LOW_STOCK, "low on stock"),
                     (V.WRONG_PRICE, "with a wrong price tag"), (V.WRONG_LABEL, "with the wrong tag"),
                     (V.EXTRA, "with an unplanned product")]:
        if by_type.get(t):
            n = by_type[t]
            parts.append(f"{n} slot{'s' if n > 1 else ''} {label}")
    if not result.violations:
        summary = f"{bay} scores {s.score:.0f}/100 ({s.band}): the shelf matches the planogram."
    else:
        first = result.violations[0]
        summary = (f"{bay} scores {s.score:.0f}/100 ({s.band}): {', '.join(parts) or 'items to check'}. "
                   f"Start with: {first.action[0].lower() + first.action[1:]}.")
        if result.revenue_at_risk:
            summary += f" About ₹{result.revenue_at_risk:,.0f} of sales a day is at risk."
    return TaskList(summary, [v.action for v in result.violations], "template")


class ActionWriter:
    def __init__(self, catalogue: dict[str, CatalogueItem], threads: int = 4, url: str = URL):
        self.catalogue = catalogue
        self.server = LlamaServer(threads, url)
        self.url = url

    def available(self) -> bool:
        return self.server.ensure()

    def write(self, result: AuditResult, bay: str, store: str = "") -> TaskList:
        fallback = template(result, bay)
        if not result.violations or not self.available():
            return fallback
        vs = result.violations[:MAX_TASKS]
        facts = "\n".join(f"{i}. [{VERB_HINT[v.type]}] {fact(v, self.catalogue)}" for i, v in enumerate(vs, 1))
        prompt = (f"Shelf audit for {bay}{' in ' + store if store else ''}. Compliance score "
                  f"{result.score.score:.0f}/100 ({result.score.band}). Problems, most urgent first:\n{facts}\n\n"
                  f"Return JSON with:\n- summary: 1-2 sentences for the store manager: how the shelf is doing and "
                  f"the most important fix.\n- tasks: exactly {len(vs)} instructions, one per problem in the same "
                  f"order. Task N is only about problem N. Each starts with the action in brackets, names the "
                  f"product, says where as 'Row X Pos Y', and is under 20 words.")
        schema = {"type": "object", "required": ["summary", "tasks"], "properties": {
            "summary": {"type": "string"},
            "tasks": {"type": "array", "items": {"type": "string"}, "minItems": len(vs), "maxItems": len(vs)}}}
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.url}/v1/chat/completions", timeout=TIMEOUT_S, json={
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "max_tokens": 60 + 40 * len(vs), "temperature": 0.2,
                "response_format": {"type": "json_schema", "json_schema": {"name": "tasks", "schema": schema}}})
            out = json.loads(r.json()["choices"][0]["message"]["content"])
            summary, tasks = str(out["summary"]).strip(), [tidy(str(t)) for t in out["tasks"]]
        except (requests.RequestException, KeyError, ValueError, TypeError):
            return fallback
        ms = (time.perf_counter() - t0) * 1000
        if len(tasks) != len(vs) or not summary:
            return TaskList(fallback.summary, fallback.tasks, "template", ms)
        # the summary may only use numbers and products that were in the facts it was given
        allowed = set(re.findall(r"\d+", prompt.replace(",", "")))
        brands = {self.catalogue[s].brand.lower() for v in vs for s in (v.expected_sku, v.found_sku)
                  if s in self.catalogue}
        said = {i.brand.lower() for i in self.catalogue.values() if i.brand.lower() in summary.lower()}
        if not set(re.findall(r"\d+", summary.replace(",", ""))) <= allowed or not said <= brands:
            summary = fallback.summary

        checked, replaced = [], 0
        for task, v in zip(tasks, vs):
            if task and task_ok(task, v, self.catalogue):
                checked.append(task)
            else:
                checked.append(v.action)
                replaced += 1
        checked += [v.action for v in result.violations[MAX_TASKS:]]
        source = "slm" if replaced == 0 and len(result.violations) <= MAX_TASKS else "slm+template"
        return TaskList(summary, checked, source, ms)
