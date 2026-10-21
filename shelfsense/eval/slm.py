"""How often does the language model's wording survive the fact checks, and how long does it take?

    python -m shelfsense.eval.slm --split test --n 30

Runs the rules engine on answer-key shelves (no vision models needed), then
asks Qwen2.5-3B for the task list and summary.
"""

import argparse
import json
from collections import Counter

import numpy as np

from ..compliance.engine import audit
from ..data import load_catalogue, load_planograms
from ..generator.build import load_answer
from ..paths import ROOT, SYNTHETIC
from ..perception.shelf_map import build_shelf_map, flatten
from ..slm.writer import ActionWriter, task_ok, template


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    catalogue = load_catalogue()
    planograms = load_planograms(catalogue)
    writer = ActionWriter(catalogue, threads=args.threads)
    if not writer.available():
        raise SystemExit("llama.cpp server or Qwen model not found (see README setup)")

    sources, seconds, kept_by_type, total_by_type = Counter(), [], Counter(), Counter()
    summaries_kept = shelves = 0
    examples = []
    for f in sorted((SYNTHETIC / args.split / "answers").glob("*.json")):
        if shelves >= args.n:
            break
        key = load_answer(f)
        if not key.violations:
            continue
        p = planograms[key.planogram_id]
        sm = build_shelf_map(key.image_id, key.shelf_map.width, key.shelf_map.height, *flatten(key.shelf_map))
        result = audit(sm, p, catalogue)
        tl = writer.write(result, p.bay_id)
        shelves += 1
        sources[tl.source] += 1
        seconds.append(tl.ms / 1000)
        summaries_kept += tl.summary != template(result, p.bay_id).summary
        for task, v in zip(tl.tasks, result.violations):
            total_by_type[v.type.value] += 1
            kept_by_type[v.type.value] += task != v.action and task_ok(task, v, catalogue)
        if len(examples) < 3:
            examples.append({"shelf": key.image_id, "summary": tl.summary,
                             "tasks": [{"type": v.type.value, "task": t, "from": "model" if t != v.action else "template"}
                                       for t, v in zip(tl.tasks, result.violations)]})
        print(f"  {shelves}/{args.n}  {tl.source:<13} {tl.ms / 1000:5.1f}s", flush=True)
    writer.server.stop()

    total = sum(total_by_type.values())
    rep = {
        "shelves": shelves, "tasks": total,
        "tasks_from_model": round(sum(kept_by_type.values()) / total, 4),
        "tasks_from_model_by_type": {t: round(kept_by_type[t] / n, 3) for t, n in sorted(total_by_type.items())},
        "summary_from_model": round(summaries_kept / shelves, 4),
        "lists": dict(sources),
        "seconds_mean": round(float(np.mean(seconds)), 1), "seconds_p90": round(float(np.percentile(seconds, 90)), 1),
        "examples": examples,
    }
    print(f"\n{shelves} shelves, {total} tasks")
    print(f"tasks worded by the model (passed every check): {rep['tasks_from_model']:.1%}")
    for t, share in rep["tasks_from_model_by_type"].items():
        print(f"    {t:<14}{share:.0%}")
    print(f"summaries from the model: {rep['summary_from_model']:.0%}")
    print(f"time per shelf: {rep['seconds_mean']} s mean, {rep['seconds_p90']} s p90 ({args.threads} threads)")
    out = ROOT / "reports" / f"slm_{args.split}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
