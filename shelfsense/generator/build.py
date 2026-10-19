"""One synthetic sample: image + YOLO labels + answer key."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from ..compliance.rules import compute_score, evaluate_slots
from ..schemas import (YOLO_CLASSES, CatalogueItem, DetClass, Planogram, ScoreBreakdown,
                       ShelfMap, SlotState, Violation)
from .augment import augment
from .mistakes import Mistake, inject_mistakes, perfect_state
from .packs import PackLibrary
from .render import render_shelf


class AnswerKey(BaseModel):
    image_id: str
    image_file: str
    planogram_id: str
    seed: list[int]
    augmented: bool
    uses_photos: bool
    shelf_map: ShelfMap
    slots: list[SlotState]
    mistakes: list[Mistake]
    violations: list[Violation]
    score: ScoreBreakdown
    revenue_at_risk: float


def generate_sample(planogram: Planogram, catalogue: dict[str, CatalogueItem], packs: PackLibrary,
                    seed: list[int], image_id: str, do_augment: bool = True,
                    inject: bool = True) -> tuple[Image.Image, AnswerKey]:
    rng = np.random.default_rng(seed)
    slots = perfect_state(planogram, catalogue)
    mistakes = inject_mistakes(slots, planogram, catalogue, rng) if inject else []
    img, shelf_map = render_shelf(planogram, slots, catalogue, packs, rng, image_id)
    if do_augment:
        img, shelf_map = augment(img, shelf_map, rng)
    violations = evaluate_slots(slots, planogram, catalogue)
    key = AnswerKey(
        image_id=image_id, image_file=f"{image_id}.jpg", planogram_id=planogram.planogram_id, seed=seed,
        augmented=do_augment, uses_photos=any(packs.has_photos(s) for s in planogram.skus()),
        shelf_map=shelf_map, slots=slots, mistakes=mistakes, violations=violations,
        score=compute_score(slots, catalogue),
        revenue_at_risk=round(sum(v.revenue_at_risk for v in violations), 2))
    return img, key


def yolo_lines(shelf_map: ShelfMap) -> list[str]:
    W, H = shelf_map.width, shelf_map.height
    lines = []
    for row in shelf_map.rows:
        for cls, objs in ((DetClass.PRODUCT, row.items), (DetClass.PRICE_TAG, row.tags), (DetClass.EMPTY_GAP, row.gaps)):
            for o in objs:
                b = o.bbox
                lines.append(f"{YOLO_CLASSES.index(cls)} {b.cx / W:.6f} {b.cy / H:.6f} {b.w / W:.6f} {b.h / H:.6f}")
    return lines


def write_sample(img: Image.Image, key: AnswerKey, split_dir: Path, jpeg_quality: int) -> None:
    img.save(split_dir / "images" / key.image_file, quality=jpeg_quality)
    (split_dir / "labels" / f"{key.image_id}.txt").write_text("\n".join(yolo_lines(key.shelf_map)) + "\n")
    (split_dir / "answers" / f"{key.image_id}.json").write_text(key.model_dump_json(indent=1), encoding="utf-8")


def load_answer(path: Path) -> AnswerKey:
    return AnswerKey(**json.loads(Path(path).read_text(encoding="utf-8")))
