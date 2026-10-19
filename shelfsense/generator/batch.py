"""Worker functions for parallel generation (kept out of __main__ so Windows can spawn them)."""

from pathlib import Path

import numpy as np

from ..data import load_catalogue, load_planograms
from .build import generate_sample, write_sample
from .packs import PackLibrary

SPLIT_CODES = {"train": 1, "val": 2, "test": 3, "preview": 4}

_ctx: dict = {}


def init_worker(planogram_ids: list[str] | None):
    catalogue = load_catalogue()
    planograms = load_planograms(catalogue)
    if planogram_ids:
        planograms = {k: v for k, v in planograms.items() if k in planogram_ids}
    _ctx.update(catalogue=catalogue, planograms=list(planograms.values()), packs=PackLibrary(catalogue))


def make_one(job: tuple) -> dict:
    split_dir, split, seed, idx, do_augment = job
    planograms = _ctx["planograms"]
    planogram = planograms[idx % len(planograms)]
    image_id = f"{split}_{idx:05d}"
    sample_seed = [seed, SPLIT_CODES.get(split, 9), idx]
    img, key = generate_sample(planogram, _ctx["catalogue"], _ctx["packs"], sample_seed, image_id, do_augment)
    quality = int(np.random.default_rng(sample_seed + [99]).integers(60, 96))
    write_sample(img, key, Path(split_dir), quality)
    return {"score": key.score.score, "band": key.score.band, "violations": [v.type.value for v in key.violations],
            "photos": key.uses_photos}
