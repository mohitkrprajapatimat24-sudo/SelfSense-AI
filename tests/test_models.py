"""Checks on the trained models. Skipped when models/ has not been built yet."""

import numpy as np
import pytest

from shelfsense.compliance.engine import audit
from shelfsense.data import load_catalogue, load_planograms
from shelfsense.generator.build import generate_sample
from shelfsense.generator.packs import PackLibrary
from shelfsense.perception.identify import COLOR_DIM, DINO_DIM, color_features
from shelfsense.perception.pipeline import DETECTOR_ONNX, EMBEDDER_ONNX, GALLERY, Perception
from shelfsense.schemas import DetClass

needs_models = pytest.mark.skipif(
    not (DETECTOR_ONNX.exists() and EMBEDDER_ONNX.exists() and (GALLERY / "gallery.faiss").exists()),
    reason="trained models not built (see README)")


@pytest.fixture(scope="module")
def ref():
    catalogue = load_catalogue()
    return catalogue, load_planograms(catalogue), PackLibrary(catalogue)


def test_colour_features_are_unit_vectors(ref):
    catalogue, _, packs = ref
    crops = [packs.get(s, 120, 200) for s in list(catalogue)[:4]]
    f = color_features(crops)
    assert f.shape == (4, COLOR_DIM)
    assert np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5)


@pytest.fixture(scope="module")
def perception(ref):
    return Perception(ref[0])


@needs_models
def test_embeddings_are_unit_vectors(ref, perception):
    catalogue, _, packs = ref
    e = perception.embedder([packs.get(s, 120, 200) for s in list(catalogue)[:3]])
    assert e.shape == (3, DINO_DIM + COLOR_DIM)
    assert np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)


@needs_models
def test_detector_finds_products_and_tags(ref, perception):
    catalogue, planograms, packs = ref
    img, key = generate_sample(planograms["P01"], catalogue, packs, [11, 11, 11], "t", inject=False)
    dets = perception.detector(img)
    n_items = sum(len(r.items) for r in key.shelf_map.rows)
    n_tags = sum(len(r.tags) for r in key.shelf_map.rows)
    assert abs(sum(d.cls == DetClass.PRODUCT for d in dets) - n_items) <= 1
    assert abs(sum(d.cls == DetClass.PRICE_TAG for d in dets) - n_tags) <= 1


@needs_models
def test_pipeline_audits_a_perfect_shelf(ref, perception):
    catalogue, planograms, packs = ref
    img, key = generate_sample(planograms["P02"], catalogue, packs, [12, 12, 12], "t", inject=False)
    result = audit(perception(img).shelf_map, planograms["P02"], catalogue)
    assert result.score.score >= 90
    assert len(result.slots) == sum(len(r.slots) for r in planograms["P02"].rows)
