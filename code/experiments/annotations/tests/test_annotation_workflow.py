import json

import numpy as np
import pytest

from Experiments.annotations.build_augmented_labels import build_training_copy_annotations
from Experiments.annotations.find_missing_candidates import (
    DetectionCandidate,
    discover_candidates,
    export_candidate_package,
    select_validation_threshold,
)
from Experiments.annotations.review_annotations import ReviewLabel, ReviewRecord


def candidate(candidate_id, image_id, category_id=2, bbox=(10, 10, 14, 14)):
    return DetectionCandidate(candidate_id, image_id, category_id, bbox, 0.97, 0.0, 0.9, "curve-1")


def review(candidate_id, label):
    return ReviewRecord(candidate_id, label, "reviewer", "2026-09-20T00:00:00Z")


def test_candidates_require_validation_selected_threshold_and_training_images(tmp_path):
    selection = select_validation_threshold(
        [
            {"threshold": 0.5, "precision": 0.91, "recall": 0.8, "source_split": "validation"},
            {"threshold": 0.8, "precision": 0.97, "recall": 0.6, "source_split": "validation"},
        ],
        0.95,
        "fold-0-validation-precision",
    )
    predictions = [
        {"image_id": "train-1", "category_id": 2, "bbox": [10, 10, 14, 14], "score": 0.96},
        {"image_id": "train-1", "category_id": 2, "bbox": [0, 0, 4, 4], "score": 0.99},
        {"image_id": "train-1", "category_id": 2, "bbox": [15, 15, 19, 19], "score": 0.4},
    ]
    official = [{"image_id": "train-1", "category_id": 2, "bbox": [0, 0, 4, 4]}]
    candidates = discover_candidates(predictions, official, {"train-1"}, selection, 0.2)
    assert len(candidates) == 1
    assert candidates[0].threshold_curve_id == "fold-0-validation-precision"
    cube = np.arange(20 * 20 * 16, dtype=np.float32).reshape(20, 20, 16)
    package = export_candidate_package(candidates, {"train-1": cube}, tmp_path)
    manifest = json.loads(package.manifest_path.read_text())
    assert package.candidate_count == 1
    assert (tmp_path / manifest["candidates"][0]["pseudo_rgb"]).read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / manifest["candidates"][0]["spectral_csv"]).read_text().splitlines()[0].startswith("y,x,band_0")
    with pytest.raises(ValueError, match="non-training"):
        discover_candidates(
            [{"image_id": "validation-1", "category_id": 2, "bbox": [1, 1, 3, 3], "score": 0.99}],
            official,
            {"train-1"},
            selection,
            0.2,
        )


def test_reviewed_labels_only_change_training_copy_and_uncertain_has_safe_fallback():
    candidates = [
        candidate("confirmed", "train"),
        candidate("uncertain", "train", bbox=(20, 20, 24, 24)),
        candidate("negative", "train", bbox=(30, 30, 34, 34)),
    ]
    reviews = [
        review("confirmed", ReviewLabel.CONFIRMED_OBJECT),
        review("uncertain", ReviewLabel.UNCERTAIN),
        review("negative", ReviewLabel.NOT_OBJECT),
    ]
    official = [
        {"id": 1, "image_id": "train", "category_id": 0, "bbox": [0, 0, 4, 4]},
        {"id": 2, "image_id": "validation", "category_id": 0, "bbox": [1, 1, 4, 4]},
        {"id": 3, "image_id": "holdout", "category_id": 0, "bbox": [2, 2, 4, 4]},
    ]
    snapshot = json.dumps(official, sort_keys=True)
    augmented = build_training_copy_annotations(
        official,
        candidates,
        reviews,
        {"train"},
        validation_image_ids={"validation"},
        holdout_image_ids={"holdout"},
        supports_ignore_regions=False,
    )
    assert augmented.confirmed_count == 1
    assert augmented.uncertain_count == 1
    assert len(augmented.training_annotations) == 2
    assert augmented.training_annotations[-1]["source"] == "manually_confirmed_missing_object"
    assert not augmented.ignore_regions
    assert augmented.excluded_hard_negative_regions[0]["candidate_id"] == "uncertain"
    assert json.dumps(official, sort_keys=True) == snapshot


def test_uncertain_candidates_become_ignore_regions_when_supported():
    item = candidate("uncertain", "train")
    augmented = build_training_copy_annotations(
        [],
        [item],
        [review("uncertain", ReviewLabel.UNCERTAIN)],
        {"train"},
        supports_ignore_regions=True,
    )
    assert augmented.ignore_regions[0]["candidate_id"] == "uncertain"
    assert not augmented.excluded_hard_negative_regions
