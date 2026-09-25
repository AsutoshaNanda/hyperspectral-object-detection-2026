import numpy as np
import pytest

from Experiments.inference.common import PredictionBatch
from Experiments.inference.multiscale import MultiScalePlan, restore_multiscale_predictions, scale_cube
from Experiments.inference.sliced_inference import generate_slice_windows, merge_sliced_predictions
from Experiments.inference.soft_nms import compare_suppression
from Experiments.inference.tta import horizontal_flip_cube, horizontal_flip_tta
from Experiments.patches.high_resolution import validate_high_resolution_feature_level
from Experiments.patches.training import PatchConfig, generate_training_patches


def batch(boxes, scores=None, labels=None, checkpoint="abc", stage="post_suppression"):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    count = len(boxes)
    return PredictionBatch(
        boxes,
        np.asarray(scores if scores is not None else [0.9] * count),
        np.asarray(labels if labels is not None else [0] * count),
        checkpoint,
        prediction_stage=stage,
    )


def test_flip_tta_restores_boxes_and_rejects_different_checkpoints():
    cube = np.arange(4 * 10 * 16).reshape(4, 10, 16)
    np.testing.assert_array_equal(horizontal_flip_cube(cube)[:, 0], cube[:, -1])
    normal = batch([[2, 1, 5, 3]])
    flipped = batch([[5, 1, 8, 3]])
    merged = horizontal_flip_tta(normal, flipped, 10)
    assert merged.kept_predictions == 1
    with pytest.raises(ValueError, match="different checkpoints"):
        horizontal_flip_tta(normal, batch([[5, 1, 8, 3]], checkpoint="other"), 10)


def test_multiscale_requires_predeclared_exact_scales_and_restores_coordinates():
    plan = MultiScalePlan((0.5, 1.0), "fixed_scales")
    cube = np.zeros((20, 40, 16), dtype=np.float32)
    assert scale_cube(cube, 0.5).shape == (10, 20, 16)
    result = restore_multiscale_predictions(
        {0.5: batch([[2, 1, 5, 4]]), 1.0: batch([[4, 2, 10, 8]])},
        (20, 40),
        plan,
    )
    assert result.kept_predictions == 1
    np.testing.assert_allclose(result.predictions.boxes[0], [4, 2, 10, 8])
    with pytest.raises(ValueError, match="exactly match"):
        restore_multiscale_predictions({1.0: batch([[4, 2, 10, 8]])}, (20, 40), plan)


def test_sliced_inference_restores_windows_and_tracks_duplicates():
    windows = generate_slice_windows((100, 140), (80, 80), overlap=(0.25, 0.25))
    predictions = {}
    for window in windows:
        predictions[window.index] = batch([[10, 10, 30, 30]])
    report = merge_sliced_predictions(predictions, windows, (100, 140), checkpoint_id="abc")
    assert report.raw_predictions == len(windows)
    assert report.result.kept_predictions <= len(windows)
    with pytest.raises(ValueError, match="different checkpoints"):
        changed = dict(predictions)
        changed[windows[0].index] = batch([[10, 10, 30, 30]], checkpoint="other")
        merge_sliced_predictions(changed, windows, (100, 140), checkpoint_id="abc")


def test_soft_nms_requires_raw_compatible_detector_predictions():
    raw = batch(
        [[0, 0, 10, 10], [1, 1, 11, 11]],
        [0.9, 0.8],
        [0, 0],
        stage="raw_pre_suppression",
    )
    results = compare_suppression(raw, detector_family="yolo", iou_threshold=0.5)
    assert set(results) == {"normal", "linear", "gaussian"}
    assert results["normal"].result.kept_predictions == 1
    assert results["linear"].result.predictions.scores[-1] < 0.8
    with pytest.raises(ValueError, match="NMS-based"):
        compare_suppression(raw, detector_family="rtdetr")
    with pytest.raises(ValueError, match="raw_pre_suppression"):
        compare_suppression(batch([[0, 0, 10, 10]]), detector_family="yolo")


def test_patch_training_preserves_bands_and_records_clipping_and_duplicates():
    cube = np.zeros((100, 140, 16), dtype=np.float32)
    boxes = np.array([[50, 20, 90, 60], [2, 2, 12, 12]], dtype=float)
    result = generate_training_patches(
        cube,
        boxes,
        np.array([1, 2]),
        "image-1",
        PatchConfig((80, 80), overlap=(0.5, 0.5), min_visible_fraction=0.5),
        annotation_ids=("a", "b"),
    )
    assert all(patch.cube.shape[-1] == 16 for patch in result.patches)
    assert result.clipped_box_instances > 0
    assert result.duplicate_box_instances > 0
    assert len({patch.patch_id for patch in result.patches}) == len(result.patches)
    with pytest.raises(ValueError, match="training-only"):
        generate_training_patches(cube, boxes, np.array([1, 2]), "image-1", PatchConfig((80, 80)), split="val")


def test_p5_guard_requires_a_new_supported_dense_level():
    report = validate_high_resolution_feature_level(
        "fpn",
        input_bands=16,
        base_feature_strides=(8, 16, 32),
        candidate_feature_strides=(4, 8, 16, 32),
        supported_feature_strides=(4, 8, 16, 32),
    )
    assert report["added_feature_strides"] == [4]
    with pytest.raises(ValueError, match="explicitly support"):
        validate_high_resolution_feature_level(
            "bad",
            input_bands=16,
            base_feature_strides=(8, 16, 32),
            candidate_feature_strides=(4, 8, 16, 32),
            supported_feature_strides=(8, 16, 32),
        )
