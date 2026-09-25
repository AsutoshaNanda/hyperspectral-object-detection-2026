import numpy as np

from .common import MergeResult, PredictionBatch, concatenate_same_checkpoint, merge_predictions, validate_hyperspectral_cube
from .multiscale import MultiScalePlan, restore_multiscale_predictions, scaled_shape


def horizontal_flip_cube(cube):
    array = validate_hyperspectral_cube(cube)
    return np.flip(array, axis=1).copy()


def restore_horizontal_boxes(boxes, image_width):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or not np.isfinite(boxes).all():
        raise ValueError("boxes must be a finite (N, 4) XYXY array")
    if not np.isfinite(image_width) or image_width <= 0:
        raise ValueError("image_width must be finite and positive")
    restored = boxes.copy()
    restored[:, 0] = image_width - boxes[:, 2]
    restored[:, 2] = image_width - boxes[:, 0]
    if len(restored) and ((restored[:, 2] <= restored[:, 0]).any() or (restored[:, 3] <= restored[:, 1]).any()):
        raise ValueError("restored boxes must have positive width and height")
    return restored


def restore_horizontal_predictions(predictions, image_width):
    return PredictionBatch(
        restore_horizontal_boxes(predictions.boxes, image_width),
        predictions.scores,
        predictions.labels,
        predictions.checkpoint_id,
        f"{predictions.source}:horizontal_restored",
        predictions.prediction_stage,
    )


def horizontal_flip_tta(
    normal_predictions,
    flipped_predictions,
    image_width,
    *,
    iou_threshold=0.5,
    class_agnostic=False,
    max_detections=None,
):
    restored = restore_horizontal_predictions(flipped_predictions, image_width)
    return merge_predictions(
        (normal_predictions, restored),
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic,
        max_detections=max_detections,
        source="T1:normal_horizontal_flip",
    )


def flip_multiscale_tta(
    normal_predictions_by_scale,
    flipped_predictions_by_scale,
    original_shape,
    plan,
    *,
    iou_threshold=0.5,
    class_agnostic=False,
    max_detections=None,
):
    if not isinstance(plan, MultiScalePlan):
        raise TypeError("plan must be a frozen MultiScalePlan")
    if set(normal_predictions_by_scale) != set(plan.scales) or set(flipped_predictions_by_scale) != set(plan.scales):
        raise ValueError("normal and flipped prediction scales must exactly match the predeclared plan")
    combined = {}
    for scale in plan.scales:
        scaled_width = scaled_shape(original_shape, scale)[1]
        restored_flip = restore_horizontal_predictions(flipped_predictions_by_scale[scale], scaled_width)
        combined[scale] = concatenate_same_checkpoint(
            (normal_predictions_by_scale[scale], restored_flip),
            source=f"T3:{plan.policy_id}:{scale:g}",
            prediction_stage="post_suppression",
        )
    result = restore_multiscale_predictions(
        combined,
        original_shape,
        plan,
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic,
        max_detections=max_detections,
    )
    predictions = PredictionBatch(
        result.predictions.boxes,
        result.predictions.scores,
        result.predictions.labels,
        result.predictions.checkpoint_id,
        f"T3:{plan.policy_id}",
        "post_suppression",
    )
    return MergeResult(
        predictions,
        result.input_predictions,
        result.kept_predictions,
        result.duplicate_predictions,
        result.clipped_predictions,
        result.discarded_predictions,
    )
