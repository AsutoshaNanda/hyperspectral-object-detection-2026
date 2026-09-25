from dataclasses import dataclass

import numpy as np

from inference_common import MergeResult, PredictionBatch, clip_prediction_batch, merge_predictions, validate_hyperspectral_cube


@dataclass(frozen=True)
class MultiScalePlan:
    scales: tuple[float, ...]
    policy_id: str
    selected_on: str = "validation_cv"
    frozen: bool = True

    def __post_init__(self):
        scales = tuple(float(scale) for scale in self.scales)
        if len(scales) < 2 or not np.isfinite(scales).all() or any(scale <= 0 for scale in scales):
            raise ValueError("a multi-scale plan requires at least two finite positive scales")
        if len(set(scales)) != len(scales):
            raise ValueError("multi-scale plan scales must be unique")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty predeclared policy identifier")
        if self.selected_on not in {"validation", "cv", "validation_cv"}:
            raise ValueError("multi-scale policies must be selected on validation/CV, never holdout or test")
        if self.frozen is not True:
            raise ValueError("the multi-scale policy must be frozen before evaluation")
        object.__setattr__(self, "scales", scales)


def scaled_shape(original_shape, scale):
    if len(original_shape) != 2:
        raise ValueError("original_shape must be (height, width)")
    height, width = original_shape
    if not isinstance(height, int) or not isinstance(width, int) or min(height, width) < 1:
        raise ValueError("original dimensions must be positive integers")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def scale_cube(cube, scale):
    array = validate_hyperspectral_cube(cube)
    target_height, target_width = scaled_shape(array.shape[:2], scale)
    if (target_height, target_width) == array.shape[:2]:
        return array.copy()
    source_y = np.clip((np.arange(target_height) + 0.5) * array.shape[0] / target_height - 0.5, 0, array.shape[0] - 1)
    source_x = np.clip((np.arange(target_width) + 0.5) * array.shape[1] / target_width - 0.5, 0, array.shape[1] - 1)
    y0 = np.floor(source_y).astype(np.int64)
    x0 = np.floor(source_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, array.shape[0] - 1)
    x1 = np.minimum(x0 + 1, array.shape[1] - 1)
    wy = (source_y - y0)[:, None, None]
    wx = (source_x - x0)[None, :, None]
    top = array[y0[:, None], x0[None, :]] * (1 - wx) + array[y0[:, None], x1[None, :]] * wx
    bottom = array[y1[:, None], x0[None, :]] * (1 - wx) + array[y1[:, None], x1[None, :]] * wx
    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32
    return (top * (1 - wy) + bottom * wy).astype(dtype, copy=False)


def restore_scaled_predictions(predictions, original_shape, scaled_image_shape, *, source=None):
    original_height, original_width = original_shape
    scaled_height, scaled_width = scaled_image_shape
    if min(original_height, original_width, scaled_height, scaled_width) < 1:
        raise ValueError("image dimensions must be positive")
    scale_x = scaled_width / original_width
    scale_y = scaled_height / original_height
    boxes = predictions.boxes / np.asarray([scale_x, scale_y, scale_x, scale_y])
    restored = PredictionBatch(
        boxes,
        predictions.scores,
        predictions.labels,
        predictions.checkpoint_id,
        source or f"{predictions.source}:scale_restored",
        predictions.prediction_stage,
    )
    return clip_prediction_batch(restored, original_width, original_height)


def restore_multiscale_predictions(
    predictions_by_scale,
    original_shape,
    plan,
    *,
    iou_threshold=0.5,
    class_agnostic=False,
    max_detections=None,
):
    if not isinstance(plan, MultiScalePlan):
        raise TypeError("plan must be a frozen MultiScalePlan")
    if set(predictions_by_scale) != set(plan.scales):
        raise ValueError("prediction scales must exactly match the predeclared multi-scale plan")
    restored = []
    clipped = 0
    discarded = 0
    for scale in plan.scales:
        shape = scaled_shape(original_shape, scale)
        batch, batch_clipped, batch_discarded = restore_scaled_predictions(
            predictions_by_scale[scale], original_shape, shape, source=f"multiscale:{plan.policy_id}:{scale:g}"
        )
        restored.append(batch)
        clipped += batch_clipped
        discarded += batch_discarded
    merged = merge_predictions(
        restored,
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic,
        max_detections=max_detections,
        source=f"T2:{plan.policy_id}",
    )
    return MergeResult(
        merged.predictions,
        merged.input_predictions + discarded,
        merged.kept_predictions,
        merged.duplicate_predictions,
        clipped,
        discarded,
    )
