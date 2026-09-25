from dataclasses import dataclass

import numpy as np

from .common import MergeResult, PredictionBatch, clip_prediction_batch, merge_predictions, validate_hyperspectral_cube


@dataclass(frozen=True)
class SliceWindow:
    x1: int
    y1: int
    x2: int
    y2: int
    index: int

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


@dataclass(frozen=True)
class SlicedInferenceReport:
    result: MergeResult
    windows: tuple[SliceWindow, ...]
    checkpoint_id: str
    raw_predictions: int


def _axis_starts(length, window, overlap):
    if not isinstance(length, int) or not isinstance(window, int) or min(length, window) < 1:
        raise ValueError("image and slice dimensions must be positive integers")
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap fractions must be in [0, 1)")
    window = min(length, window)
    stride = max(1, int(round(window * (1 - overlap))))
    starts = list(range(0, max(length - window, 0) + 1, stride))
    final = length - window
    if not starts or starts[-1] != final:
        starts.append(final)
    return tuple(dict.fromkeys(starts)), window


def generate_slice_windows(image_shape, slice_shape, overlap=(0.2, 0.2)):
    if len(image_shape) != 2 or len(slice_shape) != 2 or len(overlap) != 2:
        raise ValueError("image_shape, slice_shape and overlap must each have two values")
    image_height, image_width = image_shape
    slice_height, slice_width = slice_shape
    y_starts, actual_height = _axis_starts(image_height, slice_height, float(overlap[0]))
    x_starts, actual_width = _axis_starts(image_width, slice_width, float(overlap[1]))
    return tuple(
        SliceWindow(x, y, x + actual_width, y + actual_height, index)
        for index, (y, x) in enumerate((y, x) for y in y_starts for x in x_starts)
    )


def extract_slice(cube, window):
    array = validate_hyperspectral_cube(cube)
    if not isinstance(window, SliceWindow):
        raise TypeError("window must be a SliceWindow")
    if window.x1 < 0 or window.y1 < 0 or window.x2 > array.shape[1] or window.y2 > array.shape[0]:
        raise ValueError("slice window lies outside the image")
    return array[window.y1:window.y2, window.x1:window.x2, :].copy()


def restore_slice_predictions(predictions, window, image_shape):
    image_height, image_width = image_shape
    local, local_clipped, local_discarded = clip_prediction_batch(predictions, window.width, window.height)
    boxes = local.boxes + np.asarray([window.x1, window.y1, window.x1, window.y1])
    restored = PredictionBatch(
        boxes,
        local.scores,
        local.labels,
        local.checkpoint_id,
        f"{local.source}:slice_{window.index}_restored",
        local.prediction_stage,
    )
    bounded, global_clipped, global_discarded = clip_prediction_batch(restored, image_width, image_height)
    return bounded, local_clipped + global_clipped, local_discarded + global_discarded


def merge_sliced_predictions(
    predictions_by_window,
    windows,
    image_shape,
    *,
    checkpoint_id,
    iou_threshold=0.5,
    class_agnostic=False,
    max_detections=None,
):
    windows = tuple(windows)
    if set(predictions_by_window) != {window.index for window in windows}:
        raise ValueError("predictions must contain exactly one batch for every declared slice window")
    restored = []
    raw = 0
    clipped = 0
    discarded = 0
    for window in windows:
        batch = predictions_by_window[window.index]
        if batch.checkpoint_id != checkpoint_id:
            raise ValueError("sliced inference cannot combine predictions from different checkpoints")
        raw += len(batch)
        transformed, batch_clipped, batch_discarded = restore_slice_predictions(batch, window, image_shape)
        restored.append(transformed)
        clipped += batch_clipped
        discarded += batch_discarded
    merged = merge_predictions(
        restored,
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic,
        max_detections=max_detections,
        source="T4/P3:sliced_inference",
    )
    result = MergeResult(
        merged.predictions,
        raw,
        merged.kept_predictions,
        merged.duplicate_predictions,
        clipped,
        discarded,
    )
    return SlicedInferenceReport(result, windows, checkpoint_id, raw)


def run_sliced_inference(
    cube,
    predictor,
    checkpoint_id,
    *,
    slice_shape,
    overlap=(0.2, 0.2),
    iou_threshold=0.5,
    class_agnostic=False,
    max_detections=None,
):
    array = validate_hyperspectral_cube(cube)
    if not callable(predictor):
        raise TypeError("predictor must be callable as predictor(slice_cube, slice_window)")
    windows = generate_slice_windows(array.shape[:2], slice_shape, overlap)
    predictions = {}
    for window in windows:
        batch = predictor(extract_slice(array, window), window)
        if not isinstance(batch, PredictionBatch):
            raise TypeError("predictor must return PredictionBatch")
        predictions[window.index] = batch
    return merge_sliced_predictions(
        predictions,
        windows,
        array.shape[:2],
        checkpoint_id=checkpoint_id,
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic,
        max_detections=max_detections,
    )
