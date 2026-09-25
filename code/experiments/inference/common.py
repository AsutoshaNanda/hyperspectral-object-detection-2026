from dataclasses import dataclass

import numpy as np


N_BANDS = 16


@dataclass(frozen=True)
class PredictionBatch:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    checkpoint_id: str
    source: str = "model_output"
    prediction_stage: str = "post_suppression"

    def __post_init__(self):
        boxes = np.asarray(self.boxes, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float64)
        labels = np.asarray(self.labels)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("boxes must have shape (N, 4) in XYXY coordinates")
        if scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
            raise ValueError("scores and labels must have shape (N,)")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError("boxes and scores must be finite")
        if len(boxes) and ((boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any()):
            raise ValueError("every XYXY box must have positive width and height")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("scores must be between zero and one")
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
                raise ValueError("labels must be finite integer class IDs")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty immutable identifier or hash")
        if self.prediction_stage not in {"raw_pre_suppression", "post_suppression"}:
            raise ValueError("prediction_stage must be raw_pre_suppression or post_suppression")
        boxes = boxes.copy()
        scores = scores.copy()
        labels = labels.astype(np.int64, copy=True)
        boxes.setflags(write=False)
        scores.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "labels", labels)

    def __len__(self):
        return len(self.boxes)

    def subset(self, indexes, *, source=None, prediction_stage=None, scores=None):
        indexes = np.asarray(indexes, dtype=np.int64)
        return PredictionBatch(
            self.boxes[indexes],
            self.scores[indexes] if scores is None else scores,
            self.labels[indexes],
            self.checkpoint_id,
            self.source if source is None else source,
            self.prediction_stage if prediction_stage is None else prediction_stage,
        )


@dataclass(frozen=True)
class MergeResult:
    predictions: PredictionBatch
    input_predictions: int
    kept_predictions: int
    duplicate_predictions: int
    clipped_predictions: int = 0
    discarded_predictions: int = 0


def validate_hyperspectral_cube(cube):
    array = np.asarray(cube)
    if array.ndim != 3 or array.shape[-1] != N_BANDS:
        raise ValueError("hyperspectral cubes must have shape (height, width, 16)")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("hyperspectral cubes must contain finite numeric values")
    if min(array.shape[:2]) < 1:
        raise ValueError("hyperspectral cubes must have positive height and width")
    return array


def require_same_checkpoint(batches):
    batches = tuple(batches)
    if not batches:
        raise ValueError("at least one prediction batch is required")
    checkpoint_ids = {batch.checkpoint_id for batch in batches}
    if len(checkpoint_ids) != 1:
        raise ValueError("predictions from different checkpoints cannot be combined")
    return next(iter(checkpoint_ids))


def concatenate_same_checkpoint(batches, *, source, prediction_stage=None):
    batches = tuple(batches)
    checkpoint_id = require_same_checkpoint(batches)
    stages = {batch.prediction_stage for batch in batches}
    if prediction_stage is None:
        if len(stages) != 1:
            raise ValueError("prediction batches from different suppression stages cannot be combined")
        prediction_stage = next(iter(stages))
    return PredictionBatch(
        np.concatenate([batch.boxes for batch in batches], axis=0),
        np.concatenate([batch.scores for batch in batches], axis=0),
        np.concatenate([batch.labels for batch in batches], axis=0),
        checkpoint_id,
        source,
        prediction_stage,
    )


def box_iou_one_to_many(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float64)
    upper_left = np.maximum(box[:2], boxes[:, :2])
    lower_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(0.0, lower_right - upper_left), axis=1)
    first_area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(first_area + areas - intersection, np.finfo(np.float64).eps)


def clip_prediction_batch(batch, image_width, image_height, *, source=None):
    if not np.isfinite([image_width, image_height]).all() or image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be finite and positive")
    boxes = batch.boxes.copy()
    bounded = np.clip(boxes, [0, 0, 0, 0], [image_width, image_height, image_width, image_height])
    clipped = int(np.any(boxes != bounded, axis=1).sum())
    valid = (bounded[:, 2] > bounded[:, 0]) & (bounded[:, 3] > bounded[:, 1])
    indexes = np.flatnonzero(valid)
    return (
        PredictionBatch(
            bounded[indexes],
            batch.scores[indexes],
            batch.labels[indexes],
            batch.checkpoint_id,
            batch.source if source is None else source,
            batch.prediction_stage,
        ),
        clipped,
        int((~valid).sum()),
    )


def hard_nms_indices(boxes, scores, labels, iou_threshold=0.5, class_agnostic=False):
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between zero and one")
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    kept = []
    groups = [np.arange(len(boxes))] if class_agnostic else [np.flatnonzero(labels == label) for label in np.unique(labels)]
    for group in groups:
        order = group[np.argsort(-scores[group], kind="stable")]
        while len(order):
            current = int(order[0])
            kept.append(current)
            if len(order) == 1:
                break
            remaining = order[1:]
            order = remaining[box_iou_one_to_many(boxes[current], boxes[remaining]) <= iou_threshold]
    return np.asarray(sorted(kept, key=lambda index: (-scores[index], index)), dtype=np.int64)


def merge_predictions(batches, *, iou_threshold=0.5, class_agnostic=False, max_detections=None, source="same_checkpoint_merge"):
    combined = concatenate_same_checkpoint(batches, source=source, prediction_stage="post_suppression")
    keep = hard_nms_indices(combined.boxes, combined.scores, combined.labels, iou_threshold, class_agnostic)
    if max_detections is not None:
        if not isinstance(max_detections, int) or isinstance(max_detections, bool) or max_detections < 1:
            raise ValueError("max_detections must be a positive integer")
        keep = keep[:max_detections]
    predictions = combined.subset(keep, source=source, prediction_stage="post_suppression")
    return MergeResult(predictions, len(combined), len(predictions), len(combined) - len(predictions))
