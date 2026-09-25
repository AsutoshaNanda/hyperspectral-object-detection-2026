from dataclasses import dataclass

import numpy as np

from Experiments.inference.common import validate_hyperspectral_cube
from Experiments.inference.sliced_inference import SliceWindow, generate_slice_windows


@dataclass(frozen=True)
class PatchConfig:
    patch_shape: tuple[int, int]
    overlap: tuple[float, float] = (0.2, 0.2)
    min_visible_fraction: float = 0.5
    include_empty: bool = True

    def __post_init__(self):
        if len(self.patch_shape) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in self.patch_shape):
            raise ValueError("patch_shape must contain two positive integers")
        if len(self.overlap) != 2 or not np.isfinite(self.overlap).all() or any(not 0 <= value < 1 for value in self.overlap):
            raise ValueError("overlap fractions must be in [0, 1)")
        if not np.isfinite(self.min_visible_fraction) or not 0 < self.min_visible_fraction <= 1:
            raise ValueError("min_visible_fraction must be in (0, 1]")


@dataclass(frozen=True)
class PatchExample:
    patch_id: str
    source_image_id: str
    window: SliceWindow
    cube: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    source_annotation_ids: tuple[str, ...]
    visible_fractions: np.ndarray


@dataclass(frozen=True)
class PatchGenerationResult:
    patches: tuple[PatchExample, ...]
    source_boxes: int
    emitted_box_instances: int
    clipped_box_instances: int
    discarded_low_visibility_instances: int
    duplicate_box_instances: int
    min_visible_fraction: float


def _validate_annotations(boxes, labels, annotation_ids):
    boxes = np.asarray(boxes, dtype=np.float64)
    labels = np.asarray(labels)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or not np.isfinite(boxes).all():
        raise ValueError("boxes must be a finite (N, 4) XYXY array")
    if len(boxes) and ((boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any()):
        raise ValueError("source boxes must have positive width and height")
    if labels.shape != (len(boxes),):
        raise ValueError("labels must have shape (N,)")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError("labels must be integer class IDs")
    ids = tuple(str(index) for index in range(len(boxes))) if annotation_ids is None else tuple(map(str, annotation_ids))
    if len(ids) != len(boxes) or len(set(ids)) != len(ids):
        raise ValueError("source annotation IDs must be unique and match boxes")
    return boxes, labels.astype(np.int64), ids


def _objects_in_window(boxes, window, min_visible_fraction):
    if not len(boxes):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=bool), np.empty(0, dtype=bool)
    intersection = boxes.copy()
    intersection[:, 0] = np.maximum(intersection[:, 0], window.x1)
    intersection[:, 1] = np.maximum(intersection[:, 1], window.y1)
    intersection[:, 2] = np.minimum(intersection[:, 2], window.x2)
    intersection[:, 3] = np.minimum(intersection[:, 3], window.y2)
    widths = np.maximum(0.0, intersection[:, 2] - intersection[:, 0])
    heights = np.maximum(0.0, intersection[:, 3] - intersection[:, 1])
    original_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    visible = widths * heights / original_area
    intersects = visible > 0
    keep = visible >= min_visible_fraction
    clipped = keep & np.any(intersection != boxes, axis=1)
    local = intersection[keep] - np.asarray([window.x1, window.y1, window.x1, window.y1])
    return local, visible, keep, intersects & ~keep


def generate_training_patches(cube, boxes, labels, image_id, config, *, annotation_ids=None, split="train"):
    if split != "train":
        raise ValueError("P2 patch generation is training-only; use coordinate-restored sliced inference for evaluation")
    if not isinstance(config, PatchConfig):
        raise TypeError("config must be PatchConfig")
    array = validate_hyperspectral_cube(cube)
    boxes, labels, annotation_ids = _validate_annotations(boxes, labels, annotation_ids)
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("image_id must be a non-empty string")
    if len(boxes) and ((boxes[:, :2] < 0).any() or (boxes[:, 2] > array.shape[1]).any() or (boxes[:, 3] > array.shape[0]).any()):
        raise ValueError("training boxes must lie inside the source image")
    windows = generate_slice_windows(array.shape[:2], config.patch_shape, config.overlap)
    patches = []
    emitted_ids = []
    clipped_count = 0
    discarded_count = 0
    for window in windows:
        local_boxes, visible, keep, discarded = _objects_in_window(boxes, window, config.min_visible_fraction)
        clipped_count += int((keep & (visible < 1)).sum())
        discarded_count += int(discarded.sum())
        indexes = np.flatnonzero(keep)
        if not len(indexes) and not config.include_empty:
            continue
        patch_id = f"{image_id}__x{window.x1}_y{window.y1}_x{window.x2}_y{window.y2}"
        ids = tuple(annotation_ids[index] for index in indexes)
        emitted_ids.extend(ids)
        patches.append(
            PatchExample(
                patch_id,
                image_id,
                window,
                array[window.y1:window.y2, window.x1:window.x2, :].copy(),
                local_boxes,
                labels[indexes].copy(),
                ids,
                visible[indexes].copy(),
            )
        )
    patch_ids = [patch.patch_id for patch in patches]
    if len(patch_ids) != len(set(patch_ids)):
        raise RuntimeError("patch generation produced duplicate image IDs")
    duplicate_count = len(emitted_ids) - len(set(emitted_ids))
    return PatchGenerationResult(
        tuple(patches),
        len(boxes),
        len(emitted_ids),
        clipped_count,
        discarded_count,
        duplicate_count,
        config.min_visible_fraction,
    )
