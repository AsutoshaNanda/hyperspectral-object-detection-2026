from dataclasses import dataclass

import numpy as np

from .common import MergeResult, PredictionBatch, box_iou_one_to_many


COMPATIBLE_FAMILIES = {"yolo", "cascade_rcnn"}


@dataclass(frozen=True)
class SuppressionResult:
    method: str
    result: MergeResult
    detector_family: str
    iou_threshold: float
    score_threshold: float
    sigma: float


def normalize_detector_family(detector_family):
    if not isinstance(detector_family, str):
        raise TypeError("detector_family must be a string")
    normalized = detector_family.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"cascade", "cascadercnn", "cascade_r_cnn"}:
        normalized = "cascade_rcnn"
    if normalized.startswith("yolo"):
        normalized = "yolo"
    return normalized


def validate_soft_nms_compatibility(predictions, detector_family, *, exposes_raw_presuppression=True):
    family = normalize_detector_family(detector_family)
    if family not in COMPATIBLE_FAMILIES:
        raise ValueError("Soft-NMS is implemented only for NMS-based YOLO and Cascade R-CNN paths")
    if exposes_raw_presuppression is not True:
        raise ValueError("Soft-NMS requires raw pre-suppression predictions; this detector path does not expose them")
    if predictions.prediction_stage != "raw_pre_suppression":
        raise ValueError("Soft-NMS requires PredictionBatch(prediction_stage='raw_pre_suppression')")
    return family


def _validate_parameters(method, iou_threshold, score_threshold, sigma, max_detections):
    if method not in {"normal", "linear", "gaussian"}:
        raise ValueError("method must be normal, linear or gaussian")
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between zero and one")
    if not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between zero and one")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if max_detections is not None and (
        not isinstance(max_detections, int) or isinstance(max_detections, bool) or max_detections < 1
    ):
        raise ValueError("max_detections must be a positive integer")


def _suppress_group(boxes, scores, indexes, method, iou_threshold, score_threshold, sigma):
    working_scores = scores.copy()
    remaining = indexes[working_scores[indexes] >= score_threshold].tolist()
    selected = []
    while remaining:
        best_position = max(range(len(remaining)), key=lambda position: (working_scores[remaining[position]], -position))
        current = remaining.pop(best_position)
        selected.append(current)
        if not remaining:
            continue
        remaining_array = np.asarray(remaining, dtype=np.int64)
        overlaps = box_iou_one_to_many(boxes[current], boxes[remaining_array])
        if method == "normal":
            weights = (overlaps <= iou_threshold).astype(np.float64)
        elif method == "linear":
            weights = np.where(overlaps > iou_threshold, 1.0 - overlaps, 1.0)
        else:
            weights = np.exp(-(overlaps * overlaps) / sigma)
        working_scores[remaining_array] *= weights
        remaining = [index for index in remaining if working_scores[index] >= score_threshold]
    return selected, working_scores


def suppress_predictions(
    predictions,
    *,
    detector_family,
    method,
    iou_threshold=0.5,
    score_threshold=0.001,
    sigma=0.5,
    class_agnostic=False,
    max_detections=None,
    exposes_raw_presuppression=True,
):
    if not isinstance(predictions, PredictionBatch):
        raise TypeError("predictions must be PredictionBatch")
    family = validate_soft_nms_compatibility(
        predictions, detector_family, exposes_raw_presuppression=exposes_raw_presuppression
    )
    _validate_parameters(method, iou_threshold, score_threshold, sigma, max_detections)
    scores = predictions.scores.copy()
    selected = []
    groups = [np.arange(len(predictions))] if class_agnostic else [
        np.flatnonzero(predictions.labels == label) for label in np.unique(predictions.labels)
    ]
    for group in groups:
        group_selected, updated_scores = _suppress_group(
            predictions.boxes,
            scores,
            group,
            method,
            iou_threshold,
            score_threshold,
            sigma,
        )
        scores[group] = updated_scores[group]
        selected.extend(group_selected)
    selected = sorted(selected, key=lambda index: (-scores[index], index))
    if max_detections is not None:
        selected = selected[:max_detections]
    indexes = np.asarray(selected, dtype=np.int64)
    output = PredictionBatch(
        predictions.boxes[indexes],
        scores[indexes],
        predictions.labels[indexes],
        predictions.checkpoint_id,
        f"{family}:{method}_nms",
        "post_suppression",
    )
    merge = MergeResult(output, len(predictions), len(output), len(predictions) - len(output))
    return SuppressionResult(method, merge, family, iou_threshold, score_threshold, sigma)


def compare_suppression(
    predictions,
    *,
    detector_family,
    iou_threshold=0.5,
    score_threshold=0.001,
    sigma=0.5,
    class_agnostic=False,
    max_detections=None,
    exposes_raw_presuppression=True,
):
    return {
        method: suppress_predictions(
            predictions,
            detector_family=detector_family,
            method=method,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            sigma=sigma,
            class_agnostic=class_agnostic,
            max_detections=max_detections,
            exposes_raw_presuppression=exposes_raw_presuppression,
        )
        for method in ("normal", "linear", "gaussian")
    }
