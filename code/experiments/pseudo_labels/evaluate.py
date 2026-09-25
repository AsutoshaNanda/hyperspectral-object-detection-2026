from dataclasses import asdict, dataclass

import numpy as np

from Experiments.annotations.find_missing_candidates import box_iou_xyxy, record_bbox_xyxy
from .generate import TeacherCheckpoint, validate_teacher_for_hidden_subset


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    class_accuracy: float
    mean_box_iou: float
    predictions: int
    ground_truth: int
    matched: int

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PseudoLabelAudit:
    teacher_checkpoint_id: str
    hidden_image_ids: tuple[str, ...]
    iou_threshold: float
    threshold_curve: tuple[ThresholdMetrics, ...]

    def to_dict(self):
        return {
            "teacher_checkpoint_id": self.teacher_checkpoint_id,
            "hidden_image_ids": list(self.hidden_image_ids),
            "iou_threshold": self.iou_threshold,
            "threshold_curve": [row.to_dict() for row in self.threshold_curve],
        }


def _match(predictions, truths, iou_threshold):
    predictions_by_image = {}
    truths_by_image = {}
    for row in predictions:
        predictions_by_image.setdefault(str(row["image_id"]), []).append(row)
    for row in truths:
        truths_by_image.setdefault(str(row["image_id"]), []).append(row)
    matches = []
    for image_id in sorted(set(predictions_by_image) | set(truths_by_image)):
        image_predictions = sorted(predictions_by_image.get(image_id, ()), key=lambda row: -row["score"])
        image_truths = truths_by_image.get(image_id, ())
        unmatched = set(range(len(image_truths)))
        for prediction in image_predictions:
            options = [
                (box_iou_xyxy(prediction["bbox_xyxy"], image_truths[index]["bbox_xyxy"]), index)
                for index in unmatched
            ]
            if not options:
                continue
            iou, truth_index = max(options)
            if iou >= iou_threshold:
                unmatched.remove(truth_index)
                matches.append((prediction, image_truths[truth_index], iou))
    return matches


def audit_pseudo_label_quality(
    hidden_ground_truth,
    predictions,
    thresholds,
    teacher: TeacherCheckpoint,
    hidden_image_ids,
    *,
    iou_threshold=0.5,
    ground_truth_bbox_format="xywh",
    prediction_bbox_format="xyxy",
    forbidden_image_ids=(),
):
    hidden_ids = {str(value) for value in hidden_image_ids}
    validate_teacher_for_hidden_subset(teacher, hidden_ids, forbidden_image_ids=forbidden_image_ids)
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    threshold_values = sorted({float(value) for value in thresholds})
    if not threshold_values or any(not 0 <= value <= 1 for value in threshold_values):
        raise ValueError("thresholds must contain values in [0, 1]")
    truths = []
    for row in hidden_ground_truth:
        image_id = str(row["image_id"])
        if image_id not in hidden_ids:
            raise ValueError("Ground truth contains images outside the controlled hidden subset")
        truths.append(
            {
                "image_id": image_id,
                "category_id": int(row["category_id"]),
                "bbox_xyxy": record_bbox_xyxy(row, ground_truth_bbox_format),
            }
        )
    normalized_predictions = []
    for row in predictions:
        image_id = str(row["image_id"])
        if image_id not in hidden_ids:
            raise ValueError("Predictions contain images outside the controlled hidden subset")
        score = float(row.get("score", row.get("confidence", np.nan)))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction confidence must be in [0, 1]")
        normalized_predictions.append(
            {
                "image_id": image_id,
                "category_id": int(row.get("category_id", row.get("class_id"))),
                "bbox_xyxy": record_bbox_xyxy(row, prediction_bbox_format),
                "score": score,
            }
        )
    curve = []
    for threshold in threshold_values:
        selected = [row for row in normalized_predictions if row["score"] >= threshold]
        matches = _match(selected, truths, iou_threshold)
        matched = len(matches)
        precision = matched / len(selected) if selected else 0.0
        recall = matched / len(truths) if truths else 0.0
        class_accuracy = (
            sum(prediction["category_id"] == truth["category_id"] for prediction, truth, _ in matches) / matched
            if matched
            else 0.0
        )
        mean_iou = float(np.mean([iou for _, _, iou in matches])) if matches else 0.0
        curve.append(
            ThresholdMetrics(
                threshold,
                precision,
                recall,
                class_accuracy,
                mean_iou,
                len(selected),
                len(truths),
                matched,
            )
        )
    return PseudoLabelAudit(teacher.checkpoint_id, tuple(sorted(hidden_ids)), iou_threshold, tuple(curve))
