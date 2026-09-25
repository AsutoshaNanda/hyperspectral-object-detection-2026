from dataclasses import dataclass

import numpy as np

from Experiments.annotations.find_missing_candidates import record_bbox_xyxy
from Experiments.annotations.build_augmented_labels import build_training_copy_annotations


@dataclass(frozen=True)
class TeacherCheckpoint:
    checkpoint_id: str
    trained_image_ids: frozenset[str]
    data_scope: str = "training_fold"

    @classmethod
    def create(cls, checkpoint_id, trained_image_ids, data_scope="training_fold"):
        ids = frozenset(str(value) for value in trained_image_ids)
        if not checkpoint_id or not ids:
            raise ValueError("Teacher checkpoint id and trained image ids are required")
        return cls(str(checkpoint_id), ids, str(data_scope))


def validate_teacher_for_hidden_subset(teacher, hidden_image_ids, *, forbidden_image_ids=()):
    if not isinstance(teacher, TeacherCheckpoint):
        raise TypeError("teacher must include checkpoint training provenance")
    hidden = {str(value) for value in hidden_image_ids}
    forbidden = {str(value) for value in forbidden_image_ids}
    if not hidden:
        raise ValueError("Hidden known-GT subset must not be empty")
    overlap = teacher.trained_image_ids & hidden
    if overlap:
        raise ValueError(f"Teacher was trained on hidden audit images: {sorted(overlap)}")
    if teacher.trained_image_ids & forbidden:
        raise ValueError("Teacher provenance includes test/ranking images")
    if teacher.data_scope != "training_fold":
        raise ValueError("Teacher must be trained from training-fold data only")


def filter_pseudo_labels(
    predictions,
    threshold,
    source_image_ids,
    teacher,
    *,
    bbox_format="xyxy",
    forbidden_image_ids=(),
):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    source_ids = {str(value) for value in source_image_ids}
    forbidden = {str(value) for value in forbidden_image_ids}
    validate_teacher_for_hidden_subset(teacher, source_ids, forbidden_image_ids=forbidden)
    if source_ids & forbidden:
        raise ValueError("Pseudo-label sources include test/ranking images")
    labels = []
    for prediction in predictions:
        image_id = str(prediction["image_id"])
        if image_id not in source_ids:
            raise ValueError(f"Pseudo-label prediction is outside the declared source subset: {image_id}")
        confidence = float(prediction.get("score", prediction.get("confidence", np.nan)))
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Prediction confidence must be in [0, 1]")
        if confidence < threshold:
            continue
        labels.append(
            {
                "image_id": image_id,
                "category_id": int(prediction.get("category_id", prediction.get("class_id"))),
                "bbox_xyxy": list(record_bbox_xyxy(prediction, bbox_format)),
                "confidence": confidence,
                "teacher_checkpoint_id": teacher.checkpoint_id,
                "source": "teacher_pseudo_label",
            }
        )
    return labels


def build_pl2_confirmed_label_set(
    official_annotations,
    candidates,
    reviews,
    training_image_ids,
    *,
    validation_image_ids=(),
    holdout_image_ids=(),
    supports_ignore_regions=True,
    bbox_format="xywh",
):
    return build_training_copy_annotations(
        official_annotations,
        candidates,
        reviews,
        training_image_ids,
        validation_image_ids=validation_image_ids,
        holdout_image_ids=holdout_image_ids,
        supports_ignore_regions=supports_ignore_regions,
        bbox_format=bbox_format,
    )
