import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass

from .find_missing_candidates import DetectionCandidate
from .review_annotations import ReviewLabel, validate_reviews


@dataclass(frozen=True)
class AugmentedLabelSet:
    training_annotations: tuple[dict, ...]
    ignore_regions: tuple[dict, ...]
    excluded_hard_negative_regions: tuple[dict, ...]
    confirmed_count: int
    uncertain_count: int
    protected_annotations_digest: str


def _digest(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _bbox(candidate, bbox_format):
    x1, y1, x2, y2 = candidate.bbox_xyxy
    if bbox_format == "xyxy":
        return [x1, y1, x2, y2]
    if bbox_format == "xywh":
        return [x1, y1, x2 - x1, y2 - y1]
    raise ValueError("bbox_format must be xyxy or xywh")


def build_training_copy_annotations(
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
    training_ids = {str(value) for value in training_image_ids}
    protected_ids = {str(value) for value in validation_image_ids} | {str(value) for value in holdout_image_ids}
    if not training_ids or training_ids & protected_ids:
        raise ValueError("Training images must be non-empty and disjoint from validation/holdout")
    official = deepcopy(list(official_annotations))
    protected = [row for row in official if str(row["image_id"]) in protected_ids]
    training = [row for row in official if str(row["image_id"]) in training_ids]
    candidates = tuple(candidates)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Candidate ids must be unique")
    review_by_id = validate_reviews(candidate_by_id, reviews)
    next_numeric_id = max(
        [int(row["id"]) for row in official if isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool)],
        default=0,
    )
    ignore_regions = []
    exclusions = []
    confirmed_count = 0
    uncertain_count = 0
    for candidate_id, candidate in candidate_by_id.items():
        if not isinstance(candidate, DetectionCandidate):
            raise TypeError("candidates must contain DetectionCandidate records")
        if candidate.image_id not in training_ids or candidate.image_id in protected_ids:
            raise ValueError("Reviewed candidates may only modify training-copy labels")
        review = review_by_id[candidate_id]
        if review.label == ReviewLabel.CONFIRMED_OBJECT:
            next_numeric_id += 1
            training.append(
                {
                    "id": next_numeric_id,
                    "image_id": candidate.image_id,
                    "category_id": candidate.category_id,
                    "bbox": _bbox(candidate, bbox_format),
                    "source": "manually_confirmed_missing_object",
                    "candidate_id": candidate_id,
                }
            )
            confirmed_count += 1
        elif review.label == ReviewLabel.UNCERTAIN:
            region = {
                "image_id": candidate.image_id,
                "bbox_xyxy": list(candidate.bbox_xyxy),
                "candidate_id": candidate_id,
            }
            (ignore_regions if supports_ignore_regions else exclusions).append(region)
            uncertain_count += 1
    if _digest(protected) != _digest([row for row in official if str(row["image_id"]) in protected_ids]):
        raise RuntimeError("Protected validation/holdout annotations changed")
    return AugmentedLabelSet(
        tuple(training),
        tuple(ignore_regions),
        tuple(exclusions),
        confirmed_count,
        uncertain_count,
        _digest(protected),
    )
