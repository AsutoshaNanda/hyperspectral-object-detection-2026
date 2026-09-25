from dataclasses import asdict, dataclass
from enum import Enum


class ReviewLabel(str, Enum):
    CONFIRMED_OBJECT = "CONFIRMED_OBJECT"
    UNCERTAIN = "UNCERTAIN"
    NOT_OBJECT = "NOT_OBJECT"


@dataclass(frozen=True)
class ReviewRecord:
    candidate_id: str
    label: ReviewLabel
    reviewer: str
    reviewed_at: str
    note: str = ""

    def to_dict(self):
        row = asdict(self)
        row["label"] = self.label.value
        return row

    @classmethod
    def from_dict(cls, row):
        return cls(
            candidate_id=str(row["candidate_id"]),
            label=ReviewLabel(row["label"]),
            reviewer=str(row["reviewer"]),
            reviewed_at=str(row["reviewed_at"]),
            note=str(row.get("note", "")),
        )


def validate_reviews(candidate_ids, reviews, *, require_complete=True):
    expected = {str(value) for value in candidate_ids}
    parsed = [review if isinstance(review, ReviewRecord) else ReviewRecord.from_dict(review) for review in reviews]
    observed = [review.candidate_id for review in parsed]
    if len(observed) != len(set(observed)):
        raise ValueError("Each candidate may have only one review")
    unknown = sorted(set(observed) - expected)
    if unknown:
        raise ValueError(f"Reviews contain unknown candidates: {unknown}")
    if require_complete and set(observed) != expected:
        raise ValueError(f"Missing reviews for candidates: {sorted(expected - set(observed))}")
    for review in parsed:
        if not review.reviewer.strip() or not review.reviewed_at.strip():
            raise ValueError("Manual reviews require reviewer and reviewed_at")
    return {review.candidate_id: review for review in parsed}
