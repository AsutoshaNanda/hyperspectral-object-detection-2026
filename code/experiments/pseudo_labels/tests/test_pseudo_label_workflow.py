import pytest
import torch

from Experiments.annotations.find_missing_candidates import DetectionCandidate
from Experiments.annotations.review_annotations import ReviewLabel, ReviewRecord
from Experiments.models.sam_auxiliary import SAMEvidence, build_sam_auxiliary_head
from Experiments.pseudo_labels.evaluate import audit_pseudo_label_quality
from Experiments.pseudo_labels.generate import TeacherCheckpoint, build_pl2_confirmed_label_set
from Experiments.pseudo_labels.teacher_student import build_teacher_student_plan, execute_teacher_student


def test_pl1_audits_hidden_known_gt_with_threshold_curve_and_rejects_teacher_leakage():
    teacher = TeacherCheckpoint.create("teacher-1", {"train-a", "train-b"})
    ground_truth = [{"image_id": "hidden", "category_id": 2, "bbox": [0, 0, 10, 10]}]
    predictions = [
        {"image_id": "hidden", "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.95},
        {"image_id": "hidden", "category_id": 3, "bbox": [20, 20, 30, 30], "score": 0.6},
    ]
    audit = audit_pseudo_label_quality(ground_truth, predictions, [0.5, 0.9], teacher, {"hidden"})
    low, high = audit.threshold_curve
    assert (low.precision, low.recall, low.class_accuracy, low.mean_box_iou) == (0.5, 1.0, 1.0, 1.0)
    assert (high.precision, high.recall, high.class_accuracy, high.mean_box_iou) == (1.0, 1.0, 1.0, 1.0)
    leaked = TeacherCheckpoint.create("leaked", {"train-a", "hidden"})
    with pytest.raises(ValueError, match="hidden audit"):
        audit_pseudo_label_quality(ground_truth, predictions, [0.5], leaked, {"hidden"})


def test_pl2_reuses_only_manually_confirmed_a1_labels():
    candidate = DetectionCandidate("c1", "train", 2, (1, 1, 5, 5), 0.99, 0.0, 0.9, "curve")
    reviewed = ReviewRecord("c1", ReviewLabel.CONFIRMED_OBJECT, "reviewer", "2026-09-20T00:00:00Z")
    result = build_pl2_confirmed_label_set([], [candidate], [reviewed], {"train"})
    assert result.confirmed_count == 1
    assert result.training_annotations[0]["source"] == "manually_confirmed_missing_object"


def test_pl3_executes_training_fold_teacher_student_path_with_provenance_guards():
    plan = build_teacher_student_plan({"a", "b", "c"}, {"a", "b"}, {"c"}, forbidden_image_ids={"test"})

    def train_teacher(image_ids):
        return TeacherCheckpoint.create("teacher-1", image_ids)

    def generate(teacher, image_ids):
        return [
            {
                "image_id": image_ids[0],
                "category_id": 1,
                "bbox_xyxy": [0, 0, 5, 5],
                "teacher_checkpoint_id": teacher.checkpoint_id,
            }
        ]

    def train_student(image_ids, pseudo_labels, confirmed_labels):
        assert set(image_ids) == {"a", "b", "c"}
        assert len(pseudo_labels) == 1
        assert len(confirmed_labels) == 1
        return "student-1"

    confirmed = [{"image_id": "a", "source": "manually_confirmed_missing_object"}]
    result = execute_teacher_student(
        plan,
        train_teacher,
        generate,
        train_student,
        confirmed_training_labels=confirmed,
    )
    assert result.student_artifact_id == "student-1"
    with pytest.raises(ValueError, match="test/ranking"):
        build_teacher_student_plan({"a", "test"}, {"a"}, {"test"}, forbidden_image_ids={"test"})


def test_sam_auxiliary_is_one_detector_head_and_refuses_independent_classifier_fusion():
    evidence = SAMEvidence.create("sam1-fold-0", True, {"a", "b"})
    head = build_sam_auxiliary_head(
        "SAM2",
        4,
        torch.eye(3),
        3,
        evidence,
        allowed_training_image_ids={"a", "b"},
    )
    features = torch.ones(2, 4)
    spectra = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    assert head(features, spectra).shape == (2, 3)
    with pytest.raises(ValueError, match="Independent classifier fusion"):
        head(features, spectra, independent_classifier_logits=torch.ones(2, 3))
    with pytest.raises(ValueError, match="inside one detector"):
        build_sam_auxiliary_head(
            "SAM2",
            4,
            torch.eye(3),
            3,
            evidence,
            allowed_training_image_ids={"a", "b"},
            fusion_mode="late_prediction_fusion",
        )
    with pytest.raises(ValueError, match="SAM3 requires"):
        build_sam_auxiliary_head(
            "SAM3",
            4,
            torch.eye(3),
            3,
            evidence,
            allowed_training_image_ids={"a", "b"},
        )
