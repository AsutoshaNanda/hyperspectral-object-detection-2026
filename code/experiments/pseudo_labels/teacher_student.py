from dataclasses import dataclass

from .generate import TeacherCheckpoint


@dataclass(frozen=True)
class TeacherStudentPlan:
    training_fold_image_ids: frozenset[str]
    teacher_train_image_ids: frozenset[str]
    pseudo_source_image_ids: frozenset[str]
    student_train_image_ids: frozenset[str]
    forbidden_image_ids: frozenset[str]


@dataclass(frozen=True)
class TeacherStudentResult:
    teacher_checkpoint_id: str
    student_artifact_id: str
    pseudo_label_count: int
    confirmed_label_count: int
    data_scope: str = "training_fold"


def build_teacher_student_plan(
    training_fold_image_ids,
    teacher_train_image_ids,
    pseudo_source_image_ids,
    *,
    forbidden_image_ids=(),
):
    fold = frozenset(str(value) for value in training_fold_image_ids)
    teacher = frozenset(str(value) for value in teacher_train_image_ids)
    pseudo = frozenset(str(value) for value in pseudo_source_image_ids)
    forbidden = frozenset(str(value) for value in forbidden_image_ids)
    if not fold or not teacher or not pseudo:
        raise ValueError("Fold, teacher-training, and pseudo-source image sets must be non-empty")
    if teacher & pseudo:
        raise ValueError("Teacher training images must exclude the pseudo-label source subset")
    if not teacher <= fold or not pseudo <= fold:
        raise ValueError("Teacher/student data must remain inside the training fold")
    if fold & forbidden:
        raise ValueError("Training fold contains test/ranking images")
    return TeacherStudentPlan(fold, teacher, pseudo, teacher | pseudo, forbidden)


def execute_teacher_student(
    plan,
    train_teacher,
    generate_pseudo_labels,
    train_student,
    *,
    confirmed_training_labels=(),
):
    if not isinstance(plan, TeacherStudentPlan):
        raise TypeError("plan must be a TeacherStudentPlan")
    teacher = train_teacher(tuple(sorted(plan.teacher_train_image_ids)))
    if not isinstance(teacher, TeacherCheckpoint):
        raise TypeError("train_teacher must return TeacherCheckpoint provenance")
    if teacher.trained_image_ids != plan.teacher_train_image_ids or teacher.data_scope != "training_fold":
        raise ValueError("Teacher checkpoint provenance does not match the leakage-safe plan")
    pseudo_labels = list(generate_pseudo_labels(teacher, tuple(sorted(plan.pseudo_source_image_ids))))
    for row in pseudo_labels:
        if str(row["image_id"]) not in plan.pseudo_source_image_ids:
            raise ValueError("Generated pseudo-label lies outside the training-fold source subset")
        if row.get("teacher_checkpoint_id") != teacher.checkpoint_id:
            raise ValueError("Pseudo-label teacher provenance does not match the trained teacher")
    confirmed = list(confirmed_training_labels)
    for row in confirmed:
        if row.get("source") != "manually_confirmed_missing_object":
            raise ValueError("PL2 labels must reuse manually confirmed A1 labels")
        if str(row["image_id"]) not in plan.student_train_image_ids:
            raise ValueError("Confirmed label lies outside the student training fold")
    student_artifact_id = train_student(
        tuple(sorted(plan.student_train_image_ids)),
        tuple(pseudo_labels),
        tuple(confirmed),
    )
    if not student_artifact_id:
        raise ValueError("train_student must return a student artifact id")
    return TeacherStudentResult(
        teacher.checkpoint_id,
        str(student_artifact_id),
        len(pseudo_labels),
        len(confirmed),
    )
