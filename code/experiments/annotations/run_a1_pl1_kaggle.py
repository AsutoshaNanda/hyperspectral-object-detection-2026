import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import hsi_runner as runner
from Experiments.annotations.find_missing_candidates import (
    discover_candidates,
    export_candidate_package,
    select_validation_threshold,
)
from Experiments.annotations.review_annotations import ReviewLabel
from Experiments.pseudo_labels.evaluate import audit_pseudo_label_quality
from Experiments.pseudo_labels.generate import TeacherCheckpoint


EXPECTED_CHECKPOINT_SHA256 = "b4a54b0f5ad05a25fae96daf070e53286852f15c7091dd3fa13af2113afb9012"
EXPECTED_SPLIT_SHA256 = "15e4bbf69709dcd91a0868f06fc19bfb4c88434efd9e51afd3bd905479e1979b"
EXPECTED_HSI_RUNNER_SHA256 = "48075c8edd26ecd78f509c8fb79f0ea9d66f4fb335145a53fc72afe0643dd1ac"
TARGET_PRECISION = 0.95
LOW_IOU_THRESHOLD = 0.1
INFERENCE_FLOOR = 0.001
CONFIDENCE_THRESHOLDS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.125,
    0.15,
    0.175,
    0.2,
    0.225,
    0.25,
    0.275,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.825,
    0.85,
    0.875,
    0.9,
    0.925,
    0.95,
    0.96,
    0.97,
    0.98,
    0.985,
    0.99,
    0.995,
    0.999,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_one(root, relative_pattern, *, directory=False):
    matches = [path for path in Path(root).rglob(relative_pattern) if path.is_dir() == directory]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {relative_pattern}, found {len(matches)}: {matches}")
    return matches[0]


def verify_plan_c_artifacts(checkpoint, split_path, data_audit_path, run_config_path):
    checkpoint_hash = sha256(checkpoint)
    split_hash = sha256(split_path)
    runner_hash = sha256(runner.__file__)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Plan-C checkpoint hash mismatch: {checkpoint_hash}")
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"Plan-C legacy split hash mismatch: {split_hash}")
    if runner_hash != EXPECTED_HSI_RUNNER_SHA256:
        raise ValueError(f"Plan-C hsi_runner hash mismatch: {runner_hash}")
    split = json.loads(Path(split_path).read_text())
    audit = json.loads(Path(data_audit_path).read_text())
    config = json.loads(Path(run_config_path).read_text())
    validate_plan_c_provenance(split, audit, config)
    return split, audit, config, {
        "checkpoint_sha256": checkpoint_hash,
        "split_sha256": split_hash,
        "hsi_runner_sha256": runner_hash,
        "data_audit_sha256": sha256(data_audit_path),
        "run_config_sha256": sha256(run_config_path),
    }


def validate_plan_c_provenance(split, audit, config):
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Plan-C split must contain exactly train, val, and test")
    split_sets = {name: {str(value) for value in values} for name, values in split.items()}
    if any(not values for values in split_sets.values()):
        raise ValueError("Plan-C split partitions must be non-empty")
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"]:
        raise ValueError("Plan-C split partitions overlap")
    audited = {name: [str(value) for value in audit["split_ids"][name]] for name in split}
    expected = {name: [str(value) for value in split[name]] for name in split}
    if audited != expected or audit.get("split_mode") != "legacy":
        raise ValueError("Saved data audit does not prove the legacy split used by Plan-C")
    args = config.get("args", {})
    required = {
        "checkpoint": None,
        "family": "rtdetr",
        "input_policy": "square_pad",
        "seed": 42,
        "stem_init": "random",
    }
    if any(args.get(key) != value for key, value in required.items()):
        raise ValueError("Plan-C run configuration does not match the audited training run")
    if config.get("resolved_input_policy") != "square_pad":
        raise ValueError("Plan-C resolved input policy must be square_pad")
    low = np.asarray(audit.get("band_low_0_5pct"), dtype=np.float64)
    high = np.asarray(audit.get("band_high_99_5pct"), dtype=np.float64)
    if low.shape != (16,) or high.shape != (16,) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Plan-C audit must contain finite 16-band normalization limits")
    if np.any(high <= low):
        raise ValueError("Plan-C normalization upper limits must exceed lower limits")


def official_annotations(image_ids, xml_by_stem, class_to_id):
    annotations = []
    for stem in sorted(image_ids):
        width, height, rows = runner.parse_voc(xml_by_stem[stem], class_to_id)
        for index, (category_id, cx, cy, box_width, box_height) in enumerate(rows):
            pixel_width = box_width * width
            pixel_height = box_height * height
            annotations.append(
                {
                    "id": f"{stem}:{index}",
                    "image_id": stem,
                    "category_id": int(category_id),
                    "bbox": [
                        cx * width - pixel_width / 2,
                        cy * height - pixel_height / 2,
                        pixel_width,
                        pixel_height,
                    ],
                }
            )
    return annotations


def prepare_images(image_ids, cube_by_stem, xml_by_stem, class_to_id, low, high, output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    geometries = {}
    for index, stem in enumerate(sorted(image_ids), 1):
        cube = runner.read_cube(cube_by_stem[stem])
        width, height, _ = runner.parse_voc(xml_by_stem[stem], class_to_id)
        if cube.shape[:2] != (height, width):
            raise ValueError(f"Cube/XML geometry mismatch for {stem}")
        geometries[stem] = runner.write_tiff(
            cube,
            output_directory / f"{stem}.tiff",
            low,
            high,
            "square_pad",
        )
        if index % 250 == 0:
            print(f"Prepared {index}/{len(image_ids)} images", flush=True)
    return geometries


def infer(model, prepared_directory, geometries, device, confidence):
    results = model.predict(
        source=str(prepared_directory),
        imgsz=640,
        conf=float(confidence),
        iou=0.65,
        max_det=300,
        batch=2,
        device=device,
        stream=True,
        verbose=False,
    )
    predictions = []
    seen = set()
    for result in results:
        stem = Path(result.path).stem
        if stem not in geometries:
            raise ValueError(f"Prediction returned undeclared image {stem}")
        seen.add(stem)
        if result.boxes is None:
            continue
        boxes = runner.restore_boxes(result.boxes.xyxy.detach().cpu().numpy(), geometries[stem])
        scores = result.boxes.conf.detach().cpu().numpy()
        labels = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (float(value) for value in box)
            if x2 <= x1 or y2 <= y1:
                continue
            predictions.append(
                {
                    "image_id": stem,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2, y2],
                    "score": float(score),
                }
            )
    if seen != set(geometries):
        raise RuntimeError(f"Inference coverage mismatch: missing={sorted(set(geometries) - seen)}")
    return predictions


class LazyCubes(Mapping):
    def __init__(self, paths, allowed_image_ids):
        self.paths = paths
        self.allowed = frozenset(str(value) for value in allowed_image_ids)

    def __getitem__(self, key):
        key = str(key)
        if key not in self.allowed:
            raise KeyError(key)
        return runner.read_cube(self.paths[key])

    def __iter__(self):
        return iter(self.allowed)

    def __len__(self):
        return len(self.allowed)


def write_review_files(output_directory, candidates):
    output_directory = Path(output_directory)
    schema = {
        "allowed_labels": [label.value for label in ReviewLabel],
        "required_fields": ["candidate_id", "label", "reviewer", "reviewed_at"],
        "optional_fields": ["note"],
        "policy": {
            "CONFIRMED_OBJECT": "eligible for a training-copy label only after manual review",
            "UNCERTAIN": "ignore region or excluded targeted hard-negative crop",
            "NOT_OBJECT": "may remain background",
        },
    }
    template = [
        {"candidate_id": item.candidate_id, "label": None, "reviewer": "", "reviewed_at": "", "note": ""}
        for item in candidates
    ]
    (output_directory / "review_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    (output_directory / "annotation_reviews.json").write_text(json.dumps(template, indent=2) + "\n")


def execute(input_root, output_directory, cache_directory, device="0"):
    started = time.monotonic()
    input_root = Path(input_root)
    output_directory = Path(output_directory)
    cache_directory = Path(cache_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    cache_directory.mkdir(parents=True, exist_ok=False)
    base = find_one(input_root, "hyperspectral-2026", directory=True)
    split_path = find_one(input_root, "hsi_plan_c_ratio/split_manifest.json")
    plan_root = split_path.parent
    checkpoint = plan_root / "runs/rtdetr_ratio_control/weights/best.pt"
    split, data_audit, run_config, hashes = verify_plan_c_artifacts(
        checkpoint,
        split_path,
        plan_root / "data_audit.json",
        plan_root / "run_config.json",
    )
    train_ids = tuple(str(value) for value in split["train"])
    validation_ids = tuple(str(value) for value in split["val"])
    holdout_ids = tuple(str(value) for value in split["test"])
    cube_root = base / "data_train/data_train/VIS"
    xml_root = base / "data_train/data_train/Annotations/VIS"
    names = runner.read_classes(base / "class.txt")
    class_to_id = {name: index for index, name in enumerate(names)}
    cube_by_stem = runner.unique_stems(runner.collect_cubes(cube_root))
    xml_by_stem = runner.annotation_map(xml_root)
    requested = set(train_ids) | set(validation_ids)
    if not requested <= set(cube_by_stem) or not requested <= set(xml_by_stem):
        raise ValueError("Base data is missing Plan-C train/validation images")
    low = np.asarray(data_audit["band_low_0_5pct"], dtype=np.float64)
    high = np.asarray(data_audit["band_high_99_5pct"], dtype=np.float64)
    validation_directory = cache_directory / "legacy_validation"
    validation_geometry = prepare_images(
        validation_ids,
        cube_by_stem,
        xml_by_stem,
        class_to_id,
        low,
        high,
        validation_directory,
    )
    from ultralytics import RTDETR

    model = RTDETR(str(checkpoint))
    validation_predictions = infer(model, validation_directory, validation_geometry, device, INFERENCE_FLOOR)
    validation_truth = official_annotations(validation_ids, xml_by_stem, class_to_id)
    teacher = TeacherCheckpoint.create(hashes["checkpoint_sha256"], train_ids)
    audit = audit_pseudo_label_quality(
        validation_truth,
        validation_predictions,
        CONFIDENCE_THRESHOLDS,
        teacher,
        validation_ids,
        iou_threshold=0.5,
        forbidden_image_ids=holdout_ids,
    )
    curve = []
    for row in audit.threshold_curve:
        item = row.to_dict()
        item["source_split"] = "validation"
        curve.append(item)
    curve_id = f"plan-c-legacy-val-{hashes['checkpoint_sha256'][:12]}"
    selection = select_validation_threshold(curve, TARGET_PRECISION, curve_id)
    pl1_report = {
        **audit.to_dict(),
        "threshold_curve": curve,
        "selected_threshold": {
            "threshold": selection.threshold,
            "precision": selection.precision,
            "recall": selection.recall,
            "target_precision": selection.target_precision,
            "selection_rule": "maximum recall subject to measured validation precision >= 0.95",
            "curve_id": curve_id,
        },
        "hidden_gt_scope": "legacy_validation",
        "teacher_fit_scope": "legacy_train_only",
    }
    (output_directory / "pl1_threshold_curve.json").write_text(
        json.dumps(pl1_report, indent=2, allow_nan=False) + "\n"
    )
    training_directory = cache_directory / "legacy_training"
    training_geometry = prepare_images(
        train_ids,
        cube_by_stem,
        xml_by_stem,
        class_to_id,
        low,
        high,
        training_directory,
    )
    training_predictions = infer(model, training_directory, training_geometry, device, selection.threshold)
    training_truth = official_annotations(train_ids, xml_by_stem, class_to_id)
    candidates = discover_candidates(
        training_predictions,
        training_truth,
        train_ids,
        selection,
        LOW_IOU_THRESHOLD,
        prediction_bbox_format="xyxy",
        annotation_bbox_format="xywh",
    )
    package = export_candidate_package(
        candidates,
        LazyCubes(cube_by_stem, train_ids),
        output_directory,
    )
    write_review_files(output_directory, candidates)
    provenance = {
        "status": "measured_private_diagnostic",
        "checkpoint": str(checkpoint),
        **hashes,
        "checkpoint_training_scope": "legacy_train_only",
        "pl1_hidden_gt_scope": "legacy_validation",
        "a1_inference_scope": "legacy_train_only",
        "legacy_train_images": len(train_ids),
        "legacy_validation_images": len(validation_ids),
        "legacy_holdout_images_loaded": 0,
        "competition_test_images_loaded": 0,
        "competition_ranking_images_loaded": 0,
        "submission_created": False,
        "official_labels_modified": False,
        "candidate_iou_rule": "IoU < 0.1 to every official GT box",
        "candidate_count": package.candidate_count,
        "selected_confidence_threshold": selection.threshold,
        "target_validation_precision": TARGET_PRECISION,
        "confidence_thresholds_predeclared": list(CONFIDENCE_THRESHOLDS),
        "runtime_seconds": time.monotonic() - started,
        "runner_sha256": sha256(__file__),
    }
    (output_directory / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps(provenance, indent=2, allow_nan=False))
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/a1_pl1_diagnostic")
    parser.add_argument("--cache", default="/kaggle/temp/a1_pl1_diagnostic")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    execute(args.input_root, args.output, args.cache, args.device)


if __name__ == "__main__":
    main()
