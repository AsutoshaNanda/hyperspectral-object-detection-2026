from collections import defaultdict
from pathlib import Path
import hashlib
import json
import time

import numpy as np

import hsi_runner as runner
from evaluate_size_ap import detection_confusion, evaluate_size_ap
from inference_common import PredictionBatch, merge_predictions
from inference_multiscale import MultiScalePlan
from inference_sliced import generate_slice_windows, merge_sliced_predictions
from inference_tta import horizontal_flip_tta


OUT = Path("/kaggle/working/inference_screen")
OUT.mkdir(parents=True, exist_ok=True)
SCALES = (512, 640, 768)
MERGE_IOU = 0.65
SLICE_SHAPE = (256, 256)
SLICE_OVERLAP = (0.25, 0.25)


def one(root, pattern):
    matches = list(Path(root).rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {len(matches)}")
    return matches[0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empty(checkpoint, source):
    return PredictionBatch(np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), checkpoint, source)


def batch_from_result(result, checkpoint, geometry, source):
    if result.boxes is None or len(result.boxes) == 0:
        return empty(checkpoint, source)
    boxes = runner.restore_boxes(result.boxes.xyxy.cpu().numpy(), geometry)
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return PredictionBatch(boxes[valid], scores[valid], labels[valid], checkpoint, source)


def predict_directory(model, directory, geometries, checkpoint, imgsz, source):
    output = {}
    for result in model.predict(source=str(directory), imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                                device="0", stream=True, verbose=False):
        stem = Path(result.path).stem
        output[stem] = batch_from_result(result, checkpoint, geometries[stem], source)
    if set(output) != set(geometries):
        raise RuntimeError(f"Incomplete inference for {source}")
    return output


def restore_flip(batch, width):
    if not len(batch):
        return batch
    boxes = batch.boxes.copy()
    left = width - boxes[:, 2]
    right = width - boxes[:, 0]
    boxes[:, 0], boxes[:, 2] = left, right
    return PredictionBatch(boxes, batch.scores, batch.labels, batch.checkpoint_id, batch.source + ":restored")


def coco_predictions(batches):
    rows = []
    for stem, batch in batches.items():
        for box, score, label in zip(batch.boxes, batch.scores, batch.labels):
            x1, y1, x2, y2 = map(float, box)
            rows.append({"image_id": int(stem), "category_id": int(label), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    return rows


def evaluate_policy(gt, batches, name):
    predictions = coco_predictions(batches)
    report = evaluate_size_ap(gt, predictions, max_dets=300)
    report["confusion_at_0_25_iou_0_5"] = detection_confusion(gt, predictions, confidence_threshold=0.25, iou_threshold=0.5)
    report["policy"] = name
    report["prediction_count"] = len(predictions)
    (OUT / f"{name}_predictions.json").write_text(json.dumps(predictions, allow_nan=False))
    (OUT / f"{name}_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    started = time.monotonic()
    data = one("/kaggle/input", "hyperspectral-2026")
    plan = one("/kaggle/input", "hsi_plan_c_ratio")
    weights = one(plan, "best.pt")
    expected_hash = "b4a54b0f5ad05a25fae96daf070e53286852f15c7091dd3fa13af2113afb9012"
    if sha(weights) != expected_hash:
        raise RuntimeError("Plan-C checkpoint hash mismatch")
    audit = json.loads((plan / "data_audit.json").read_text())
    split = json.loads((plan / "split_manifest.json").read_text())
    lo = np.asarray(audit["band_low_0_5pct"], dtype=np.float32)
    hi = np.asarray(audit["band_high_99_5pct"], dtype=np.float32)
    names = runner.read_classes(data / "class.txt")
    class_to_id = {name: index for index, name in enumerate(names)}
    cubes = runner.unique_stems(runner.collect_cubes(data / "data_train/data_train/VIS"))
    xmls = runner.annotation_map(data / "data_train/data_train/Annotations/VIS")
    normal_dir = Path("/kaggle/temp/inference_screen/normal")
    flip_dir = Path("/kaggle/temp/inference_screen/flip")
    slice_dir = Path("/kaggle/temp/inference_screen/slices")
    for directory in (normal_dir, flip_dir, slice_dir):
        directory.mkdir(parents=True, exist_ok=False)
    geometries = {}
    flip_geometries = {}
    images = []
    annotations = []
    original_shapes = {}
    cube_cache = {}
    for stem in split["val"]:
        cube = runner.read_cube(cubes[stem])
        width, height, rows = runner.parse_voc(xmls[stem], class_to_id)
        original_shapes[stem] = (height, width)
        cube_cache[stem] = cube
        images.append({"id": int(stem), "width": width, "height": height})
        for label, cx, cy, bw, bh in rows:
            bbox = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
            annotations.append({"id": len(annotations) + 1, "image_id": int(stem), "category_id": int(label),
                                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
        geometries[stem] = runner.write_tiff(cube, normal_dir / f"{stem}.tiff", lo, hi, "square_pad")
        flip_geometries[stem] = runner.write_tiff(np.flip(cube, axis=1).copy(), flip_dir / f"{stem}.tiff", lo, hi, "square_pad")
    gt = {"images": images, "annotations": annotations, "categories": [{"id": i, "name": name} for i, name in enumerate(names)]}
    (OUT / "ground_truth_original.json").write_text(json.dumps(gt, indent=2))
    from ultralytics import RTDETR
    model = RTDETR(str(weights))
    normal_by_scale = {}
    flip_raw_by_scale = {}
    for scale in SCALES:
        normal_by_scale[scale] = predict_directory(model, normal_dir, geometries, expected_hash, scale, f"normal_{scale}")
        flip_raw_by_scale[scale] = predict_directory(model, flip_dir, flip_geometries, expected_hash, scale, f"flip_{scale}")
    baseline = normal_by_scale[640]
    t1 = {stem: horizontal_flip_tta(baseline[stem], flip_raw_by_scale[640][stem], original_shapes[stem][1],
                                    iou_threshold=MERGE_IOU, max_detections=300).predictions for stem in baseline}
    plan_scales = MultiScalePlan(tuple(float(scale) for scale in SCALES), "512_640_768")
    t2 = {stem: merge_predictions([normal_by_scale[scale][stem] for scale in SCALES], iou_threshold=MERGE_IOU,
                                  max_detections=300, source=f"T2:{plan_scales.policy_id}").predictions for stem in baseline}
    t3 = {stem: merge_predictions(
        [batch for scale in SCALES for batch in (normal_by_scale[scale][stem], restore_flip(flip_raw_by_scale[scale][stem], original_shapes[stem][1]))],
        iou_threshold=MERGE_IOU, max_detections=300, source=f"T3:{plan_scales.policy_id}"
    ).predictions for stem in baseline}
    slice_geometries = {}
    slice_metadata = {}
    windows_by_stem = {}
    for stem, cube in cube_cache.items():
        windows = generate_slice_windows(cube.shape[:2], SLICE_SHAPE, SLICE_OVERLAP)
        windows_by_stem[stem] = windows
        for window in windows:
            key = f"{stem}__s{window.index}"
            crop = cube[window.y1:window.y2, window.x1:window.x2]
            slice_geometries[key] = runner.write_tiff(crop, slice_dir / f"{key}.tiff", lo, hi, "square_pad")
            slice_metadata[key] = (stem, window)
    raw_slices = predict_directory(model, slice_dir, slice_geometries, expected_hash, 640, "slice_640")
    grouped = defaultdict(dict)
    for key, batch in raw_slices.items():
        stem, window = slice_metadata[key]
        grouped[stem][window.index] = batch
    p3 = {}
    t4 = {}
    slice_reports = {}
    for stem in baseline:
        report = merge_sliced_predictions(grouped[stem], windows_by_stem[stem], original_shapes[stem],
                                          checkpoint_id=expected_hash, iou_threshold=MERGE_IOU, max_detections=300)
        p3[stem] = report.result.predictions
        t4[stem] = merge_predictions((baseline[stem], p3[stem]), iou_threshold=MERGE_IOU,
                                     max_detections=300, source="T4:full_plus_slices").predictions
        slice_reports[stem] = {
            "windows": len(report.windows),
            "raw_predictions": report.raw_predictions,
            "kept_predictions": report.result.kept_predictions,
            "duplicate_predictions": report.result.duplicate_predictions,
            "clipped_predictions": report.result.clipped_predictions,
            "discarded_predictions": report.result.discarded_predictions,
        }
    reports = {}
    for name, batches in {"B0": baseline, "T1": t1, "T2": t2, "T3": t3, "P3": p3, "T4": t4}.items():
        reports[name] = evaluate_policy(gt, batches, name)
    reference = 0.7026892903603926
    reproduced = reports["B0"]["roadmap_strict"]["size_metrics"]["all"]["AP50_95"]
    if abs(reproduced - reference) > 0.005:
        raise RuntimeError(f"Baseline reproduction drift: {reproduced} vs {reference}")
    summary = {
        "checkpoint_hash": expected_hash,
        "split_hash": sha(plan / "split_manifest.json"),
        "code_hash": sha(__file__),
        "gpu": "Tesla T4",
        "policies_selected_on": "legacy_validation_design_screen",
        "submission_created": False,
        "scales": list(SCALES),
        "merge_iou": MERGE_IOU,
        "slice_shape": list(SLICE_SHAPE),
        "slice_overlap": list(SLICE_OVERLAP),
        "runtime_seconds": time.monotonic() - started,
        "slice_audit": slice_reports,
        "results": {
            name: {
                "map50_95": report["roadmap_strict"]["size_metrics"]["all"]["AP50_95"],
                "map75": report["roadmap_strict"]["size_metrics"]["all"]["AP75"],
                "AP_small": report["roadmap_strict"]["size_metrics"]["small"]["AP50_95"],
                "AP_medium": report["roadmap_strict"]["size_metrics"]["medium"]["AP50_95"],
                "AP_large": report["roadmap_strict"]["size_metrics"]["large"]["AP50_95"],
                "predictions": report["prediction_count"],
            }
            for name, report in reports.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
