import argparse
import contextlib
import copy
import importlib.metadata
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


AREA_LABELS = ("all", "small", "medium", "large")
OFFICIAL_SOURCE = "https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py"


def _prepare_inputs(coco_gt, predictions):
    gt = copy.deepcopy(coco_gt)
    predictions = copy.deepcopy(list(predictions))
    for key in ("images", "categories", "annotations"):
        if key not in gt:
            raise ValueError(f"COCO ground truth is missing {key}")
    image_ids = [image["id"] for image in gt["images"]]
    category_ids = [category["id"] for category in gt["categories"]]
    if not image_ids or not category_ids:
        raise ValueError("At least one image and category are required")
    if len(set(image_ids)) != len(image_ids) or len(set(category_ids)) != len(category_ids):
        raise ValueError("Image and category IDs must be unique")
    if any(not isinstance(i, int) or isinstance(i, bool) for i in image_ids + category_ids):
        raise ValueError("Image and category IDs must be integers")
    for image in gt["images"]:
        if any(not math.isfinite(float(image.get(k, 0))) or image.get(k, 0) <= 0 for k in ("width", "height")):
            raise ValueError("Original image width and height must be finite and positive")
    supplied_areas_replaced = 0
    for is_prediction, rows in ((False, gt["annotations"]), (True, predictions)):
        for index, row in enumerate(rows, 1):
            if row["image_id"] not in image_ids or row["category_id"] not in category_ids:
                raise ValueError("Annotation or prediction references an unknown image/category")
            bbox = np.asarray(row.get("bbox", []), dtype=np.float64)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or (bbox[2:] <= 0).any():
                raise ValueError("Bboxes must be finite original-pixel XYWH with positive width/height")
            area = float(bbox[2] * bbox[3])
            if not math.isfinite(area):
                raise ValueError("BBox area must be finite")
            row["bbox"] = bbox.tolist()
            if is_prediction:
                score = float(row["score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Prediction scores must be finite and between zero and one")
                row["score"] = score
            else:
                supplied_areas_replaced += int("area" in row and row["area"] != area)
                row["area"] = area
                row["id"] = index
                row["iscrowd"] = int(bool(row.get("iscrowd", 0)))
                if row.get("ignore", 0) and not row["iscrowd"]:
                    raise ValueError("Non-crowd ignore annotations need an explicit scoring policy")
    gt.setdefault("info", {})
    return gt, predictions, supplied_areas_replaced


def _mean_available(values):
    values = np.asarray(values)
    available = values[values > -1]
    return float(available.mean()) if available.size else None


def _metric_row(evaluator, area_index, class_index=None):
    precision = evaluator.eval["precision"][:, :, :, area_index, -1]
    recall = evaluator.eval["recall"][:, :, area_index, -1]
    if class_index is not None:
        precision = precision[:, :, class_index]
        recall = recall[:, class_index]
    thresholds = evaluator.params.iouThrs
    return {
        "AP50_95": _mean_available(precision),
        "AP50": _mean_available(precision[np.isclose(thresholds, 0.5)]),
        "AP75": _mean_available(precision[np.isclose(thresholds, 0.75)]),
        "AR50_95": _mean_available(recall),
        "recall50": _mean_available(recall[np.isclose(thresholds, 0.5)]),
        "recall75": _mean_available(recall[np.isclose(thresholds, 0.75)]),
    }


def _evaluate(coco_gt, coco_dt, *, strict, max_dets):
    from pycocotools.cocoeval import COCOeval

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = sorted(set((1, min(10, max_dets), max_dets)))
    if strict:
        evaluator.params.areaRng = [
            [0, float("inf")],
            [0, float(np.nextafter(1024.0, -np.inf))],
            [1024, float(np.nextafter(9216.0, -np.inf))],
            [9216, float("inf")],
        ]
    evaluator.evaluate()
    evaluator.accumulate()
    categories = {category["id"]: category["name"] for category in coco_gt.dataset["categories"]}
    return {
        "size_metrics": {label: _metric_row(evaluator, i) for i, label in enumerate(AREA_LABELS)},
        "class_by_size": [
            {"category_id": int(category_id), "category_name": categories[category_id],
             "sizes": {label: _metric_row(evaluator, i, k) for i, label in enumerate(AREA_LABELS)}}
            for k, category_id in enumerate(evaluator.params.catIds)
        ],
    }


def evaluate_size_ap(coco_gt, predictions, *, max_dets=100):
    from pycocotools.coco import COCO

    if not isinstance(max_dets, int) or isinstance(max_dets, bool) or max_dets < 1:
        raise ValueError("max_dets must be a positive integer")
    gt, predictions, replaced = _prepare_inputs(coco_gt, predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = gt
        ground_truth.createIndex()
        if predictions:
            detections = ground_truth.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {"images": copy.deepcopy(gt["images"]),
                                  "categories": copy.deepcopy(gt["categories"]), "annotations": []}
            detections.createIndex()
        standard = _evaluate(ground_truth, detections, strict=False, max_dets=max_dets)
        strict = _evaluate(ground_truth, detections, strict=True, max_dets=max_dets)
    raw_areas = [{"annotation_id": row["id"], "image_id": row["image_id"],
                  "category_id": row["category_id"], "area": row["area"], "iscrowd": row["iscrowd"]}
                 for row in gt["annotations"]]
    bins = [0, 64, 256, 1024, 4096, 9216, 16384, 65536, float("inf")]
    histogram = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        rows = [row for row in raw_areas if lower <= row["area"] < upper]
        histogram.append({"lower_inclusive": lower, "upper_exclusive": upper if math.isfinite(upper) else None,
                          "count": len(rows), "non_crowd_count": sum(not row["iscrowd"] for row in rows)})
    return {
        "coordinate_space": "original_image_pixels_xywh",
        "ground_truth_area_policy": "bbox_width_times_height_without_rescaling",
        "supplied_gt_areas_replaced": replaced,
        "pycocotools_version": importlib.metadata.version("pycocotools"),
        "evaluator_source": OFFICIAL_SOURCE,
        "iou_thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)],
        "max_detections_per_image_per_class": max_dets,
        "confidence_filter": None,
        "boundary_policy": {
            "roadmap_strict": "small [0,1024), medium [1024,9216), large [9216,infinity)",
            "coco_standard": "small [0,1024], medium [1024,9216], large [9216,1e10]; inclusive overlapping boundaries",
        },
        "roadmap_strict": strict,
        "coco_standard": standard,
        "raw_box_areas": raw_areas,
        "box_area_histogram": histogram,
    }


def _iou(first, second):
    x = max(first[0], second[0])
    y = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - x) * max(0.0, bottom - y)
    return intersection / (first[2] * first[3] + second[2] * second[3] - intersection)


def detection_confusion(coco_gt, predictions, *, confidence_threshold, iou_threshold):
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between zero and one")
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than zero and at most one")
    gt, predictions, _ = _prepare_inputs(coco_gt, predictions)
    if any(row["iscrowd"] for row in gt["annotations"]):
        raise ValueError("This independent confusion diagnostic does not support crowd annotations")
    categories = sorted(gt["categories"], key=lambda category: category["id"])
    indexes = {category["id"]: i for i, category in enumerate(categories)}
    background = len(categories)
    matrix = np.zeros((background + 1, background + 1), dtype=np.int64)
    truths_by_image = defaultdict(list)
    predictions_by_image = defaultdict(list)
    for row in gt["annotations"]:
        truths_by_image[row["image_id"]].append(row)
    for row in predictions:
        if row["score"] >= confidence_threshold:
            predictions_by_image[row["image_id"]].append(row)
    for image in gt["images"]:
        truths = truths_by_image[image["id"]]
        unmatched = set(range(len(truths)))
        for prediction in sorted(predictions_by_image[image["id"]], key=lambda row: -row["score"]):
            matches = [(i, _iou(prediction["bbox"], truths[i]["bbox"])) for i in sorted(unmatched)]
            best = max(matches, key=lambda match: match[1]) if matches else None
            predicted_class = indexes[prediction["category_id"]]
            if best is not None and best[1] >= iou_threshold:
                true_index = best[0]
                unmatched.remove(true_index)
                matrix[indexes[truths[true_index]["category_id"]], predicted_class] += 1
            else:
                matrix[background, predicted_class] += 1
        for index in unmatched:
            matrix[indexes[truths[index]["category_id"]], background] += 1
    per_class = []
    for index, category in enumerate(categories):
        true_positive = int(matrix[index, index])
        predicted_count = int(matrix[:, index].sum())
        true_count = int(matrix[index, :].sum())
        per_class.append({"category_id": category["id"], "category_name": category["name"],
                          "true_positives": true_positive, "false_positives": predicted_count - true_positive,
                          "false_negatives": true_count - true_positive,
                          "precision": true_positive / predicted_count if predicted_count else None,
                          "recall": true_positive / true_count if true_count else None})
    return {"matrix": matrix.tolist(), "orientation": "rows=true, columns=predicted",
            "category_ids": [category["id"] for category in categories] + [None],
            "labels": [category["name"] for category in categories] + ["background"],
            "confidence_threshold": confidence_threshold, "iou_threshold": iou_threshold,
            "matching_policy": "descending score, class-agnostic highest-IoU unmatched GT; stable input-order ties",
            "per_class": per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--confusion-confidence", type=float)
    parser.add_argument("--confusion-iou", type=float)
    args = parser.parse_args()
    if (args.confusion_confidence is None) != (args.confusion_iou is None):
        parser.error("Provide both confusion thresholds or neither")
    gt = json.loads(args.ground_truth.read_text())
    predictions = json.loads(args.predictions.read_text())
    report = evaluate_size_ap(gt, predictions, max_dets=args.max_dets)
    if args.confusion_confidence is not None:
        report["confusion"] = detection_confusion(gt, predictions, confidence_threshold=args.confusion_confidence,
                                                  iou_threshold=args.confusion_iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
