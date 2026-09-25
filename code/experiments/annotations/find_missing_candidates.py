import csv
import json
import re
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    precision: float
    recall: float
    target_precision: float
    curve_id: str
    source_split: str = "validation"


@dataclass(frozen=True)
class DetectionCandidate:
    candidate_id: str
    image_id: str
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    max_official_iou: float
    confidence_threshold: float
    threshold_curve_id: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CandidatePackage:
    manifest_path: Path
    artifact_directory: Path
    candidate_count: int


def select_validation_threshold(curve, target_precision: float, curve_id: str):
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    if not curve_id.strip():
        raise ValueError("curve_id must identify measured validation evidence")
    eligible = []
    for row in curve:
        if row.get("source_split", "validation") != "validation":
            raise ValueError("Confidence threshold evidence must come from validation")
        threshold = float(row["threshold"])
        precision = float(row["precision"])
        recall = float(row["recall"])
        if not all(np.isfinite([threshold, precision, recall])):
            raise ValueError("Validation curve values must be finite")
        if not 0 <= threshold <= 1 or not 0 <= precision <= 1 or not 0 <= recall <= 1:
            raise ValueError("Validation curve values must be in [0, 1]")
        if precision >= target_precision:
            eligible.append((recall, precision, threshold))
    if not eligible:
        raise ValueError("No measured validation threshold reaches target_precision")
    recall, precision, threshold = max(eligible)
    return ThresholdSelection(threshold, precision, recall, target_precision, curve_id)


def box_iou_xyxy(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("Boxes must contain four coordinates")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Boxes must be finite")
    if first[2] <= first[0] or first[3] <= first[1] or second[2] <= second[0] or second[3] <= second[1]:
        raise ValueError("Boxes must have positive width and height")
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return float(intersection / (first_area + second_area - intersection))


def record_bbox_xyxy(record: Mapping, bbox_format: str):
    values = np.asarray(record["bbox"], dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("bbox must contain four finite coordinates")
    if bbox_format == "xywh":
        x, y, width, height = values
        values = np.array([x, y, x + width, y + height], dtype=np.float64)
    elif bbox_format != "xyxy":
        raise ValueError("bbox_format must be xyxy or xywh")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("bbox must have positive width and height")
    return tuple(float(value) for value in values)


def discover_candidates(
    predictions: Iterable[Mapping],
    official_annotations: Iterable[Mapping],
    training_image_ids: Iterable,
    threshold_selection: ThresholdSelection,
    low_iou_threshold: float,
    *,
    prediction_bbox_format: str = "xyxy",
    annotation_bbox_format: str = "xywh",
):
    if not isinstance(threshold_selection, ThresholdSelection):
        raise TypeError("threshold_selection must come from a measured validation precision curve")
    if threshold_selection.source_split != "validation":
        raise ValueError("Candidate confidence threshold must be selected on validation")
    if not 0 <= low_iou_threshold <= 1:
        raise ValueError("low_iou_threshold must be in [0, 1]")
    training_ids = {str(value) for value in training_image_ids}
    if not training_ids:
        raise ValueError("training_image_ids must not be empty")
    truths = {}
    for annotation in official_annotations:
        image_id = str(annotation["image_id"])
        if image_id in training_ids:
            truths.setdefault(image_id, []).append(record_bbox_xyxy(annotation, annotation_bbox_format))
    candidates = []
    per_image_index = {}
    for prediction in predictions:
        image_id = str(prediction["image_id"])
        if image_id not in training_ids:
            raise ValueError(f"Candidate discovery received non-training image {image_id}")
        confidence = float(prediction.get("score", prediction.get("confidence", np.nan)))
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Prediction confidence must be in [0, 1]")
        if confidence < threshold_selection.threshold:
            continue
        bbox = record_bbox_xyxy(prediction, prediction_bbox_format)
        overlaps = [box_iou_xyxy(bbox, truth) for truth in truths.get(image_id, ())]
        max_iou = max(overlaps, default=0.0)
        if max_iou >= low_iou_threshold:
            continue
        box_index = per_image_index.get(image_id, 0)
        per_image_index[image_id] = box_index + 1
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", image_id)
        candidates.append(
            DetectionCandidate(
                f"{safe_id}_{box_index:05d}",
                image_id,
                int(prediction.get("category_id", prediction.get("class_id"))),
                bbox,
                confidence,
                max_iou,
                threshold_selection.threshold,
                threshold_selection.curve_id,
            )
        )
    return candidates


def _png_bytes(image):
    height, width, channels = image.shape
    if channels != 3 or image.dtype != np.uint8:
        raise ValueError("PNG input must be uint8 RGB")
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return signature + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(
        b"IDAT", zlib.compress(raw)
    ) + chunk(b"IEND", b"")


def _pseudo_rgb(cube, bands):
    selected = np.asarray(cube[..., bands], dtype=np.float64)
    result = np.zeros(selected.shape, dtype=np.uint8)
    for channel in range(3):
        values = selected[..., channel]
        low, high = np.percentile(values, [1, 99])
        if high > low:
            result[..., channel] = np.rint(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return result


def export_candidate_package(
    candidates: Sequence[DetectionCandidate],
    cubes_by_image: Mapping,
    output_directory,
    *,
    pseudo_rgb_bands=(11, 7, 3),
):
    output_directory = Path(output_directory)
    artifact_directory = output_directory / "annotation_candidates"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for candidate in candidates:
        cube = cubes_by_image.get(candidate.image_id, cubes_by_image.get(str(candidate.image_id)))
        if cube is None:
            raise KeyError(f"Missing cube for training image {candidate.image_id}")
        cube = np.asarray(cube)
        if cube.ndim != 3 or cube.shape[2] <= max(pseudo_rgb_bands):
            raise ValueError("Each cube must be HWC and contain the requested pseudo-RGB bands")
        x1, y1, x2, y2 = candidate.bbox_xyxy
        left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        right, bottom = min(cube.shape[1], int(np.ceil(x2))), min(cube.shape[0], int(np.ceil(y2)))
        if right <= left or bottom <= top:
            raise ValueError(f"Candidate {candidate.candidate_id} is outside its image")
        crop = cube[top:bottom, left:right]
        image_name = f"{candidate.candidate_id}_pseudo_rgb.png"
        spectral_name = f"{candidate.candidate_id}_spectral.csv"
        (artifact_directory / image_name).write_bytes(_png_bytes(_pseudo_rgb(crop, pseudo_rgb_bands)))
        with (artifact_directory / spectral_name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["y", "x", *[f"band_{index}" for index in range(crop.shape[2])]])
            for row in range(crop.shape[0]):
                for column in range(crop.shape[1]):
                    writer.writerow([top + row, left + column, *crop[row, column].tolist()])
        manifest_rows.append(
            {
                **candidate.to_dict(),
                "bbox_xyxy": list(candidate.bbox_xyxy),
                "pseudo_rgb": f"annotation_candidates/{image_name}",
                "spectral_csv": f"annotation_candidates/{spectral_name}",
            }
        )
    manifest_path = output_directory / "annotation_candidates.json"
    manifest_path.write_text(json.dumps({"candidates": manifest_rows}, indent=2, sort_keys=True) + "\n")
    return CandidatePackage(manifest_path, artifact_directory, len(candidates))
