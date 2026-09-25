from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import kagglehub
import torch

import hsi_runner as runner
from evaluate_size_ap import evaluate_size_ap, detection_confusion


OUT = Path("/kaggle/working/roadmap_day1")
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def event(experiment_id, status, **fields):
    row = dict(experiment_id=experiment_id, status=status,
               recorded_at=datetime.now(timezone.utc).isoformat(), **fields)
    with (OUT / "experiment_ledger.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def one(root, pattern):
    matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} in {root}, found {len(matches)}")
    return matches[0]


def diagnostics(base, model_id, artifact, checkpoint_hash, family, run_dir):
    started = time.monotonic()
    work = one(artifact, run_dir)
    weights = one(work, "best.pt")
    if sha(weights) != checkpoint_hash:
        raise RuntimeError(f"Historical checkpoint hash mismatch: {model_id}")
    split = json.loads((work / "split_manifest.json").read_text())
    audit = json.loads((work / "data_audit.json").read_text())
    config = json.loads((work / "run_config.json").read_text())
    names = runner.read_classes(base / "class.txt")
    cube_paths = runner.unique_stems(runner.collect_cubes(base / "data_train/data_train/VIS"))
    xml_paths = runner.annotation_map(base / "data_train/data_train/Annotations/VIS")
    lo = np.asarray(audit["band_low_0_5pct"], dtype=np.float32)
    hi = np.asarray(audit["band_high_99_5pct"], dtype=np.float32)
    policy = config.get("resolved_input_policy", "native")
    if policy not in ("native", "square_pad"):
        raise ValueError(policy)
    prepared = Path("/kaggle/temp/roadmap_day1") / model_id
    prepared.mkdir(parents=True, exist_ok=False)
    images, annotations, geometries = [], [], {}
    for stem in split["val"]:
        cube = runner.read_cube(cube_paths[stem])
        width, height, rows = runner.parse_voc(xml_paths[stem], {name: i for i, name in enumerate(names)})
        if cube.shape[:2] != (height, width):
            raise ValueError(f"Spatial mismatch: {stem}")
        images.append({"id": int(stem), "width": width, "height": height})
        for cid, cx, cy, bw, bh in rows:
            bbox = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
            annotations.append({"id": len(annotations) + 1, "image_id": int(stem), "category_id": int(cid), "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
        geometries[stem] = runner.write_tiff(cube, prepared / f"{stem}.tiff", lo, hi, policy)
    gt = {"images": images, "annotations": annotations, "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}
    save(f"{model_id}_gt_original.json", gt)
    from ultralytics import RTDETR, YOLO
    model = (RTDETR if family == "rtdetr" else YOLO)(str(weights))
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    predictions, seen = [], set()
    for result in model.predict(source=str(prepared), imgsz=config["args"]["imgsz"], conf=0.001,
                                iou=0.65, max_det=300, device=device, stream=True, verbose=False):
        stem = Path(result.path).stem
        seen.add(stem)
        boxes = runner.restore_boxes(result.boxes.xyxy.cpu().numpy(), geometries[stem])
        for box, score, cid in zip(boxes, result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(float, box)
            if x2 > x1 and y2 > y1:
                predictions.append({"image_id": int(stem), "category_id": int(cid), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    if seen != set(split["val"]):
        raise RuntimeError("Incomplete validation inference")
    save(f"{model_id}_predictions_original.json", predictions)
    report = evaluate_size_ap(gt, predictions, max_dets=300)
    report["confusion_at_fixed_diagnostic_threshold"] = detection_confusion(gt, predictions, confidence_threshold=0.25, iou_threshold=0.5)
    report["threshold_note"] = "0.25/0.5 is a fixed reporting operating point, not an annotation-candidate or submission threshold."
    report["provenance"] = {"checkpoint_hash": checkpoint_hash, "config_hash": sha(work / "run_config.json"),
                            "split_hash": sha(work / "split_manifest.json"), "code_hash": sha(__file__),
                            "evaluation_split": "legacy_validation_only", "device": device,
                            "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None,
                            "normalization": "saved training-only percentile parameters", "input_policy": policy,
                            "runtime_seconds": time.monotonic() - started,
                            "versions": {p: importlib.metadata.version(p) for p in ("torch", "ultralytics", "pycocotools", "numpy")}}
    save(f"{model_id}_size_ap.json", report)
    event(f"D1-{model_id}", "measured_legacy_validation", evidence=f"{model_id}_size_ap.json",
          observation=report["roadmap_strict"]["size_metrics"],
          evidence_quality="original-pixel ground truth and predictions from hash-matched historical checkpoint",
          hypothesis="size diagnostics measured; no training improvement tested",
          decision="use measured size support to design P-series; require fixed-fold training for promotion",
          smallest_next_test="matched P1 higher-resolution evaluation after baseline CV")


def main():
    data = Path(kagglehub.dataset_download("itsasup/hyperspectral-d01-base-data-col-fix"))
    base = one(data, "hyperspectral-2026")
    backup = Path(kagglehub.dataset_download("itsasup/working-backup-plan-c-hod"))
    plan_c = one(backup, "hsi_plan_c_ratio")
    inventory = [{"path": str(p.relative_to(data)), "bytes": p.stat().st_size} for p in data.rglob("*") if p.is_file()]
    save("data_inventory.json", inventory)
    with zipfile.ZipFile(OUT / "training_annotation_metadata.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (base / "data_train/data_train/Annotations/VIS").rglob("*.xml"):
            archive.write(path, str(path.relative_to(base)))
        archive.write(base / "class.txt", "class.txt")
    command = [sys.executable, "/kaggle/working/build_folds.py", "--xml-dir", str(base / "data_train/data_train/Annotations/VIS"),
               "--cube-dir", str(base / "data_train/data_train/VIS"), "--class-file", str(base / "class.txt"),
               "--legacy-split", str(plan_c / "split_manifest.json"), "--output-dir", str(OUT / "folds"), "--seed", "42"]
    result = subprocess.run(command, text=True, capture_output=True)
    (OUT / "fold_build.log").write_text(result.stdout + result.stderr)
    event("V1", "folds_built" if result.returncode == 0 else "failed_requires_diagnosis",
          evidence="fold_build.log", returncode=result.returncode,
          decision="review grouping metadata and fold diagnostics before training", smallest_next_test="group-evidence audit")
    jobs = [
        ("rtdetr_plan_c", "dataset", "itsasup/working-backup-plan-c-hod", "b4a54b0f5ad05a25fae96daf070e53286852f15c7091dd3fa13af2113afb9012", "rtdetr", "hsi_plan_c_ratio"),
        ("rtdetr_plan_b", "mounted", None, "87292ca78adee9163a1dca6d8aafaa48ebb7a550e2347e2d265d2db1ad493395", "rtdetr", "hsi_plan_b_v2"),
        ("yolo_plan_a", "mounted", None, "52800d6c16b33155e34d6e53ecdb31bb009ed0e1356d1777ca67b29d0509e291", "yolo", "hsi_plan_a_recovery"),
    ]
    for model_id, kind, source, expected_hash, family, run_dir in jobs:
        try:
            artifact = backup if kind == "dataset" else Path("/kaggle/input")
            diagnostics(base, model_id, artifact, expected_hash, family, run_dir)
        except Exception as error:
            (OUT / f"{model_id}_failure.txt").write_text(traceback.format_exc())
            event(f"D1-{model_id}", "failed_requires_diagnosis", error=str(error),
                  decision="repair exact failure and rerun; do not drop required experiment",
                  smallest_next_test="resolve recorded failure", evidence=f"{model_id}_failure.txt")
    with zipfile.ZipFile("/kaggle/working/roadmap_day1_evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.rglob("*"):
            if path.is_file():
                archive.write(path, str(path.relative_to(OUT)))
    print("Evidence package: /kaggle/working/roadmap_day1_evidence.zip", flush=True)


if __name__ == "__main__":
    main()
