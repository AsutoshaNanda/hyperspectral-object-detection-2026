"""Horizontal-flip TTA prediction (the "T1" policy measured on the Kaggle inference screen: +0.0025).

Same model, inference only: predict the normal image and a left-right flipped copy, flip the second
set of boxes back, merge with class-wise hard NMS (IoU 0.65), keep the top 300 per image.
Ultralytics' augment=True is a no-op for RT-DETR, hence this explicit implementation.
The ranking-set rule allows this ("flipping TTA", organizer Phase-2 notice).
Usage (env vars PRED_CONF / PRED_MAX_DET / PRED_CKPT as in mac_predict.py):
  python mac_predict_tta.py <run_dir> <out_dir> <cubes_dir> [<cubes_dir> ...]
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tta"))
import hsi_runner as R  # noqa: E402
from inference_common import PredictionBatch  # noqa: E402
from inference_tta import horizontal_flip_tta  # noqa: E402

run, out, cube_dirs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
audit = json.loads((run / "data_audit.json").read_text())
cfg = json.loads((run / "run_config.json").read_text())["args"]
lo = np.asarray(audit["band_low_0_5pct"], dtype=np.float64)
hi = np.asarray(audit["band_high_99_5pct"], dtype=np.float64)
policy = audit["input_policy"]
conf = float(os.environ.get("PRED_CONF", cfg["predict_conf"]))
max_det = int(os.environ.get("PRED_MAX_DET", cfg["predict_max_det"]))
ckpt = Path(os.environ.get("PRED_CKPT", run / "best.pt"))
imgsz = cfg["imgsz"]
MERGE_IOU = 0.65

files = []
for d in cube_dirs:
    files += R.collect_cubes(d)
by_stem = R.unique_stems(files)
normal_dir, flip_dir = out / "tiff_normal", out / "tiff_flip"
normal_dir.mkdir(parents=True, exist_ok=True)
flip_dir.mkdir(parents=True, exist_ok=True)
geo_n, geo_f, widths = {}, {}, {}
for stem, path in sorted(by_stem.items()):
    cube = R.read_cube(path)
    widths[stem] = cube.shape[1]
    geo_n[stem] = R.write_tiff(cube, normal_dir / f"{stem}.tiff", lo, hi, policy)
    geo_f[stem] = R.write_tiff(np.flip(cube, axis=1).copy(), flip_dir / f"{stem}.tiff", lo, hi, policy)
(out / "inference_geometry.json").write_text(json.dumps(geo_n))

model = R.model_class(SimpleNamespace(family=cfg["family"]))(str(ckpt))
ckid = str(ckpt)


def run_dir(directory, geometry, tag):
    res = {}
    for r in model.predict(source=str(directory), imgsz=imgsz, conf=conf, iou=cfg["iou"], max_det=max_det,
                           device="mps", stream=True, verbose=False):
        stem = Path(r.path).stem
        if r.boxes is None or len(r.boxes) == 0:
            res[stem] = PredictionBatch(np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), ckid, tag)
            continue
        b = R.restore_boxes(r.boxes.xyxy.cpu().numpy(), geometry[stem])
        s = r.boxes.conf.cpu().numpy()
        c = r.boxes.cls.cpu().numpy().astype(int)
        ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        res[stem] = PredictionBatch(b[ok], s[ok], c[ok], ckid, tag)
    if set(res) != set(geometry):
        raise RuntimeError(f"incomplete inference for {tag}")
    return res


print("checkpoint", ckpt, "imgsz", imgsz, "conf", conf, "max_det", max_det, "TTA: normal + horizontal flip")
normal = run_dir(normal_dir, geo_n, "normal")
flipped = run_dir(flip_dir, geo_f, "flip")
rows, empty = [], []
for stem in sorted(normal):
    merged = horizontal_flip_tta(normal[stem], flipped[stem], widths[stem], iou_threshold=MERGE_IOU,
                                 max_detections=max_det).predictions
    if len(merged) == 0:
        empty.append(stem)
    for (x1, y1, x2, y2), s, c in zip(merged.boxes, merged.scores, merged.labels):
        rows.append({"image_id": R.image_id_value(stem), "class_id": int(c), "confidence": float(s),
                     "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
sub = pd.DataFrame(rows)
sub.insert(0, "id", np.arange(len(sub), dtype=int))
assert sub["class_id"].between(0, 17).all() and sub["confidence"].between(0, 1).all()
assert ((sub.x2 > sub.x1) & (sub.y2 > sub.y1)).all() and np.isfinite(sub[["x1", "y1", "x2", "y2"]].to_numpy()).all()
assert sub.image_id.nunique() + len(empty) == len(by_stem)
sub.to_csv(out / "submission.csv", index=False)
audit_out = {"processed_images": len(by_stem), "no_detection_ids": empty, "predictions_total": len(sub),
             "tta": "T1 normal+hflip, hard-NMS iou 0.65", "conf": conf, "max_det": max_det, "imgsz": imgsz,
             "per_image_max": int(sub.image_id.value_counts().max())}
(out / "prediction_audit.json").write_text(json.dumps(audit_out, indent=2))
print(json.dumps(audit_out, indent=2))
