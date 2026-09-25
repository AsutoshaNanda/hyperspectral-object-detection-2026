"""Ruler check: does eval_ultra.py reproduce Ultralytics' own mAP?  (run with /venv/main python)
Uses a finished RT-DETR run (default selftrain_all): predicts its prepared 'test' split with
model.predict (conf 0.001, max_det 300) and scores with eval_ultra; compare to evaluation.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import RTDETR

sys.path.insert(0, "/workspace/cascade")
from eval_ultra import score  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "selftrain_all"
work = Path(f"/workspace/{run}")
sel = json.loads((work / "selection.json").read_text())
ev = json.loads((work / "evaluation.json").read_text())
root = Path(__import__("yaml").safe_load((work / "dataset.yaml").read_text())["path"])
model = RTDETR(sel["selected_checkpoint"])
images, anns, preds = [], [], []
aid = 1
for i, tif in enumerate(sorted((root / "images" / "test").glob("*.tiff")), start=1):
    ok, frames = cv2.imreadmulti(str(tif), flags=cv2.IMREAD_UNCHANGED)
    h, w = frames[0].shape[:2]
    images.append({"id": i, "width": w, "height": h})
    lab = root / "labels" / "test" / f"{tif.stem}.txt"
    for line in lab.read_text().split("\n") if lab.exists() else []:
        if not line.strip():
            continue
        c, cx, cy, bw, bh = map(float, line.split())
        anns.append({"id": aid, "image_id": i, "category_id": int(c),
                     "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]})
        aid += 1
    r = model.predict(str(tif), imgsz=1024, conf=0.001, max_det=300, verbose=False)[0]
    for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        preds.append({"image_id": i, "category_id": int(c), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
out = Path("/workspace/cascade/ruler")
out.mkdir(exist_ok=True)
(out / "gt.json").write_text(json.dumps({"images": images, "annotations": anns}))
(out / "preds.json").write_text(json.dumps(preds))
mine = score(out / "gt.json", out / "preds.json")
print(json.dumps({"run": run, "ultralytics_val_map50_95": ev["test"]["map50_95"],
                  "eval_ultra_map50_95": mine["map50_95"],
                  "difference": mine["map50_95"] - ev["test"]["map50_95"]}, indent=2))
