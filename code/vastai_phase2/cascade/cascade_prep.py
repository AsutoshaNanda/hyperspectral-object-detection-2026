"""Prepare 16-band data for Cascade R-CNN (MMDetection), run with /venv/main python.

Same split as the Combo holdout comparison: Plan-C manifest (train 2397 / val 300 / holdout 300;
the holdout 300 == fold-0 test used for Combo's 0.672). Normalisation = the runner's per-band
0.5-99.5 percentile clip to uint8, fitted on TRAIN stems only. Native resolution, no padding
(MMDetection keeps aspect ratio itself). Boxes stay in original pixel coordinates.
Outputs /workspace/cascade/data/{train,val,test}/<stem>.npy (HxWx16 uint8) + COCO jsons.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace")
import hsi_runner as R  # noqa: E402

DATA = Path("/workspace/data")
OUT = Path("/workspace/cascade/data")
MANIFEST = json.loads(Path("/workspace/folds/planC_split_manifest.json").read_text())

cubes_root = [p for p in DATA.rglob("VIS") if "data_train" in str(p) and "Annotations" not in str(p)][0]
xml_root = [p for p in DATA.rglob("VIS") if "Annotations" in str(p)][0]
names = R.read_classes(str(next(DATA.rglob("class.txt"))))
c2i = {n: i for i, n in enumerate(names)}
cube_by_stem = R.unique_stems(R.collect_cubes(str(cubes_root)))
xml_by_stem = R.annotation_map(str(xml_root))

train_stems = [str(s) for s in MANIFEST["train"]]
lo, hi = R.band_limits([cube_by_stem[s] for s in train_stems], seed=42)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "band_limits.json").write_text(json.dumps({"lo": lo.tolist(), "hi": hi.tolist()}))

sums = np.zeros(16)
sqs = np.zeros(16)
count = 0
for split in ("train", "val", "test"):
    images, annotations = [], []
    ann_id = 1
    (OUT / split).mkdir(parents=True, exist_ok=True)
    for img_id, stem in enumerate(sorted(map(str, MANIFEST[split])), start=1):
        x, _ = R.prepare_cube(R.read_cube(cube_by_stem[stem]), lo, hi, "native")
        h, w = x.shape[:2]
        w_ann, h_ann, rows = R.parse_voc(xml_by_stem[stem], c2i)
        assert (w, h) == (w_ann, h_ann), (stem, (w, h), (w_ann, h_ann))
        np.save(OUT / split / f"{stem}.npy", x)
        if split == "train":
            flat = x.reshape(-1, 16).astype(np.float64)
            sums += flat.sum(0)
            sqs += (flat ** 2).sum(0)
            count += len(flat)
        images.append({"id": img_id, "file_name": f"{stem}.npy", "width": w, "height": h, "stem": stem})
        for cls, cx, cy, bw, bh in rows:
            bx, by, bwp, bhp = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": int(cls) + 1,
                                "bbox": [bx, by, bwp, bhp], "area": bwp * bhp, "iscrowd": 0})
            ann_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    (OUT / f"{split}.json").write_text(json.dumps(coco))
    print(f"{split}: {len(images)} images, {len(annotations)} boxes")

mean = sums / count
std = np.sqrt(sqs / count - mean ** 2)
(OUT / "pixel_stats.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(), "classes": names}))
print("mean", np.round(mean, 2).tolist())
print("std", np.round(std, 2).tolist())
