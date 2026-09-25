"""Predict with a trained run's checkpoint on competition images (test, and ranking in Phase 2).

Inference only. Reuses hsi_runner's own preprocessing (same band limits saved in the run's
data_audit.json, square_pad) and its predict() (same conf/iou/max_det, same CSV checks).
Usage:
  python mac_predict.py <run_dir> <out_dir> <cubes_dir> [<cubes_dir> ...]
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import hsi_runner as R  # noqa: E402

run = Path(sys.argv[1])
out = Path(sys.argv[2])
cube_dirs = sys.argv[3:]
audit = json.loads((run / "data_audit.json").read_text())
cfg = json.loads((run / "run_config.json").read_text())["args"]
lo = np.asarray(audit["band_low_0_5pct"], dtype=np.float64)
hi = np.asarray(audit["band_high_99_5pct"], dtype=np.float64)
policy = audit["input_policy"]

tiff_dir = out / "tiff"
tiff_dir.mkdir(parents=True, exist_ok=True)
files = []
for d in cube_dirs:
    files += R.collect_cubes(d)
by_stem = R.unique_stems(files)  # raises on duplicate IDs across test/ranking
geometry = {}
for stem, path in sorted(by_stem.items()):
    geometry[stem] = R.write_tiff(R.read_cube(path), tiff_dir / f"{stem}.tiff", lo, hi, policy)
(out / "inference_geometry.json").write_text(json.dumps(geometry))

import os  # noqa: E402
# Optional overrides (Plan C's best public file used conf 0.001 / max 300 instead of its saved 0.05 / 100).
conf = float(os.environ.get("PRED_CONF", cfg["predict_conf"]))
max_det = int(os.environ.get("PRED_MAX_DET", cfg["predict_max_det"]))
ckpt = Path(os.environ.get("PRED_CKPT", run / "best.pt"))
tta = os.environ.get("PRED_TTA", "0") == "1"  # flip/multi-scale TTA (allowed on ranking set)
args = SimpleNamespace(family=cfg["family"], imgsz=cfg["imgsz"], predict_conf=conf,
                       iou=cfg["iou"], predict_max_det=max_det, device="mps",
                       workdir=str(out), sample_submission=None)
print("checkpoint", ckpt, "imgsz", args.imgsz, "conf", conf, "max_det", max_det, "tta", tta)
sub = R.predict(args, ckpt, tiff_dir, geometry, tta)
print("images:", len(geometry), "->", sub)
print((out / "prediction_audit.json").read_text())
