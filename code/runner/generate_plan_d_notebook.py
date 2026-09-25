from pathlib import Path
import json


root = Path(__file__).parent
runner = (root / "submission01_yolo16M.py").read_text()
target = root / "Plan_D_LowConf"
target.mkdir(parents=True, exist_ok=True)

markdown = """# Plan D: RT-DETR validated low-confidence re-inference

This reuses the exact Plan B v2 checkpoint. It changes no model weights and keeps the original native RT-DETR preprocessing. It changes only the exported prediction floor from confidence 0.05 and 100 maximum detections to the locally validated confidence 0.001 and 300 maximum detections. It creates a candidate CSV but never submits automatically.
"""

setup = """import sys, subprocess, importlib.util
from pathlib import Path
import torch
device = "cpu"
print("PyTorch:", torch.__version__, "device:", device)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "ultralytics==8.4.147", "filelock>=3.16", "ultralytics-thop"], check=True)
if importlib.util.find_spec("polars") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "polars"], check=True)
"""

write_runner = f"from pathlib import Path\nPath('/kaggle/working/hsi_runner.py').write_text({runner!r})\n"

run = """import subprocess, sys
from pathlib import Path
roots = list(Path("/kaggle/input").rglob("hyperspectral-2026"))
assert len(roots) == 1, f"Expected one competition data folder, got {roots}"
checkpoints = [path for path in Path("/kaggle/input").rglob("best.pt") if "rtdetr_full/weights" in path.as_posix()]
assert len(checkpoints) == 1, f"Expected the Plan B v2 RT-DETR checkpoint, got {checkpoints}"
base = roots[0]
sample = base.parent / "sample_submission_corrected.csv"
assert sample.exists(), "Corrected sample submission is missing"
cmd = [sys.executable, "/kaggle/working/hsi_runner.py",
       "--train-cubes", str(base / "data_train/data_train/VIS"),
       "--test-cubes", str(base / "data_test/data_test/VIS"),
       "--train-xml", str(base / "data_train/data_train/Annotations/VIS"),
       "--class-file", str(base / "class.txt"),
       "--sample-submission", str(sample),
       "--family", "rtdetr", "--checkpoint", str(checkpoints[0]),
       "--input-policy", "native", "--stem-init", "random",
       "--device", device,
       "--run-name", "rtdetr_low_conf_inference",
       "--workdir", "/kaggle/working/hsi_plan_d_low_conf",
       "--cache-dir", "/kaggle/temp/hsi_plan_d_low_conf_cache",
       "--imgsz", "640", "--batch", "2", "--workers", "2",
       "--conf", "0.001", "--max-det", "300",
       "--predict-conf", "0.001", "--predict-max-det", "300",
       "--min-holdout-map", "0.50"]
subprocess.run(cmd, check=True)
"""

review = """import hashlib, json, pandas as pd
from pathlib import Path
work = Path("/kaggle/working/hsi_plan_d_low_conf")
gate = json.loads((work / "submission_gate.json").read_text())
evaluation = json.loads((work / "evaluation.json").read_text())
audit = json.loads((work / "prediction_audit.json").read_text())
submission = work / "submission.csv"
assert gate["passed"], gate
assert submission.exists(), "Gate passed but submission.csv is missing"
frame = pd.read_csv(submission)
decision = {
    "observation": "The saved RT-DETR checkpoint was re-inferred at the same low floor used by validation",
    "evidence_quality": "same checkpoint and preprocessing; export threshold and maximum detections are the only material changes",
    "hypothesis": "awaiting public score",
    "decision": "submit this candidate once after CSV review, then record whether recall improved public mAP",
    "smallest_next_test": "RT-DETR square-padding control",
    "rows_old": 11604,
    "rows_new": len(frame),
    "images_without_detections_old": 3,
    "images_without_detections_new": len(audit["no_detection_ids"]),
    "holdout_map50_95": evaluation["test"]["map50_95"],
    "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
}
(work / "evidence_decision.json").write_text(json.dumps(decision, indent=2))
print(json.dumps(decision, indent=2))
display(frame.head())
"""


def cell(cell_type, source):
    result = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


notebook = {
    "cells": [
        cell("markdown", markdown),
        cell("code", setup),
        cell("code", write_runner),
        cell("code", run),
        cell("code", review),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(target / "hyperspectral_plan_d_rtdetr_low_conf.ipynb").write_text(json.dumps(notebook, indent=1))
