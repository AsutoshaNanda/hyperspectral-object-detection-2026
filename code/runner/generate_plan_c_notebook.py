from pathlib import Path
import json


root = Path(__file__).parent
runner = (root / "submission01_yolo16M.py").read_text()
target = root / "Plan_C_Ratio"
target.mkdir(parents=True, exist_ok=True)

markdown = """# Plan C: controlled RT-DETR aspect-ratio test

This changes one material factor from Plan B v2: native 512 by 256 cubes are center-padded to 512 by 512 before RT-DETR can stretch them. Labels and final predictions are mapped through the same geometry. Model, seed, split logic, epochs, image size, batch size, augmentation, and random 16-channel stem remain fixed. The notebook records validation and holdout deltas against Plan B v2 and does not submit to Kaggle automatically.
"""

setup = """import sys, subprocess, importlib.util
from pathlib import Path
import torch
assert torch.cuda.is_available(), "Enable GPU. This run needs Kaggle's CUDA PyTorch."
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "ultralytics==8.4.147", "filelock>=3.16", "ultralytics-thop"], check=True)
if importlib.util.find_spec("polars") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "polars"], check=True)
"""

write_runner = f"from pathlib import Path\nPath('/kaggle/working/hsi_runner.py').write_text({runner!r})\n"

run = """import subprocess, sys
from pathlib import Path
roots = list(Path("/kaggle/input").rglob("hyperspectral-2026"))
assert len(roots) == 1, f"Expected one competition data folder, got {roots}"
base = roots[0]
sample = base.parent / "sample_submission_corrected.csv"
assert sample.exists(), "Corrected sample submission is missing"
cmd = [sys.executable, "/kaggle/working/hsi_runner.py",
       "--train-cubes", str(base / "data_train/data_train/VIS"),
       "--test-cubes", str(base / "data_test/data_test/VIS"),
       "--train-xml", str(base / "data_train/data_train/Annotations/VIS"),
       "--class-file", str(base / "class.txt"),
       "--sample-submission", str(sample),
       "--family", "rtdetr", "--model", "rtdetr-l.pt",
       "--input-policy", "square_pad", "--stem-init", "random",
       "--run-name", "rtdetr_ratio_control",
       "--workdir", "/kaggle/working/hsi_plan_c_ratio",
       "--cache-dir", "/kaggle/temp/hsi_plan_c_ratio_cache",
       "--epochs", "50", "--imgsz", "640", "--batch", "2", "--workers", "2",
       "--predict-conf", "0.05", "--predict-max-det", "100",
       "--min-holdout-map", "0.50"]
subprocess.run(cmd, check=True)
"""

review = """import json
from pathlib import Path
work = Path("/kaggle/working/hsi_plan_c_ratio")
evaluation = json.loads((work / "evaluation.json").read_text())
gate = json.loads((work / "submission_gate.json").read_text())
baseline = {
    "val": {"map50_95": 0.6658722520835962, "map50": 0.9625481592929565, "map75": 0.8160333387847114},
    "test": {"map50_95": 0.637055468237113, "map50": 0.9421560423763289, "map75": 0.7533592409119105},
}
deltas = {
    split: {metric: evaluation[split][metric] - baseline[split][metric] for metric in ("map50_95", "map50", "map75")}
    for split in ("val", "test")
}
supports = deltas["val"]["map50_95"] >= 0.005 and deltas["test"]["map50_95"] >= 0
decision = {
    "observation": "RT-DETR square-padding result compared with the saved Plan B v2 baseline",
    "evidence_quality": "controlled same-seed and same-split Kaggle run; one material preprocessing change",
    "deltas": deltas,
    "hypothesis": "supported" if supports else "not_supported",
    "decision": "promote square padding" if supports else "retain native preprocessing and test a different mechanism",
    "smallest_next_test": "activation-scale-preserving RGB-to-16-band stem initialization under the winning input policy",
    "submission_gate": gate,
}
(work / "evidence_decision.json").write_text(json.dumps(decision, indent=2))
print(json.dumps(decision, indent=2))
print("Submission file:", work / "submission.csv" if (work / "submission.csv").exists() else "not created")
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

(target / "hyperspectral_plan_c_rtdetr_ratio.ipynb").write_text(json.dumps(notebook, indent=1))
