from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Experiments/kaggle/inference_screen"


def cell(cell_type, source):
    result = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def main():
    files = {
        "hsi_runner.py": ROOT / "Submission/submission01_yolo16M.py",
        "evaluate_size_ap.py": ROOT / "Experiments/evaluation/evaluate_size_ap.py",
        "inference_common.py": ROOT / "Experiments/inference/common.py",
        "inference_multiscale.py": ROOT / "Experiments/inference/multiscale.py",
        "inference_sliced.py": ROOT / "Experiments/inference/sliced_inference.py",
        "inference_tta.py": ROOT / "Experiments/inference/tta.py",
        "run_inference_screen.py": ROOT / "Experiments/run_inference_screen.py",
    }
    sources = "from pathlib import Path\n"
    for name, path in files.items():
        text = path.read_text()
        if name.startswith("inference_"):
            text = text.replace("from .common import", "from inference_common import")
            text = text.replace("from .multiscale import", "from inference_multiscale import")
        sources += f"Path('/kaggle/working/{name}').write_text({text!r})\n"
    setup = """import importlib.util
import subprocess
import sys

packages = []
for name, spec in [('ultralytics', 'ultralytics==8.4.147'), ('pycocotools', 'pycocotools')]:
    if importlib.util.find_spec(name) is None:
        packages.append(spec)
if packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *packages], check=True)
"""
    execute = """import subprocess
import sys

subprocess.run([sys.executable, '/kaggle/working/run_inference_screen.py'], check=True)
"""
    notebook = {
        "cells": [
            cell("markdown", "# Same-checkpoint inference screen\n\nValidation-only T1-T4 and P3 comparison. No competition submission is created."),
            cell("code", setup),
            cell("code", sources),
            cell("code", execute),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"},
                     "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "sourceType": "notebook", "isGpuEnabled": True}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": "itsasup/hsi-same-checkpoint-inference-screen",
        "title": "hsi-same-checkpoint-inference-screen",
        "code_file": "hsi_same_checkpoint_inference_screen.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": ["itsasup/hyperspectral-d01-base-data-col-fix", "itsasup/working-backup-plan-c-hod"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / metadata["code_file"]).write_text(json.dumps(notebook, indent=1))
    (TARGET / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
