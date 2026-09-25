from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Experiments/kaggle/day1_diagnostics"


def cell(cell_type, source):
    item = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        item.update({"execution_count": None, "outputs": []})
    return item


def main():
    sources = {
        "hsi_runner.py": ROOT / "Submission/submission01_yolo16M.py",
        "build_folds.py": ROOT / "Experiments/build_folds.py",
        "evaluate_size_ap.py": ROOT / "Experiments/evaluation/evaluate_size_ap.py",
        "run_day1_diagnostics.py": ROOT / "Experiments/run_day1_diagnostics.py",
    }
    write_sources = "from pathlib import Path\n"
    for name, path in sources.items():
        write_sources += f"Path('/kaggle/working/{name}').write_text({path.read_text()!r})\n"
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

subprocess.run([sys.executable, '/kaggle/working/run_day1_diagnostics.py'], check=True)
"""
    notebook = {
        "cells": [
            cell("markdown", "# Roadmap Day 1 diagnostics\n\nBuilds fixed folds and measures original-coordinate size AP for the saved YOLO, RT-DETR-B, and RT-DETR-C checkpoints. It does not submit predictions."),
            cell("code", setup),
            cell("code", write_sources),
            cell("code", execute),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "sourceType": "notebook", "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": "itsasup/hyperspectral-roadmap-day-1-diagnostics",
        "title": "Hyperspectral Roadmap Day 1 Diagnostics",
        "code_file": "roadmap_day1_diagnostics.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["hyperspectral", "object-detection", "diagnostics"],
        "dataset_sources": [
            "itsasup/hyperspectral-d01-base-data-col-fix",
            "itsasup/working-backup-plan-c-hod",
        ],
        "kernel_sources": [
            "itsasup/hyperspectral-plan-b-v2-rt-detr-16-band",
            "itsasup/hyperspectral-plan-a-yolo-recovery",
        ],
        "competition_sources": [],
        "model_sources": [],
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "roadmap_day1_diagnostics.ipynb").write_text(json.dumps(notebook, indent=1))
    (TARGET / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
