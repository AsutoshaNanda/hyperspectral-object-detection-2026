from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Experiments/kaggle/sam1_diagnostic"


def cell(cell_type, source):
    result = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def main():
    files = {
        "hsi_runner.py": ROOT / "Submission/submission01_yolo16M.py",
        "build_folds.py": ROOT / "Experiments/build_folds.py",
        "evaluate_material_pairs.py": ROOT / "Experiments/evaluation/evaluate_material_pairs.py",
        "run_sam1_diagnostic.py": ROOT / "Experiments/run_sam1_diagnostic.py",
    }
    sources = "from pathlib import Path\n"
    for name, path in files.items():
        sources += f"Path('/kaggle/working/{name}').write_text({path.read_text()!r})\n"
    notebook = {
        "cells": [
            cell("markdown", "# SAM1 material-pair diagnostic\n\nThree-fold training-prototype and validation-only spectral-angle evaluation. No model training and no submission."),
            cell("code", sources),
            cell("code", "import subprocess, sys\nsubprocess.run([sys.executable, '/kaggle/working/run_sam1_diagnostic.py'], check=True)\n"),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"},
                     "kaggle": {"accelerator": "none", "isInternetEnabled": False, "language": "python", "sourceType": "notebook", "isGpuEnabled": False}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": "itsasup/hsi-sam1-material-pair-diagnostic", "title": "hsi-sam1-material-pair-diagnostic",
        "code_file": "hsi_sam1_material_pair_diagnostic.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": ["itsasup/hyperspectral-d01-base-data-col-fix", "itsasup/working-backup-plan-c-hod"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / metadata["code_file"]).write_text(json.dumps(notebook, indent=1))
    (TARGET / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
