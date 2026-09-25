from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Experiments/kaggle/localization_tails"


def cell(cell_type, source):
    result = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def make_notebook(epochs):
    experiment_id = f"l1-tail-{epochs}e-f0"
    runner = (ROOT / "Submission/submission01_yolo16M.py").read_text()
    write_runner = f"from pathlib import Path\nPath('/kaggle/working/hsi_runner.py').write_text({runner!r})\n"
    setup = """import importlib.util
import subprocess
import sys

if importlib.util.find_spec('ultralytics') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'ultralytics==8.4.147', 'ultralytics-thop'], check=True)
"""
    run = f"""from pathlib import Path
import subprocess
import sys

data_matches = list(Path('/kaggle/input').rglob('hyperspectral-2026'))
plan_matches = list(Path('/kaggle/input').rglob('v1-rtdetr-c-f0/split_manifest.json'))
assert len(data_matches) == 1, data_matches
assert len(plan_matches) == 1, plan_matches
base = data_matches[0]
plan = plan_matches[0].parent
weights = list(plan.rglob('best.pt'))
assert len(weights) == 1, weights
import hashlib
assert hashlib.sha256(weights[0].read_bytes()).hexdigest() == 'e3ef9ab5ee1f69e97166cc0266d4cab72c9f5aef97fb9211f1b8c2d6028ab6fb'
work = Path('/kaggle/working/{experiment_id}')
command = [
    sys.executable, '/kaggle/working/hsi_runner.py',
    '--train-cubes', str(base / 'data_train/data_train/VIS'),
    '--test-cubes', str(base / 'data_test/data_test/VIS'),
    '--train-xml', str(base / 'data_train/data_train/Annotations/VIS'),
    '--class-file', str(base / 'class.txt'),
    '--workdir', str(work),
    '--cache-dir', '/kaggle/temp/{experiment_id}',
    '--split-manifest', str(plan / 'split_manifest.json'),
    '--checkpoint', str(weights[0]),
    '--family', 'rtdetr',
    '--model', 'rtdetr-l.pt',
    '--epochs', '50',
    '--imgsz', '640',
    '--batch', '2',
    '--workers', '2',
    '--device', '0',
    '--seed', '42',
    '--input-policy', 'square_pad',
    '--stem-init', 'random',
    '--localization-epochs', '{epochs}',
    '--localization-lr0', '0.0001',
    '--run-name', '{experiment_id}',
    '--conf', '0.001',
    '--iou', '0.65',
    '--max-det', '300',
    '--design-only',
]
subprocess.run(command, check=True)
print((work / 'selection.json').read_text())
"""
    return {
        "cells": [
            cell("markdown", f"# {experiment_id}\n\nValidation-only localization tail from the hash-matched Plan-C checkpoint. No submission is created."),
            cell("code", setup),
            cell("code", write_runner),
            cell("code", run),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"},
                     "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "sourceType": "notebook", "isGpuEnabled": True}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for epochs in (10, 12, 15):
        slug = f"hsi-l1-tail-{epochs}e"
        folder = TARGET / slug
        folder.mkdir(parents=True, exist_ok=True)
        code_file = slug + ".ipynb"
        (folder / code_file).write_text(json.dumps(make_notebook(epochs), indent=1))
        metadata = {
            "id": f"itsasup/{slug}", "title": slug, "code_file": code_file,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
            "dataset_sources": ["itsasup/hyperspectral-d01-base-data-col-fix"],
            "kernel_sources": ["itsasup/hsi-v1-rtdetr-c-f0"], "competition_sources": [], "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
