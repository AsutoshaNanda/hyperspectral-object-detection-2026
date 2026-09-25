from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Experiments/kaggle/top10_screens"


SCREENS = [
    {"id": "screen-control-f0", "extra": []},
    {"id": "s1b-rgb-mean-f0", "extra": ["--stem-init", "rgb_mean"]},
    {"id": "n1-zscore-f0", "extra": ["--normalization", "N1"]},
    {"id": "n3-snv-f0", "extra": ["--normalization", "N3"]},
    {"id": "h1b-spectral-copy-f0", "extra": ["--spectral-aug-copies", "1"]},
    {"id": "m1d-multiscale010-f0", "extra": ["--multi-scale", "0.10"]},
    {"id": "p1-img1024-f0", "imgsz": 1024, "extra": []},
]


def cell(cell_type, source):
    result = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook(config):
    experiment_id = config["id"]
    imgsz = config.get("imgsz", 640)
    extra = repr(config["extra"])
    files = {
        "hsi_runner.py": ROOT / "Submission/submission01_yolo16M.py",
        "build_folds.py": ROOT / "Experiments/build_folds.py",
        "normalization.py": ROOT / "Experiments/preprocessing/normalization.py",
        "mnf.py": ROOT / "Experiments/preprocessing/mnf.py",
        "spectral_filters.py": ROOT / "Experiments/preprocessing/spectral_filters.py",
        "pipeline.py": ROOT / "Experiments/preprocessing/pipeline.py",
    }
    write_sources = "from pathlib import Path\n"
    for name, path in files.items():
        write_sources += f"Path('/kaggle/working/{name}').write_text({path.read_text()!r})\n"
    setup = """import importlib.util
import subprocess
import sys

if importlib.util.find_spec('ultralytics') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'ultralytics==8.4.147', 'ultralytics-thop'], check=True)
"""
    run = f"""import hashlib
import json
from pathlib import Path
import subprocess
import sys

data_matches = list(Path('/kaggle/input').rglob('hyperspectral-2026'))
legacy_matches = list(Path('/kaggle/input').rglob('hsi_plan_c_ratio/split_manifest.json'))
assert len(data_matches) == 1, data_matches
assert len(legacy_matches) == 1, legacy_matches
base = data_matches[0]
fold_dir = Path('/kaggle/working/folds')
subprocess.run([
    sys.executable, '/kaggle/working/build_folds.py',
    '--xml-dir', str(base / 'data_train/data_train/Annotations/VIS'),
    '--cube-dir', str(base / 'data_train/data_train/VIS'),
    '--class-file', str(base / 'class.txt'),
    '--legacy-split', str(legacy_matches[0]),
    '--output-dir', str(fold_dir),
    '--seed', '42',
], check=True)
work = Path('/kaggle/working/{experiment_id}')
command = [
    sys.executable, '/kaggle/working/hsi_runner.py',
    '--train-cubes', str(base / 'data_train/data_train/VIS'),
    '--test-cubes', str(base / 'data_test/data_test/VIS'),
    '--train-xml', str(base / 'data_train/data_train/Annotations/VIS'),
    '--class-file', str(base / 'class.txt'),
    '--workdir', str(work),
    '--cache-dir', '/kaggle/temp/{experiment_id}',
    '--groups-file', str(fold_dir / 'groups.csv'),
    '--split-manifest', str(fold_dir / 'fold0.json'),
    '--family', 'rtdetr', '--model', 'rtdetr-l.pt',
    '--epochs', '10', '--imgsz', '{imgsz}', '--batch', '2', '--workers', '2',
    '--device', '0', '--seed', '42', '--input-policy', 'square_pad', '--stem-init', 'random',
    '--run-name', '{experiment_id}', '--conf', '0.001', '--iou', '0.65', '--max-det', '300',
    '--design-only',
] + {extra}
subprocess.run(command, check=True)
evaluation = json.loads((work / 'evaluation_base.json').read_text())
record = {{
    'experiment_id': '{experiment_id}',
    'screen_budget_epochs': 10,
    'fold': 0,
    'split_hash': hashlib.sha256((fold_dir / 'fold0.json').read_bytes()).hexdigest(),
    'code_hash': hashlib.sha256(Path('/kaggle/working/hsi_runner.py').read_bytes()).hexdigest(),
    'evaluation': evaluation,
    'submission_created': False,
}}
(work / 'screen_evidence.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
"""
    return {
        "cells": [
            cell("markdown", f"# {experiment_id}\n\nMatched 10-epoch fold-zero screen. It cannot promote directly to the final model and creates no submission."),
            cell("code", setup),
            cell("code", write_sources),
            cell("code", run),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = []
    for config in SCREENS:
        slug = "hsi-" + config["id"]
        folder = TARGET / config["id"]
        folder.mkdir(parents=True, exist_ok=True)
        code_file = slug + ".ipynb"
        (folder / code_file).write_text(json.dumps(notebook(config), indent=1))
        metadata = {
            "id": f"itsasup/{slug}", "title": slug, "code_file": code_file,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": [],
            "dataset_sources": ["itsasup/hyperspectral-d01-base-data-col-fix", "itsasup/working-backup-plan-c-hod"],
            "kernel_sources": [], "competition_sources": [], "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append({"id": config["id"], "kernel": metadata["id"], "folder": str(folder.relative_to(ROOT))})
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"screens": len(manifest), "target": str(TARGET)}, indent=2))


if __name__ == "__main__":
    main()
