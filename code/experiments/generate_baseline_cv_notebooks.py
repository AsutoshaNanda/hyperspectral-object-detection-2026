from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Experiments/kaggle/baseline_cv"


CONFIGS = [
    {"family": "rtdetr", "model": "rtdetr-l.pt", "epochs": 50, "imgsz": 640, "batch": 2,
     "input_policy": "square_pad", "stem_init": "random", "prefix": "v1-rtdetr-c"},
    {"family": "rtdetr", "model": "rtdetr-l.pt", "epochs": 50, "imgsz": 640, "batch": 2,
     "input_policy": "native", "stem_init": "random", "prefix": "v1-rtdetr-b"},
    {"family": "yolo", "model": "yolo26m.pt", "epochs": 80, "imgsz": 1024, "batch": 4,
     "input_policy": "native", "stem_init": "random", "prefix": "v1-yolo-a"},
]


def cell(cell_type, source):
    result = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook(config, fold):
    write_sources = "from pathlib import Path\n"
    for name, path in {
        "hsi_runner.py": ROOT / "Submission/submission01_yolo16M.py",
        "build_folds.py": ROOT / "Experiments/build_folds.py",
    }.items():
        write_sources += f"Path('/kaggle/working/{name}').write_text({path.read_text()!r})\n"
    experiment_id = f"{config['prefix']}-f{fold}"
    run = f"""import hashlib
import json
from pathlib import Path
import subprocess
import sys

data_matches = list(Path('/kaggle/input').rglob('hyperspectral-2026'))
backup_matches = list(Path('/kaggle/input').rglob('hsi_plan_c_ratio/split_manifest.json'))
assert len(data_matches) == 1, data_matches
assert len(backup_matches) == 1, backup_matches
base = data_matches[0]
legacy_split = backup_matches[0]
fold_dir = Path('/kaggle/working/folds')
subprocess.run([
    sys.executable, '/kaggle/working/build_folds.py',
    '--xml-dir', str(base / 'data_train/data_train/Annotations/VIS'),
    '--cube-dir', str(base / 'data_train/data_train/VIS'),
    '--class-file', str(base / 'class.txt'),
    '--legacy-split', str(legacy_split),
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
    '--split-manifest', str(fold_dir / 'fold{fold}.json'),
    '--family', '{config['family']}',
    '--model', '{config['model']}',
    '--epochs', '{config['epochs']}',
    '--imgsz', '{config['imgsz']}',
    '--batch', '{config['batch']}',
    '--workers', '2',
    '--device', '0',
    '--seed', '42',
    '--input-policy', '{config['input_policy']}',
    '--stem-init', '{config['stem_init']}',
    '--run-name', '{experiment_id}',
    '--conf', '0.001',
    '--iou', '0.65',
    '--max-det', '300',
    '--design-only',
]
subprocess.run(command, check=True)
evidence = {{
    'experiment_id': '{experiment_id}',
    'parent_experiment_id': None,
    'fold': {fold},
    'status': 'measured_design_fold',
    'code_hash': hashlib.sha256(Path('/kaggle/working/hsi_runner.py').read_bytes()).hexdigest(),
    'fold_hash': hashlib.sha256((fold_dir / 'fold{fold}.json').read_bytes()).hexdigest(),
    'groups_hash': hashlib.sha256((fold_dir / 'groups.csv').read_bytes()).hexdigest(),
    'evaluation': json.loads((work / 'evaluation_base.json').read_text()),
    'selection': json.loads((work / 'selection.json').read_text()),
}}
(work / 'evidence_record.json').write_text(json.dumps(evidence, indent=2))
print(json.dumps(evidence, indent=2))
"""
    setup = """import importlib.util
import subprocess
import sys

if importlib.util.find_spec('ultralytics') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'ultralytics==8.4.147', 'ultralytics-thop'], check=True)
"""
    return {
        "cells": [
            cell("markdown", f"# {experiment_id}\n\nFull-budget fixed-fold baseline. Design-only: validation is measured; the previously inspected holdout and competition test are not evaluated."),
            cell("code", setup),
            cell("code", write_sources),
            cell("code", run),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "sourceType": "notebook", "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = []
    for config in CONFIGS:
        for fold in range(3):
            experiment_id = f"{config['prefix']}-f{fold}"
            folder = TARGET / experiment_id
            folder.mkdir(parents=True, exist_ok=True)
            file_name = experiment_id + ".ipynb"
            (folder / file_name).write_text(json.dumps(notebook(config, fold), indent=1))
            metadata = {
                "id": f"itsasup/hsi-{experiment_id}",
                "title": f"hsi-{experiment_id}",
                "code_file": file_name,
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
            (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            manifest.append({"experiment_id": experiment_id, "folder": str(folder.relative_to(ROOT)), "kernel": metadata["id"], "config": config, "fold": fold})
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"notebooks": len(manifest), "target": str(TARGET)}, indent=2))


if __name__ == "__main__":
    main()
