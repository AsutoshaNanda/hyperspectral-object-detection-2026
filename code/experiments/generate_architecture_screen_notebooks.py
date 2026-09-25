from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Experiments/kaggle/architecture_screens"


CANDIDATES = {
    "df1-dfine-s-f0": {
        "repo": "https://github.com/Peterande/D-FINE.git",
        "revision": "master",
        "subdir": "D-FINE",
        "checkpoint": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "config_path": "configs/dfine/custom/hsi_df1.yml",
        "train_script": "train.py",
        "amp_flag": "--use-amp",
        "backbone_file": "src/nn/backbone/hgnetv2.py",
        "backbone_old": "in_chs=stem_channels[0],",
        "backbone_new": "in_chs=16,",
        "config": """__include__: ['./dfine_hgnetv2_s_custom.yml']
output_dir: /kaggle/working/df1-dfine-s-f0
num_classes: 18
remap_mscoco_category: False
epochs: 10
checkpoint_freq: 1
sync_bn: False
HGNetv2:
  pretrained: False
  freeze_at: -1
  freeze_norm: False
train_dataloader:
  total_batch_size: 2
  num_workers: 2
  drop_last: True
  dataset:
    img_folder: /kaggle/working/hsi_arch_data/images/train
    ann_file: /kaggle/working/hsi_arch_data/annotations/instances_train.json
    transforms:
      type: Compose
      ops:
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
  collate_fn:
    type: BatchImageCollateFunction
    base_size: 640
    base_size_repeat: 1
    stop_epoch: 10
val_dataloader:
  total_batch_size: 2
  num_workers: 2
  drop_last: False
  dataset:
    img_folder: /kaggle/working/hsi_arch_data/images/val
    ann_file: /kaggle/working/hsi_arch_data/annotations/instances_val.json
    transforms:
      type: Compose
      ops:
        - {type: Resize, size: [640, 640]}
""",
    },
    "e1a-rtdetrv2-r18-f0": {
        "repo": "https://github.com/lyuwenyu/RT-DETR.git",
        "revision": "main",
        "subdir": "RT-DETR/rtdetrv2_pytorch",
        "checkpoint": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "config_path": "configs/rtdetrv2/hsi_e1a.yml",
        "train_script": "tools/train.py",
        "amp_flag": "--use-amp",
        "backbone_file": "src/nn/backbone/presnet.py",
        "backbone_old": "[3, ch_in // 2, 3, 2, \"conv1_1\"],",
        "backbone_new": "[16, ch_in // 2, 3, 2, \"conv1_1\"],",
        "config": """__include__: ['./rtdetrv2_r18vd_120e_coco.yml']
output_dir: /kaggle/working/e1a-rtdetrv2-r18-f0
num_classes: 18
remap_mscoco_category: False
epoches: 10
checkpoint_freq: 1
sync_bn: False
PResNet:
  pretrained: False
  freeze_at: -1
  freeze_norm: False
train_dataloader:
  total_batch_size: 2
  num_workers: 2
  drop_last: True
  dataset:
    img_folder: /kaggle/working/hsi_arch_data/images/train
    ann_file: /kaggle/working/hsi_arch_data/annotations/instances_train.json
    transforms:
      type: Compose
      ops:
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
  collate_fn:
    type: BatchImageCollateFunction
    scales: ~
    stop_epoch: 10
val_dataloader:
  total_batch_size: 2
  num_workers: 2
  drop_last: False
  dataset:
    img_folder: /kaggle/working/hsi_arch_data/images/val
    ann_file: /kaggle/working/hsi_arch_data/annotations/instances_val.json
    transforms:
      type: Compose
      ops:
        - {type: Resize, size: [640, 640]}
""",
    },
}


def cell(cell_type, source):
    result = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook(experiment_id, candidate):
    embedded = {
        "hsi_runner.py": ROOT / "Submission/submission01_yolo16M.py",
        "build_folds.py": ROOT / "Experiments/build_folds.py",
        "architecture_dataset.py": ROOT / "Experiments/architecture_dataset.py",
    }
    write_sources = "from pathlib import Path\n"
    for name, path in embedded.items():
        write_sources += f"Path('/kaggle/working/{name}').write_text({path.read_text()!r})\n"
    setup = f"""import importlib.util
import subprocess
import sys
from pathlib import Path

packages = ['faster-coco-eval>=1.6.6', 'tensorboard', 'scipy', 'PyYAML']
if '{experiment_id}'.startswith('df1'):
    packages += ['calflops', 'transformers', 'loguru']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', *packages], check=True)
repo_parent = Path('/kaggle/working/repos')
repo_parent.mkdir(parents=True, exist_ok=True)
repo_root = Path('/kaggle/working/repos/{candidate['subdir']}')
if not repo_root.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', '{candidate['revision']}', '{candidate['repo']}', str(repo_parent / '{candidate['subdir'].split('/')[0]}')], check=True)
assert repo_root.exists(), repo_root
"""
    data = """import subprocess
import sys
from pathlib import Path

base_matches = list(Path('/kaggle/input').rglob('hyperspectral-2026'))
legacy_matches = list(Path('/kaggle/input').rglob('hsi_plan_c_ratio/split_manifest.json'))
assert len(base_matches) == 1, base_matches
assert len(legacy_matches) == 1, legacy_matches
base = base_matches[0]
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
subprocess.run([
    sys.executable, '/kaggle/working/architecture_dataset.py',
    '--cube-dir', str(base / 'data_train/data_train/VIS'),
    '--xml-dir', str(base / 'data_train/data_train/Annotations/VIS'),
    '--class-file', str(base / 'class.txt'),
    '--split-manifest', str(fold_dir / 'fold0.json'),
    '--output-dir', '/kaggle/working/hsi_arch_data',
    '--seed', '42',
], check=True)
"""
    patch_and_train = f"""import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch

repo_root = Path('/kaggle/working/repos/{candidate['subdir']}')
dataset_file = repo_root / 'src/data/dataset/coco_dataset.py'
source = dataset_file.read_text()
source = source.replace('from PIL import Image', 'from PIL import Image\\nimport os\\nimport numpy as np\\nfrom torchvision.tv_tensors import Image as TVImage')
old_load = "image, target = super(FasterCocoDetection, self).__getitem__(idx)"
new_load = (
    "image_id = self.ids[idx]\\n"
    "        info = self.coco.loadImgs(image_id)[0]\\n"
    "        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))\\n"
    "        array = np.load(os.path.join(self.img_folder, info['file_name']), allow_pickle=False)\\n"
    "        image = TVImage(torch.from_numpy(array).float().div_(255.0))"
)
assert source.count(old_load) == 1
source = source.replace(old_load, new_load)
source = source.replace('image.size[::-1]', 'image.shape[-2:]')
source = source.replace('w, h = image.size', 'h, w = image.shape[-2:]')
dataset_file.write_text(source)

backbone_file = repo_root / '{candidate['backbone_file']}'
backbone = backbone_file.read_text()
assert backbone.count({candidate['backbone_old']!r}) == 1
backbone_file.write_text(backbone.replace({candidate['backbone_old']!r}, {candidate['backbone_new']!r}))

config_path = repo_root / '{candidate['config_path']}'
config_path.write_text({candidate['config']!r})

checkpoint_url = '{candidate['checkpoint']}'
downloaded = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', progress=True)
state_dict = downloaded['ema']['module'] if 'ema' in downloaded else downloaded.get('model', downloaded)
candidates = [key for key, value in state_dict.items() if torch.is_tensor(value) and value.ndim == 4 and value.shape[1] == 3]
preferred = [key for key in candidates if any(token in key.lower() for token in ('stem1', 'conv1'))]
assert len(preferred) == 1, {{'all': candidates, 'preferred': preferred}}
stem_key = preferred[0]
rgb = state_dict[stem_key]
state_dict[stem_key] = rgb.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1).mul_(3.0 / 16.0)
patched_checkpoint = Path('/kaggle/working/{experiment_id}-rgb16.pth')
torch.save(downloaded, patched_checkpoint)

output_dir = Path('/kaggle/working/{experiment_id}')
command = [
    sys.executable, str(repo_root / '{candidate['train_script']}'),
    '-c', str(config_path), '-t', str(patched_checkpoint),
    '--seed', '42', '{candidate['amp_flag']}', '--output-dir', str(output_dir),
]
subprocess.run(command, cwd=repo_root, check=True)
checkpoints = sorted(output_dir.rglob('*.pth'))
assert checkpoints, output_dir
record = {{
    'experiment_id': '{experiment_id}',
    'fold': 0,
    'epochs': 10,
    'input_channels': 16,
    'classes': 18,
    'stem_initialization': 'rgb_mean_with_3_over_16_correction',
    'source_repository': '{candidate['repo']}',
    'source_revision_requested': '{candidate['revision']}',
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True).strip(),
    'pretrained_checkpoint': checkpoint_url,
    'adapted_stem_key': stem_key,
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'dataset_evidence': json.loads(Path('/kaggle/working/hsi_arch_data/dataset_evidence.json').read_text()),
    'checkpoints': [str(path) for path in checkpoints],
    'submission_created': False,
}}
(output_dir / 'architecture_evidence.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
"""
    return {
        "cells": [
            cell("markdown", f"# {experiment_id}\n\nOfficial-source 16-band, 18-class, fixed-fold 10-epoch architecture screen. No competition test data or submission is used."),
            cell("code", setup),
            cell("code", write_sources),
            cell("code", data),
            cell("code", patch_and_train),
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
    for experiment_id, candidate in CANDIDATES.items():
        folder = TARGET / experiment_id
        folder.mkdir(parents=True, exist_ok=True)
        code_file = f"hsi-{experiment_id}.ipynb"
        (folder / code_file).write_text(json.dumps(notebook(experiment_id, candidate), indent=1))
        metadata = {
            "id": f"itsasup/hsi-{experiment_id}",
            "title": f"hsi-{experiment_id}",
            "code_file": code_file,
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
        manifest.append({"id": experiment_id, "kernel": metadata["id"], "folder": str(folder.relative_to(ROOT))})
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"architecture_screens": len(manifest), "target": str(TARGET)}, indent=2))


if __name__ == "__main__":
    main()
