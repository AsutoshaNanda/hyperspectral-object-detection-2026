import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from Experiments.architecture_dataset import build_dataset
from Submission import submission01_yolo16M as hsi_runner


ROOT = Path(__file__).resolve().parents[1]


TOP10 = {
    "L1b": "Experiments/kaggle/localization_tails/hsi-l1-tail-10e/hsi-l1-tail-10e.ipynb",
    "L1c": "Experiments/kaggle/localization_tails/hsi-l1-tail-12e/hsi-l1-tail-12e.ipynb",
    "L1d": "Experiments/kaggle/localization_tails/hsi-l1-tail-15e/hsi-l1-tail-15e.ipynb",
    "S1b": "Experiments/kaggle/top10_screens/s1b-rgb-mean-f0/hsi-s1b-rgb-mean-f0.ipynb",
    "N1": "Experiments/kaggle/top10_screens/n1-zscore-f0/hsi-n1-zscore-f0.ipynb",
    "H1b": "Experiments/kaggle/top10_screens/h1b-spectral-copy-f0/hsi-h1b-spectral-copy-f0.ipynb",
    "M1d": "Experiments/kaggle/top10_screens/m1d-multiscale010-f0/hsi-m1d-multiscale010-f0.ipynb",
    "P1": "Experiments/kaggle/top10_screens/p1-img1024-f0/hsi-p1-img1024-f0.ipynb",
    "DF1": "Experiments/kaggle/architecture_screens/df1-dfine-s-f0/hsi-df1-dfine-s-f0.ipynb",
    "E1a": "Experiments/kaggle/architecture_screens/e1a-rtdetrv2-r18-f0/hsi-e1a-rtdetrv2-r18-f0.ipynb",
}


def test_all_top10_notebooks_exist_and_compile():
    assert len(TOP10) == 10
    for experiment_id, relative_path in TOP10.items():
        path = ROOT / relative_path
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        assert notebook["cells"], experiment_id
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{path}:cell-{index}", "exec")
        metadata_path = path.parent / "kernel-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["code_file"] == path.name
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] is True
        assert metadata["competition_sources"] == []


def test_architecture_notebooks_lock_16_bands_and_18_classes():
    for experiment_id in ("DF1", "E1a"):
        source = (ROOT / TOP10[experiment_id]).read_text()
        assert "input_channels': 16" in source
        assert "classes': 18" in source
        assert "rgb_mean_with_3_over_16_correction" in source
        assert "submission_created': False" in source
        assert "fold0.json" in source


def test_architecture_dataset_builds_16_band_coco_without_holdout(tmp_path, monkeypatch):
    cube_dir = tmp_path / "cubes"
    xml_dir = tmp_path / "xml"
    cube_dir.mkdir()
    xml_dir.mkdir()
    for stem in ("1", "2"):
        raw = np.arange(64, dtype=np.uint8).reshape(8, 8) + int(stem)
        Image.fromarray(raw).save(cube_dir / f"{stem}.png")
        (xml_dir / f"{stem}.xml").write_text(
            "<annotation><size><width>2</width><height>2</height></size>"
            "<object><name>c0</name><bndbox><xmin>0</xmin><ymin>0</ymin>"
            "<xmax>1</xmax><ymax>1</ymax></bndbox></object></annotation>"
        )
    class_file = tmp_path / "classes.txt"
    class_file.write_text("\n".join(f"c{i}" for i in range(18)) + "\n")
    split_file = tmp_path / "fold0.json"
    split_file.write_text(json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}))
    output_dir = tmp_path / "output"
    monkeypatch.setitem(sys.modules, "hsi_runner", hsi_runner)
    args = type(
        "Args",
        (),
        {
            "cube_dir": str(cube_dir),
            "xml_dir": str(xml_dir),
            "class_file": str(class_file),
            "split_manifest": str(split_file),
            "output_dir": str(output_dir),
            "seed": 42,
        },
    )()
    evidence = build_dataset(args)
    train_array = np.load(output_dir / "images/train/1.npy", allow_pickle=False)
    val_payload = json.loads((output_dir / "annotations/instances_val.json").read_text())
    assert train_array.shape == (16, 2, 2)
    assert len(val_payload["categories"]) == 18
    assert val_payload["annotations"][0]["category_id"] == 0
    assert evidence["holdout_images_not_used"] == 1
