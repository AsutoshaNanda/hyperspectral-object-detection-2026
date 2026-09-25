import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "Submission" / "submission01_yolo16M.py"
SPEC = importlib.util.spec_from_file_location("hsi_runner", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_x2cube_preserves_each_four_by_four_cell():
    raw = np.arange(64).reshape(8, 8)
    cube = RUNNER.X2Cube(raw)
    assert cube.shape == (2, 2, 16)
    np.testing.assert_array_equal(cube[0, 0], raw[:4, :4].reshape(-1))
    np.testing.assert_array_equal(cube[1, 1], raw[4:, 4:].reshape(-1))


def test_square_padding_and_box_round_trip():
    cube = np.arange(2 * 4 * 16, dtype=np.float32).reshape(2, 4, 16)
    lo = np.zeros(16, dtype=np.float32)
    hi = np.full(16, cube.max(), dtype=np.float32)
    prepared, geometry = RUNNER.prepare_cube(cube, lo, hi, "square_pad")
    assert prepared.shape == (4, 4, 16)
    assert geometry["pad_top"] == 1
    rows = [(3, 0.5, 0.5, 0.5, 0.5)]
    transformed = RUNNER.transform_rows(rows, geometry)
    assert transformed[0] == (3, 0.5, 0.5, 0.5, 0.25)
    model_box = np.array([[1.0, 1.5, 3.0, 2.5]], dtype=np.float32)
    restored = RUNNER.restore_boxes(model_box, geometry)
    np.testing.assert_allclose(restored, [[1.0, 0.5, 3.0, 1.5]])


def test_prepare_cube_accepts_training_fitted_signed_pipeline():
    import sys

    preprocessing = Path(__file__).parents[1] / "Experiments" / "preprocessing"
    sys.path.insert(0, str(preprocessing))
    from pipeline import fit_pipeline

    rng = np.random.default_rng(5)
    training = [rng.normal(50, 8, size=(8, 10, 16)).astype(np.float32) for _ in range(3)]
    state = fit_pipeline(training, "N1")
    prepared, geometry = RUNNER.prepare_cube(training[0], None, None, "native", state)
    assert prepared.dtype == np.uint8
    assert prepared.shape == training[0].shape
    assert geometry["original_height"] == 8


def test_spectral_augmentation_is_reproducible_and_spatially_aligned():
    cube = np.arange(5 * 7 * 16, dtype=np.float32).reshape(5, 7, 16)
    first = RUNNER.spectral_augment(cube, RUNNER.stable_rng(42, "17", 0), 0.05, 0.03, 0.01)
    second = RUNNER.spectral_augment(cube, RUNNER.stable_rng(42, "17", 0), 0.05, 0.03, 0.01)
    assert first.shape == cube.shape
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, cube)


def test_cv_split_is_disjoint_and_complete():
    stems = [str(i) for i in range(120)]
    boxes = {stem: [(i % 6, 0.5, 0.5, 0.2, 0.2)] for i, stem in enumerate(stems)}
    groups = {stem: stem for stem in stems}
    train, val, holdout = RUNNER.cv_split(stems, boxes, groups, 42, 3, 0)
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(holdout)
    assert set(val).isdisjoint(holdout)
    assert set(train) | set(val) | set(holdout) == set(stems)


def test_parse_class_ids():
    assert RUNNER.parse_class_ids("16,5,16") == [5, 16]
    assert RUNNER.parse_class_ids("") == []


def test_rgb_stem_expansion_preserves_equal_channel_activation_scale():
    source = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, bias=False))
    target = torch.nn.Sequential(torch.nn.Conv2d(16, 4, 3, bias=False))
    torch.manual_seed(7)
    source[0].weight.data.normal_()
    RUNNER.expand_rgb_stem(target, source)
    plane = torch.randn(2, 1, 9, 9)
    source_output = source(plane.repeat(1, 3, 1, 1))
    target_output = target(plane.repeat(1, 16, 1, 1))
    torch.testing.assert_close(source_output, target_output)


def test_dataset_build_applies_rtdetr_padding_to_images_and_labels(tmp_path):
    train_cubes = tmp_path / "train"
    test_cubes = tmp_path / "test"
    xml_dir = tmp_path / "xml"
    train_cubes.mkdir()
    test_cubes.mkdir()
    xml_dir.mkdir()
    class_file = tmp_path / "classes.txt"
    class_file.write_text("\n".join(f"class_{i}" for i in range(18)))
    for i in range(30):
        np.save(train_cubes / f"{i}.npy", np.full((4, 8, 16), i + 1, dtype=np.float32))
        (xml_dir / f"{i}.xml").write_text(
            f"<annotation><size><width>8</width><height>4</height></size><object><name>class_{i % 18}</name><bndbox><xmin>2</xmin><ymin>1</ymin><xmax>6</xmax><ymax>3</ymax></bndbox></object></annotation>"
        )
    np.save(test_cubes / "100.npy", np.ones((4, 8, 16), dtype=np.float32))
    args = SimpleNamespace(
        workdir=str(tmp_path / "work"),
        cache_dir=None,
        class_file=str(class_file),
        train_cubes=str(train_cubes),
        test_cubes=str(test_cubes),
        ranking_cubes=None,
        train_xml=str(xml_dir),
        groups_file=None,
        split_manifest=None,
        cv_fold_index=None,
        cv_folds=3,
        seed=42,
        family="rtdetr",
        input_policy="auto",
        spectral_aug_classes=[],
        spectral_aug_copies=0,
        spectral_gain=0.05,
        spectral_tilt=0.03,
        spectral_noise=0.01,
    )
    _, _, geometry = RUNNER.build_dataset(args)
    assert geometry["100"] == {
        "original_height": 4,
        "original_width": 8,
        "model_height": 8,
        "model_width": 8,
        "pad_top": 2,
        "pad_left": 0,
    }
    manifest = __import__("json").loads((Path(args.workdir) / "split_manifest.json").read_text())
    stem = manifest["train"][0]
    label = (Path(args.workdir) / "dataset" / "labels" / "train" / f"{stem}.txt").read_text().split()
    np.testing.assert_allclose([float(value) for value in label[1:]], [0.5, 0.5, 0.5, 0.25])


def test_metric_summary_records_per_class_ap75_and_errors():
    box = SimpleNamespace(
        ap_class_index=np.array([0, 1]),
        ap=np.array([0.6, 0.4]),
        ap50=np.array([0.9, 0.8]),
        all_ap=np.array([[0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4], [0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]]),
        p=np.array([0.8, 0.7]),
        r=np.array([0.7, 0.6]),
        map=0.5,
        map50=0.85,
        map75=0.55,
    )
    matrix = np.zeros((19, 19))
    matrix[0, 0] = 4
    matrix[1, 0] = 1
    matrix[0, 18] = 2
    matrix[18, 1] = 3
    summary = RUNNER.metric_summary(SimpleNamespace(box=box, confusion_matrix=SimpleNamespace(matrix=matrix)))
    assert summary["per_class"]["0"]["ap75"] == 0.6
    assert summary["false_positives_per_class"][0] == 2
    assert summary["false_negatives_per_class"][0] == 1
    assert summary["false_negatives_per_class"][1] == 3
