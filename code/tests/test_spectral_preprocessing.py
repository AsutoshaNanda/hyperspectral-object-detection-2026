import json

import numpy as np
import pytest

from Experiments.evaluation.evaluate_material_pairs import (
    evaluate_material_pairs,
    extract_annotation_spectra,
    extract_box_spectrum,
)
from Experiments.preprocessing.mnf import (
    fit_mnf,
    load_mnf_state,
    reconstruct_mnf,
    save_mnf_state,
)
from Experiments.preprocessing.normalization import (
    apply_normalization,
    fit_normalization,
    load_normalization_state,
    save_normalization_state,
)
from Experiments.preprocessing.spectral_augmentation import (
    SpectralAugmentationConfig,
    make_training_copies,
)
from Experiments.preprocessing.spectral_filters import (
    apply_msc,
    fit_msc,
    load_msc_state,
    save_msc_state,
    savitzky_golay,
)


def _training_cube(seed=4, shape=(12, 10, 16)):
    rng = np.random.default_rng(seed)
    bands = np.linspace(2.0, 6.0, 16)
    return (rng.normal(size=shape) * 0.2 + bands).astype(np.float32)


@pytest.mark.parametrize("method", ["N0", "N1", "N2", "N3", "N4"])
def test_normalizations_preserve_shape_source_and_state_round_trip(tmp_path, method):
    train = _training_cube()
    evaluation = _training_cube(8, (5, 6, 16))
    original = evaluation.copy()
    state = fit_normalization(train, method)
    state_path = tmp_path / f"{method}.json"
    save_normalization_state(state, state_path)
    restored = load_normalization_state(state_path)
    transformed = apply_normalization(evaluation, restored)
    assert transformed.shape == evaluation.shape
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(evaluation, original)
    assert json.loads(state_path.read_text())["fit_scope"] == "training_fold"
    if method == "N2":
        assert restored.parameters["percentiles"] == [2.0, 98.0]
    if method == "N3":
        np.testing.assert_allclose(transformed.mean(axis=-1), 0.0, atol=2e-6)
        np.testing.assert_allclose(transformed.std(axis=-1), 1.0, atol=2e-6)
    if method == "N4":
        np.testing.assert_allclose(transformed.sum(axis=-1), 1.0, atol=2e-6)


def test_normalization_fit_is_unchanged_by_evaluation_data():
    train = _training_cube()
    state = fit_normalization(train, "N1")
    extreme_evaluation = np.full((3, 4, 16), 1e6, dtype=np.float32)
    apply_normalization(extreme_evaluation, state)
    assert state == fit_normalization(train, "N1")


def test_mnf_reconstructs_16_bands_and_serializes_without_fitting_evaluation(tmp_path):
    train = _training_cube(shape=(18, 14, 16))
    evaluation = _training_cube(9, (7, 5, 16))
    original = evaluation.copy()
    state = fit_mnf(train)
    state_path = tmp_path / "mnf.json"
    save_mnf_state(state, state_path)
    restored = load_mnf_state(state_path)
    full = reconstruct_mnf(evaluation, restored, 16)
    reduced = reconstruct_mnf(evaluation, restored, 8)
    np.testing.assert_allclose(full, evaluation, atol=2e-5)
    assert reduced.shape == evaluation.shape
    assert restored.fit_scope == "training_fold"
    np.testing.assert_array_equal(evaluation, original)


def test_spectral_filters_use_only_wavelength_axis_and_training_reference(tmp_path):
    train = _training_cube()
    evaluation = _training_cube(6, (4, 3, 16))
    original = evaluation.copy()
    smoothed = savitzky_golay(evaluation, window_length=5, polyorder=2)
    assert smoothed.shape == evaluation.shape
    np.testing.assert_array_equal(evaluation, original)
    state = fit_msc(train)
    state_path = tmp_path / "msc.json"
    save_msc_state(state, state_path)
    restored = load_msc_state(state_path)
    corrected = apply_msc(evaluation, restored)
    assert corrected.shape == evaluation.shape
    assert restored.fit_scope == "training_fold"
    with pytest.raises(ValueError):
        savitzky_golay(evaluation, window_length=4, polyorder=2)


def test_spectral_augmentation_is_deterministic_targetable_and_non_mutating():
    cube = _training_cube(shape=(5, 7, 16))
    original = cube.copy()
    config = SpectralAugmentationConfig(copies=1)
    first = make_training_copies(cube, 42, "sample", config, class_ids=(3,), target_classes={3})
    second = make_training_copies(cube, 42, "sample", config, class_ids=(3,), target_classes={3})
    skipped = make_training_copies(cube, 42, "sample", config, class_ids=(3,), target_classes={7})
    assert len(first) == 1
    np.testing.assert_array_equal(first[0], second[0])
    assert first[0].shape == cube.shape
    assert skipped == []
    np.testing.assert_array_equal(cube, original)
    with pytest.raises(ValueError):
        make_training_copies(cube, 42, "sample", config, split="validation")


def test_box_spectrum_shrinks_and_annotation_extraction_preserves_16_bands():
    cube = np.zeros((6, 6, 16), dtype=np.float32)
    cube[1:5, 1:5] = np.arange(16, dtype=np.float32)
    cube[1, 1:5] = 1000
    spectrum = extract_box_spectrum(cube, [1, 1, 4, 4], shrink_fraction=0.25)
    np.testing.assert_array_equal(spectrum, np.arange(16, dtype=np.float32))
    spectra, labels, ids = extract_annotation_spectra(
        {"a": cube},
        [{"id": 9, "image_id": "a", "category_id": 2, "bbox": [1, 1, 4, 4]}],
        shrink_fraction=0.25,
    )
    assert spectra.shape == (1, 16)
    assert labels.tolist() == [2]
    assert ids == [9]


def test_sam_pair_report_uses_training_only_prototypes_and_thresholds():
    rng = np.random.default_rng(12)
    first = np.linspace(1.0, 3.0, 16)
    second = np.linspace(3.0, 1.0, 16)
    train_x = np.vstack((first + rng.normal(0, 0.02, (20, 16)), second + rng.normal(0, 0.02, (20, 16))))
    train_y = np.repeat([0, 1], 20)
    eval_x = np.vstack((first + rng.normal(0, 0.03, (10, 16)), second + rng.normal(0, 0.03, (10, 16))))
    eval_y = np.repeat([0, 1], 10)
    pairs = (("Apple", "Apple Plastic"),)
    report = evaluate_material_pairs(train_x, train_y, eval_x, eval_y, ["Apple", "Apple Plastic"], pairs)
    changed_eval = eval_x + 50.0
    changed = evaluate_material_pairs(train_x, train_y, changed_eval, eval_y, ["Apple", "Apple Plastic"], pairs)
    pair = report["pairs"][0]
    assert pair["evaluation_auc"] > 0.99
    assert pair["evaluation_accuracy"] > 0.95
    assert report["prototypes"] == changed["prototypes"]
    assert pair["threshold"] == changed["pairs"][0]["threshold"]
    assert pair["threshold_fit_scope"] == "training_fold"


def test_sam_class_names_accept_dataset_underscores():
    first = np.tile(np.linspace(1, 2, 16), (4, 1))
    second = np.tile(np.linspace(2, 1, 16), (4, 1))
    spectra = np.concatenate((first, second))
    labels = np.array([0] * 4 + [1] * 4)
    report = evaluate_material_pairs(
        spectra,
        labels,
        spectra,
        labels,
        ["apple", "apple_plastic"],
        pairs=(("Apple", "Apple Plastic"),),
    )
    assert report["pairs"][0]["class_ids"] == [0, 1]


def test_all_preprocessing_rejects_non_16_band_data():
    bad = np.ones((4, 4, 15), dtype=np.float32)
    with pytest.raises(ValueError):
        fit_normalization(bad, "N0")
    with pytest.raises(ValueError):
        fit_mnf(bad)
    with pytest.raises(ValueError):
        savitzky_golay(bad)
    with pytest.raises(ValueError):
        make_training_copies(bad, 1, "bad")
