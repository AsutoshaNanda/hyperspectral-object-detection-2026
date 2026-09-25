import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
PRE = ROOT / "Experiments/preprocessing"
for name in ("normalization", "mnf", "spectral_filters"):
    spec = importlib.util.spec_from_file_location(name, PRE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys
    sys.modules[name] = module
spec = importlib.util.spec_from_file_location("pipeline", PRE / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def cubes(seed=7):
    rng = np.random.default_rng(seed)
    return [rng.normal(100 + index, 15, size=(12, 14, 16)).astype(np.float32) for index in range(4)]


def test_signed_methods_preserve_zero_reference_and_produce_uint8():
    for method in ("N1", "N3"):
        state = pipeline.fit_pipeline(cubes(), method)
        assert state.signed_encoding["method"] == "symmetric_signed"
        assert state.signed_encoding["zero_encoded_as"] == 0.5
        result = pipeline.apply_pipeline(cubes()[0], state, output="uint8")
        assert result.dtype == np.uint8
        assert result.shape == (12, 14, 16)


def test_pipeline_state_is_training_fitted_and_round_trips(tmp_path):
    state = pipeline.fit_pipeline(cubes(), "N4", spectral_method="SP2", savgol_window=5, savgol_order=2)
    path = tmp_path / "state.json"
    pipeline.save_pipeline_state(state, path)
    loaded = pipeline.load_pipeline_state(path)
    np.testing.assert_array_equal(
        pipeline.apply_pipeline(cubes()[1], state, output="uint8"),
        pipeline.apply_pipeline(cubes()[1], loaded, output="uint8"),
    )
    assert loaded.fit_scope == "training_fold"


def test_mnf_pipeline_requires_explicit_validation_selected_components():
    try:
        pipeline.fit_pipeline(cubes(), "N0", spectral_method="SP1")
    except ValueError as error:
        assert "mnf_components" in str(error)
    else:
        raise AssertionError("SP1 must require a component count")


def test_mnf_pipeline_reconstructs_sixteen_bands():
    state = pipeline.fit_pipeline(cubes(), "N0", spectral_method="SP1", mnf_components=12)
    result = pipeline.apply_pipeline(cubes()[0], state)
    assert result.shape[-1] == 16
    assert np.isfinite(result).all()
