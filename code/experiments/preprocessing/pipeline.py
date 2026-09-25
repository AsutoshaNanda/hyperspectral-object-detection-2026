from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

if __package__:
    from .mnf import MNFState, fit_mnf, reconstruct_mnf
    from .normalization import NormalizationState, apply_normalization, fit_normalization
    from .spectral_filters import MSCState, apply_msc, fit_msc, savitzky_golay
else:
    from mnf import MNFState, fit_mnf, reconstruct_mnf
    from normalization import NormalizationState, apply_normalization, fit_normalization
    from spectral_filters import MSCState, apply_msc, fit_msc, savitzky_golay


@dataclass(frozen=True)
class SpectralPipelineState:
    normalization: dict
    spectral_method: str
    spectral_parameters: dict
    signed_encoding: dict
    fit_scope: str = "training_fold"
    n_bands: int = 16
    version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        state = cls(**value)
        _validate_state(state)
        return state


def fit_pipeline(
    training_cubes,
    normalization_method,
    spectral_method="SP0",
    seed=42,
    samples_per_cube=256,
    mnf_components=None,
    savgol_window=5,
    savgol_order=2,
):
    spectral_method = spectral_method.upper()
    if spectral_method not in ("SP0", "SP1", "SP2", "SP3"):
        raise ValueError("spectral_method must be SP0, SP1, SP2, or SP3")
    if samples_per_cube < 32:
        raise ValueError("samples_per_cube must be at least 32")
    rng = np.random.default_rng(seed)
    sampled = []
    patches = []
    for source in training_cubes:
        cube = np.asarray(source, dtype=np.float32)
        if cube.ndim != 3 or cube.shape[-1] != 16 or not np.isfinite(cube).all():
            raise ValueError("Each training cube must be a finite H x W x 16 array")
        flat = cube.reshape(-1, 16)
        count = min(samples_per_cube, len(flat))
        sampled.append(flat[rng.choice(len(flat), count, replace=False)])
        if spectral_method == "SP1":
            side = max(2, int(np.sqrt(count)))
            side = min(side, cube.shape[0], cube.shape[1])
            top = int(rng.integers(0, cube.shape[0] - side + 1))
            left = int(rng.integers(0, cube.shape[1] - side + 1))
            patches.append(cube[top : top + side, left : left + side])
    if not sampled:
        raise ValueError("training_cubes must not be empty")
    sample = np.concatenate(sampled)
    parameters = {
        "seed": int(seed),
        "samples_per_cube": int(samples_per_cube),
        "training_cube_count": len(sampled),
    }
    if spectral_method == "SP1":
        if mnf_components is None:
            raise ValueError("SP1 requires validation-selected mnf_components")
        mnf = fit_mnf(patches)
        parameters.update(mnf_state=mnf.to_dict(), mnf_components=int(mnf_components))
        transformed = reconstruct_mnf(sample, mnf, int(mnf_components))
    elif spectral_method == "SP2":
        parameters.update(window_length=int(savgol_window), polyorder=int(savgol_order))
        transformed = savitzky_golay(sample, window_length=savgol_window, polyorder=savgol_order)
    elif spectral_method == "SP3":
        msc = fit_msc(sample)
        parameters.update(msc_state=msc.to_dict())
        transformed = apply_msc(sample, msc)
    else:
        transformed = sample
    normalization = fit_normalization(transformed, normalization_method)
    normalized = apply_normalization(transformed, normalization)
    encoding = _fit_encoding(normalized, normalization.method)
    state = SpectralPipelineState(
        normalization=normalization.to_dict(),
        spectral_method=spectral_method,
        spectral_parameters=parameters,
        signed_encoding=encoding,
    )
    _validate_state(state)
    return state


def apply_pipeline(cube, state, output="float32"):
    _validate_state(state)
    array = np.asarray(cube, dtype=np.float32)
    if array.ndim < 2 or array.shape[-1] != 16 or not np.isfinite(array).all():
        raise ValueError("cube must be a finite array with 16 bands on its last axis")
    parameters = state.spectral_parameters
    if state.spectral_method == "SP1":
        transformed = reconstruct_mnf(array, MNFState.from_dict(parameters["mnf_state"]), int(parameters["mnf_components"]))
    elif state.spectral_method == "SP2":
        transformed = savitzky_golay(array, window_length=int(parameters["window_length"]), polyorder=int(parameters["polyorder"]))
    elif state.spectral_method == "SP3":
        transformed = apply_msc(array, MSCState.from_dict(parameters["msc_state"]))
    else:
        transformed = array
    normalized = apply_normalization(transformed, NormalizationState.from_dict(state.normalization))
    unit = _encode(normalized, state.signed_encoding)
    if output == "float32":
        return unit.astype(np.float32)
    if output == "uint8":
        return np.rint(unit * 255).astype(np.uint8)
    raise ValueError("output must be float32 or uint8")


def save_pipeline_state(state, path):
    _validate_state(state)
    Path(path).write_text(json.dumps(state.to_dict(), indent=2, allow_nan=False) + "\n")


def load_pipeline_state(path):
    return SpectralPipelineState.from_dict(json.loads(Path(path).read_text()))


def _fit_encoding(normalized, method):
    if method in ("N0", "N2"):
        return {"method": "unit_interval", "zero_encoded_as": 0.0, "fit_scope": "training_fold"}
    if method in ("N1", "N3"):
        bound = max(float(np.percentile(np.abs(normalized), 99.5)), 1e-6)
        return {"method": "symmetric_signed", "bound": bound, "zero_encoded_as": 0.5, "fit_scope": "training_fold"}
    upper = max(float(np.percentile(normalized, 99.5)), 1e-6)
    lower = min(0.0, float(np.percentile(normalized, 0.5)))
    return {"method": "global_affine", "lower": lower, "upper": upper, "zero_encoded_as": float(-lower / (upper - lower)), "fit_scope": "training_fold"}


def _encode(values, encoding):
    if encoding["method"] == "unit_interval":
        result = np.clip(values, 0, 1)
    elif encoding["method"] == "symmetric_signed":
        bound = float(encoding["bound"])
        result = np.clip(values, -bound, bound) / (2 * bound) + 0.5
    else:
        lower, upper = float(encoding["lower"]), float(encoding["upper"])
        result = np.clip((values - lower) / (upper - lower), 0, 1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Storage encoding produced non-finite values")
    return result


def _validate_state(state):
    if state.fit_scope != "training_fold" or state.n_bands != 16:
        raise ValueError("Pipeline state must be fitted on a 16-band training fold")
    NormalizationState.from_dict(state.normalization)
    if state.spectral_method not in ("SP0", "SP1", "SP2", "SP3"):
        raise ValueError("Unknown spectral method")
    method = state.signed_encoding.get("method")
    if method not in ("unit_interval", "symmetric_signed", "global_affine"):
        raise ValueError("Unknown storage encoding")
    if state.signed_encoding.get("fit_scope") != "training_fold":
        raise ValueError("Storage encoding must be training-fold fitted")
    if method == "symmetric_signed" and float(state.signed_encoding.get("bound", 0)) <= 0:
        raise ValueError("Signed encoding bound must be positive")
    if method == "global_affine" and float(state.signed_encoding.get("upper", 0)) <= float(state.signed_encoding.get("lower", 0)):
        raise ValueError("Global encoding range must be positive")
