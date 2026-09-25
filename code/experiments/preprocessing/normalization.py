import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


N_BANDS = 16
METHODS = ("N0", "N1", "N2", "N3", "N4")


@dataclass(frozen=True)
class NormalizationState:
    method: str
    parameters: dict
    fit_scope: str = "training_fold"
    n_bands: int = N_BANDS
    numeric_storage: str = "float32"
    version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        state = cls(**value)
        _validate_state(state)
        return state


def _validate_array(values, name="spectra"):
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[-1] != N_BANDS:
        raise ValueError(f"{name} must have 16 bands on its last axis")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _training_matrix(training_data):
    if isinstance(training_data, np.ndarray):
        arrays = [training_data]
    else:
        arrays = list(training_data)
    if not arrays:
        raise ValueError("training_data must not be empty")
    flattened = [_validate_array(array, "training_data").reshape(-1, N_BANDS) for array in arrays]
    matrix = np.concatenate(flattened, axis=0).astype(np.float64, copy=False)
    if matrix.shape[0] < 2:
        raise ValueError("training_data must contain at least two spectral vectors")
    return matrix


def _per_band_percentiles(matrix, percentiles):
    low, high = np.percentile(matrix, percentiles, axis=0)
    if np.any(high <= low):
        raise ValueError("Every band must have a positive fitted range")
    return low, high


def fit_normalization(
    training_data: np.ndarray | Iterable[np.ndarray],
    method: str,
    clip_percentiles=(0.5, 99.5),
    robust_percentiles=(2.0, 98.0),
    epsilon=1e-6,
):
    method = method.upper()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    matrix = _training_matrix(training_data)
    parameters = {"epsilon": float(epsilon)}
    if method == "N1":
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.maximum(scale, epsilon)
        parameters.update(mean=mean.tolist(), scale=scale.tolist())
    else:
        percentiles = robust_percentiles if method == "N2" else clip_percentiles
        low, high = _per_band_percentiles(matrix, percentiles)
        parameters.update(
            clip_low=low.tolist(),
            clip_high=high.tolist(),
            percentiles=[float(percentiles[0]), float(percentiles[1])],
        )
        if method == "N2":
            parameters["robust_definition"] = (
                f"training-fold per-band {float(percentiles[0]):g}th/{float(percentiles[1]):g}th percentile"
            )
        if method == "N4":
            parameters["area_definition"] = "sum across the 16 equally spaced bands"
    state = NormalizationState(method=method, parameters=parameters)
    _validate_state(state)
    return state


def apply_normalization(values, state: NormalizationState):
    _validate_state(state)
    source = _validate_array(values).astype(np.float64, copy=True)
    parameters = state.parameters
    epsilon = float(parameters["epsilon"])
    if state.method == "N1":
        result = (source - np.asarray(parameters["mean"])) / np.asarray(parameters["scale"])
    else:
        low = np.asarray(parameters["clip_low"])
        high = np.asarray(parameters["clip_high"])
        clipped = np.clip(source, low, high)
        if state.method in ("N0", "N2"):
            result = (clipped - low) / (high - low)
        elif state.method == "N3":
            vector_mean = clipped.mean(axis=-1, keepdims=True)
            vector_scale = clipped.std(axis=-1, keepdims=True)
            result = (clipped - vector_mean) / np.maximum(vector_scale, epsilon)
        else:
            area = clipped.sum(axis=-1, keepdims=True)
            safe = np.abs(area) > epsilon
            result = np.divide(clipped, area, out=np.zeros_like(clipped), where=safe)
    if not np.isfinite(result).all():
        raise FloatingPointError("Normalization produced non-finite values")
    return result.astype(np.float32)


def save_normalization_state(state: NormalizationState, path):
    _validate_state(state)
    Path(path).write_text(json.dumps(state.to_dict(), indent=2, allow_nan=False) + "\n")


def load_normalization_state(path):
    return NormalizationState.from_dict(json.loads(Path(path).read_text()))


def _validate_state(state):
    if state.method not in METHODS:
        raise ValueError(f"Unknown normalization method: {state.method}")
    if state.fit_scope != "training_fold" or state.n_bands != N_BANDS:
        raise ValueError("Normalization state must be fitted on a 16-band training fold")
    parameters = state.parameters
    if not np.isfinite(float(parameters.get("epsilon", np.nan))) or float(parameters["epsilon"]) <= 0:
        raise ValueError("Normalization epsilon must be finite and positive")
    required = ("mean", "scale") if state.method == "N1" else ("clip_low", "clip_high")
    for key in required:
        values = np.asarray(parameters.get(key), dtype=np.float64)
        if values.shape != (N_BANDS,) or not np.isfinite(values).all():
            raise ValueError(f"Normalization parameter {key} must contain 16 finite values")
    if state.method == "N1" and np.any(np.asarray(parameters["scale"], dtype=np.float64) <= 0):
        raise ValueError("Normalization scales must be positive")
    if state.method != "N1" and np.any(
        np.asarray(parameters["clip_high"], dtype=np.float64)
        <= np.asarray(parameters["clip_low"], dtype=np.float64)
    ):
        raise ValueError("Normalization clip ranges must be positive")
