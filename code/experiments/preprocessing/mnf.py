import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


N_BANDS = 16


@dataclass(frozen=True)
class MNFState:
    mean: list
    components: list
    inverse_components: list
    eigenvalues: list
    signal_covariance: list
    noise_covariance: list
    noise_estimation: str
    fit_scope: str = "training_fold"
    n_bands: int = N_BANDS
    version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        state = cls(**value)
        _state_arrays(state)
        return state


def _as_cubes(training_data):
    if isinstance(training_data, np.ndarray):
        if training_data.ndim in (2, 3):
            cubes = [training_data]
        elif training_data.ndim == 4:
            cubes = list(training_data)
        else:
            raise ValueError("training_data must contain spectral matrices or cubes")
    else:
        cubes = list(training_data)
    if not cubes:
        raise ValueError("training_data must not be empty")
    validated = []
    for cube in cubes:
        array = np.asarray(cube, dtype=np.float64)
        if array.ndim not in (2, 3) or array.shape[-1] != N_BANDS:
            raise ValueError("Every training item must have 16 bands on its last axis")
        if not np.isfinite(array).all():
            raise ValueError("training_data must contain only finite values")
        validated.append(array)
    return validated


def _noise_differences(cubes):
    differences = []
    for cube in cubes:
        axes = (0,) if cube.ndim == 2 else (0, 1)
        for axis in axes:
            if cube.shape[axis] > 1:
                differences.append(np.diff(cube, axis=axis).reshape(-1, N_BANDS))
    if not differences:
        raise ValueError("MNF noise estimation requires adjacent training samples")
    return np.concatenate(differences, axis=0)


def fit_mnf(training_data, regularization=1e-6):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    cubes = _as_cubes(training_data)
    matrix = np.concatenate([cube.reshape(-1, N_BANDS) for cube in cubes], axis=0)
    if matrix.shape[0] <= N_BANDS:
        raise ValueError("MNF fitting requires more training vectors than spectral bands")
    differences = _noise_differences(cubes)
    mean = matrix.mean(axis=0)
    signal_covariance = np.cov(matrix, rowvar=False)
    noise_covariance = np.cov(differences, rowvar=False) / 2.0
    if not np.isfinite(signal_covariance).all() or not np.isfinite(noise_covariance).all():
        raise ValueError("MNF covariance estimation requires enough varying training samples")
    floor = max(float(np.trace(noise_covariance)) / N_BANDS * regularization, regularization)
    noise_covariance = noise_covariance + np.eye(N_BANDS) * floor
    noise_values, noise_vectors = np.linalg.eigh(noise_covariance)
    noise_values = np.maximum(noise_values, floor)
    whitening = noise_vectors @ np.diag(1.0 / np.sqrt(noise_values))
    whitened_signal = whitening.T @ signal_covariance @ whitening
    eigenvalues, rotations = np.linalg.eigh(whitened_signal)
    order = np.argsort(eigenvalues)[::-1]
    components = whitening @ rotations[:, order]
    inverse_components = np.linalg.inv(components)
    state = MNFState(
        mean=mean.tolist(),
        components=components.tolist(),
        inverse_components=inverse_components.tolist(),
        eigenvalues=eigenvalues[order].tolist(),
        signal_covariance=signal_covariance.tolist(),
        noise_covariance=noise_covariance.tolist(),
        noise_estimation="half covariance of horizontal and vertical adjacent training-fold differences",
    )
    _state_arrays(state)
    return state


def transform_mnf(values, state: MNFState):
    mean, components, _, _ = _state_arrays(state)
    array = _validate_values(values)
    transformed = (array.astype(np.float64, copy=False) - mean) @ components
    return transformed.astype(np.float32)


def reconstruct_mnf(values, state: MNFState, n_components):
    if not isinstance(n_components, int) or isinstance(n_components, bool) or not 1 <= n_components <= N_BANDS:
        raise ValueError("n_components must be an integer from 1 to 16 selected on validation")
    mean, components, inverse_components, _ = _state_arrays(state)
    array = _validate_values(values)
    scores = (array.astype(np.float64, copy=False) - mean) @ components
    scores[..., n_components:] = 0.0
    reconstructed = scores @ inverse_components + mean
    if not np.isfinite(reconstructed).all():
        raise FloatingPointError("MNF reconstruction produced non-finite values")
    return reconstructed.astype(np.float32)


def reconstruction_error(original, reconstructed):
    original_array = _validate_values(original).astype(np.float64, copy=False)
    reconstructed_array = _validate_values(reconstructed).astype(np.float64, copy=False)
    if original_array.shape != reconstructed_array.shape:
        raise ValueError("original and reconstructed spectra must have the same shape")
    difference = reconstructed_array - original_array
    return {
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "mae": float(np.mean(np.abs(difference))),
    }


def save_mnf_state(state: MNFState, path):
    _state_arrays(state)
    Path(path).write_text(json.dumps(state.to_dict(), indent=2, allow_nan=False) + "\n")


def load_mnf_state(path):
    return MNFState.from_dict(json.loads(Path(path).read_text()))


def _validate_values(values):
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[-1] != N_BANDS:
        raise ValueError("Spectra must have 16 bands on their last axis")
    if not np.isfinite(array).all():
        raise ValueError("Spectra must contain only finite values")
    return array


def _state_arrays(state):
    if state.fit_scope != "training_fold" or state.n_bands != N_BANDS:
        raise ValueError("MNF state must be fitted on a 16-band training fold")
    mean = np.asarray(state.mean, dtype=np.float64)
    components = np.asarray(state.components, dtype=np.float64)
    inverse = np.asarray(state.inverse_components, dtype=np.float64)
    eigenvalues = np.asarray(state.eigenvalues, dtype=np.float64)
    signal_covariance = np.asarray(state.signal_covariance, dtype=np.float64)
    noise_covariance = np.asarray(state.noise_covariance, dtype=np.float64)
    if mean.shape != (N_BANDS,) or components.shape != (N_BANDS, N_BANDS):
        raise ValueError("Invalid MNF state dimensions")
    if inverse.shape != components.shape or eigenvalues.shape != (N_BANDS,):
        raise ValueError("Invalid MNF inverse or eigenvalue dimensions")
    if signal_covariance.shape != components.shape or noise_covariance.shape != components.shape:
        raise ValueError("Invalid MNF covariance dimensions")
    if not all(
        np.isfinite(value).all()
        for value in (mean, components, inverse, eigenvalues, signal_covariance, noise_covariance)
    ):
        raise ValueError("MNF state must contain only finite values")
    return mean, components, inverse, eigenvalues
