import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


N_BANDS = 16


@dataclass(frozen=True)
class MSCState:
    reference_spectrum: list
    fit_scope: str = "training_fold"
    n_bands: int = N_BANDS
    reference_method: str = "mean of training-fold spectral vectors"
    version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        state = cls(**value)
        _reference(state)
        return state


def savitzky_golay(values, window_length=5, polyorder=2):
    array = _validate_values(values)
    if not isinstance(window_length, int) or isinstance(window_length, bool):
        raise TypeError("window_length must be an integer")
    if window_length < 3 or window_length > 15 or window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer from 3 to 15")
    if not isinstance(polyorder, int) or isinstance(polyorder, bool) or not 0 <= polyorder < window_length:
        raise ValueError("polyorder must be a non-negative integer smaller than window_length")
    result = savgol_filter(
        array.astype(np.float64, copy=True),
        window_length=window_length,
        polyorder=polyorder,
        axis=-1,
        mode="interp",
    )
    return result.astype(np.float32)


def fit_msc(training_data):
    if isinstance(training_data, np.ndarray):
        arrays = [training_data]
    else:
        arrays = list(training_data)
    if not arrays:
        raise ValueError("training_data must not be empty")
    matrix = np.concatenate([_validate_values(array).reshape(-1, N_BANDS) for array in arrays], axis=0)
    reference = matrix.astype(np.float64, copy=False).mean(axis=0)
    if np.var(reference) <= 0:
        raise ValueError("MSC reference spectrum must vary across bands")
    return MSCState(reference_spectrum=reference.tolist())


def apply_msc(values, state: MSCState, epsilon=1e-8):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    array = _validate_values(values).astype(np.float64, copy=True)
    reference = _reference(state)
    centered_reference = reference - reference.mean()
    denominator = float(centered_reference @ centered_reference)
    flat = array.reshape(-1, N_BANDS)
    slopes = ((flat - flat.mean(axis=1, keepdims=True)) @ centered_reference) / denominator
    intercepts = flat.mean(axis=1) - slopes * reference.mean()
    stable = np.abs(slopes) > epsilon
    corrected = flat.copy()
    corrected[stable] = (flat[stable] - intercepts[stable, None]) / slopes[stable, None]
    corrected = corrected.reshape(array.shape)
    if not np.isfinite(corrected).all():
        raise FloatingPointError("MSC produced non-finite values")
    return corrected.astype(np.float32)


def save_msc_state(state: MSCState, path):
    _reference(state)
    Path(path).write_text(json.dumps(state.to_dict(), indent=2, allow_nan=False) + "\n")


def load_msc_state(path):
    return MSCState.from_dict(json.loads(Path(path).read_text()))


def _validate_values(values):
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[-1] != N_BANDS:
        raise ValueError("Spectra must have 16 bands on their last axis")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Spectra must contain only finite numeric values")
    return array


def _reference(state):
    if state.fit_scope != "training_fold" or state.n_bands != N_BANDS:
        raise ValueError("MSC state must be fitted on a 16-band training fold")
    reference = np.asarray(state.reference_spectrum, dtype=np.float64)
    if reference.shape != (N_BANDS,) or not np.isfinite(reference).all() or np.var(reference) <= 0:
        raise ValueError("MSC reference must contain 16 finite, non-constant values")
    return reference
