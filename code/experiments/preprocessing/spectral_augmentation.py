import hashlib
from dataclasses import asdict, dataclass

import numpy as np


N_BANDS = 16


@dataclass(frozen=True)
class SpectralAugmentationConfig:
    copies: int = 1
    gain_limit: float = 0.05
    tilt_limit: float = 0.03
    noise_std: float = 0.01

    def __post_init__(self):
        if not isinstance(self.copies, int) or isinstance(self.copies, bool) or self.copies < 0:
            raise ValueError("copies must be a non-negative integer")
        for name in ("gain_limit", "tilt_limit", "noise_std"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.gain_limit >= 1 or self.tilt_limit >= 1:
            raise ValueError("gain_limit and tilt_limit must stay below one")

    def to_dict(self):
        return asdict(self)


def stable_rng(seed, sample_id, copy_index):
    digest = hashlib.sha256(f"{seed}:{sample_id}:{copy_index}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def spectral_augment(values, rng, gain_limit=0.05, tilt_limit=0.03, noise_std=0.01):
    array = _validate_values(values).astype(np.float32, copy=True)
    limits = np.asarray([gain_limit, tilt_limit, noise_std], dtype=np.float64)
    if not np.isfinite(limits).all() or (limits < 0).any():
        raise ValueError("Augmentation limits must be finite and non-negative")
    if gain_limit >= 1 or tilt_limit >= 1:
        raise ValueError("gain_limit and tilt_limit must stay below one")
    gain = rng.uniform(1.0 - gain_limit, 1.0 + gain_limit)
    tilt = rng.uniform(-tilt_limit, tilt_limit)
    wavelength_axis = np.linspace(-1.0, 1.0, N_BANDS, dtype=np.float32)
    array *= gain * (1.0 + tilt * wavelength_axis)
    if noise_std > 0:
        spatial_axes = tuple(range(array.ndim - 1))
        scale = np.maximum(array.std(axis=spatial_axes), 1e-6)
        noise = rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        array += noise * scale
    if not np.isfinite(array).all():
        raise FloatingPointError("Spectral augmentation produced non-finite values")
    return array


def make_training_copies(
    values,
    seed,
    sample_id,
    config=SpectralAugmentationConfig(),
    class_ids=(),
    target_classes=None,
    split="train",
):
    array = _validate_values(values)
    if split != "train":
        raise ValueError("Spectral augmentation copies are allowed only for the training split")
    if target_classes is not None and not set(class_ids).intersection(target_classes):
        return []
    return [
        spectral_augment(
            array,
            stable_rng(seed, sample_id, copy_index),
            config.gain_limit,
            config.tilt_limit,
            config.noise_std,
        )
        for copy_index in range(config.copies)
    ]


def _validate_values(values):
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[-1] != N_BANDS:
        raise ValueError("Spectra must have 16 bands on their last axis")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Spectra must contain only finite numeric values")
    return array
