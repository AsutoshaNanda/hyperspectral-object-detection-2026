import numpy as np


def _strides(values, name):
    values = tuple(values)
    if not values or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        raise ValueError(f"{name} must contain positive integer strides")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate strides")
    return values


def validate_high_resolution_feature_level(
    architecture_name,
    *,
    input_bands,
    base_feature_strides,
    candidate_feature_strides,
    supported_feature_strides,
    requested_stride=4,
):
    if not isinstance(architecture_name, str) or not architecture_name.strip():
        raise ValueError("architecture_name must be non-empty")
    if input_bands != 16:
        raise ValueError("P5 requires an architecture path that preserves all 16 input bands")
    if not isinstance(requested_stride, int) or isinstance(requested_stride, bool) or requested_stride < 1:
        raise ValueError("requested_stride must be a positive integer")
    base = _strides(base_feature_strides, "base_feature_strides")
    candidate = _strides(candidate_feature_strides, "candidate_feature_strides")
    supported = _strides(supported_feature_strides, "supported_feature_strides")
    if requested_stride in base:
        raise ValueError("the requested high-resolution level already exists in the baseline")
    if requested_stride not in candidate:
        raise ValueError("the P5 candidate must add the requested high-resolution feature level")
    if not set(base).issubset(candidate):
        raise ValueError("the P5 candidate must retain every baseline feature level")
    if not set(candidate).issubset(supported):
        raise ValueError("the architecture does not explicitly support every requested feature level")
    return {
        "architecture": architecture_name,
        "input_bands": input_bands,
        "base_feature_strides": list(base),
        "candidate_feature_strides": list(candidate),
        "added_feature_strides": sorted(set(candidate) - set(base)),
        "requested_stride": requested_stride,
        "compatible": True,
    }
