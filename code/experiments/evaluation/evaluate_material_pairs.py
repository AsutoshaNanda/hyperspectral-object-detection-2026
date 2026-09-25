import argparse
import json
from pathlib import Path

import numpy as np


N_BANDS = 16
DEFAULT_PAIRS = (("Apple", "Apple Plastic"), ("Egg", "Egg Plastic"))


def spectral_angle(spectra, prototypes, epsilon=1e-12):
    spectra_array = _spectra(spectra, "spectra")
    prototype_array = _spectra(prototypes, "prototypes")
    dots = spectra_array @ prototype_array.T
    spectrum_norms = np.linalg.norm(spectra_array, axis=1, keepdims=True)
    prototype_norms = np.linalg.norm(prototype_array, axis=1, keepdims=True).T
    cosine = dots / np.maximum(spectrum_norms * prototype_norms, epsilon)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def extract_box_spectrum(cube, bbox_xywh, shrink_fraction=0.1, estimator="median", trim_fraction=0.1):
    array = np.asarray(cube)
    if array.ndim != 3 or array.shape[-1] != N_BANDS or not np.isfinite(array).all():
        raise ValueError("cube must be a finite H x W x 16 array")
    if not 0 <= shrink_fraction < 0.5:
        raise ValueError("shrink_fraction must be in [0, 0.5)")
    x, y, width, height = np.asarray(bbox_xywh, dtype=np.float64)
    if not np.isfinite([x, y, width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("bbox_xywh must be finite with positive width and height")
    x0 = max(0, int(np.floor(x + shrink_fraction * width)))
    y0 = max(0, int(np.floor(y + shrink_fraction * height)))
    x1 = min(array.shape[1], int(np.ceil(x + width - shrink_fraction * width)))
    y1 = min(array.shape[0], int(np.ceil(y + height - shrink_fraction * height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Shrunk box contains no pixels")
    pixels = array[y0:y1, x0:x1].reshape(-1, N_BANDS).astype(np.float64, copy=False)
    if estimator == "median":
        return np.median(pixels, axis=0).astype(np.float32)
    if estimator != "trimmed_mean" or not 0 <= trim_fraction < 0.5:
        raise ValueError("estimator must be median or trimmed_mean with trim_fraction in [0, 0.5)")
    cut = int(np.floor(pixels.shape[0] * trim_fraction))
    ordered = np.sort(pixels, axis=0)
    selected = ordered[cut:pixels.shape[0] - cut] if cut else ordered
    return selected.mean(axis=0).astype(np.float32)


def extract_annotation_spectra(
    cubes_by_image,
    annotations,
    shrink_fraction=0.1,
    estimator="median",
    trim_fraction=0.1,
):
    rows = []
    labels = []
    annotation_ids = []
    for index, annotation in enumerate(annotations):
        image_id = annotation["image_id"]
        cube = cubes_by_image.get(image_id, cubes_by_image.get(str(image_id)))
        if cube is None:
            raise KeyError(f"Missing cube for image {image_id}")
        rows.append(
            extract_box_spectrum(
                cube,
                annotation["bbox"],
                shrink_fraction=shrink_fraction,
                estimator=estimator,
                trim_fraction=trim_fraction,
            )
        )
        labels.append(int(annotation["category_id"]))
        annotation_ids.append(annotation.get("id", index))
    if not rows:
        raise ValueError("annotations must not be empty")
    return np.stack(rows), np.asarray(labels, dtype=np.int64), annotation_ids


def fit_prototypes(training_spectra, training_labels, estimator="median"):
    spectra = _spectra(training_spectra, "training_spectra")
    labels = _labels(training_labels, spectra.shape[0], "training_labels")
    class_ids = np.unique(labels)
    if class_ids.size < 2:
        raise ValueError("At least two training classes are required")
    if estimator == "median":
        prototypes = np.stack([np.median(spectra[labels == class_id], axis=0) for class_id in class_ids])
    elif estimator == "mean":
        prototypes = np.stack([spectra[labels == class_id].mean(axis=0) for class_id in class_ids])
    else:
        raise ValueError("prototype estimator must be median or mean")
    return class_ids, prototypes.astype(np.float64)


def evaluate_material_pairs(
    training_spectra,
    training_labels,
    evaluation_spectra,
    evaluation_labels,
    class_names,
    pairs=DEFAULT_PAIRS,
    prototype_estimator="median",
):
    train_x = _spectra(training_spectra, "training_spectra")
    train_y = _labels(training_labels, train_x.shape[0], "training_labels")
    eval_x = _spectra(evaluation_spectra, "evaluation_spectra")
    eval_y = _labels(evaluation_labels, eval_x.shape[0], "evaluation_labels")
    class_ids, prototypes = fit_prototypes(train_x, train_y, prototype_estimator)
    name_to_id = {_class_key(name): int(class_id) for class_id, name in _class_items(class_names)}
    prototype_index = {int(class_id): index for index, class_id in enumerate(class_ids)}
    train_angles = spectral_angle(train_x, prototypes)
    eval_angles = spectral_angle(eval_x, prototypes)
    reports = []
    for first_name, second_name in pairs:
        first_id = _resolve_class(first_name, name_to_id)
        second_id = _resolve_class(second_name, name_to_id)
        if first_id not in prototype_index or second_id not in prototype_index:
            raise ValueError(f"Both classes in {first_name}/{second_name} need training samples")
        train_mask = np.isin(train_y, (first_id, second_id))
        eval_mask = np.isin(eval_y, (first_id, second_id))
        if np.count_nonzero(train_y[train_mask] == first_id) == 0 or np.count_nonzero(train_y[train_mask] == second_id) == 0:
            raise ValueError(f"Both classes in {first_name}/{second_name} need training samples")
        if np.count_nonzero(eval_y[eval_mask] == first_id) == 0 or np.count_nonzero(eval_y[eval_mask] == second_id) == 0:
            raise ValueError(f"Both classes in {first_name}/{second_name} need evaluation samples")
        first_index = prototype_index[first_id]
        second_index = prototype_index[second_id]
        train_margin = train_angles[train_mask, second_index] - train_angles[train_mask, first_index]
        eval_margin = eval_angles[eval_mask, second_index] - eval_angles[eval_mask, first_index]
        threshold, train_accuracy = _select_threshold(train_margin, train_y[train_mask] == first_id)
        eval_positive = eval_y[eval_mask] == first_id
        eval_prediction = eval_margin >= threshold
        same_angles = np.where(
            eval_positive,
            eval_angles[eval_mask, first_index],
            eval_angles[eval_mask, second_index],
        )
        cross_angles = np.where(
            eval_positive,
            eval_angles[eval_mask, second_index],
            eval_angles[eval_mask, first_index],
        )
        reports.append(
            {
                "classes": [str(first_name), str(second_name)],
                "class_ids": [first_id, second_id],
                "threshold": float(threshold),
                "threshold_fit_scope": "training_fold",
                "training_accuracy_at_selected_threshold": float(train_accuracy),
                "evaluation_accuracy": float(np.mean(eval_prediction == eval_positive)),
                "evaluation_auc": float(_binary_auc(eval_positive, eval_margin)),
                "evaluation_margin_overlap": float(_distribution_overlap(eval_margin[eval_positive], eval_margin[~eval_positive])),
                "same_class_angle_radians": _summary(same_angles),
                "cross_class_angle_radians": _summary(cross_angles),
                "training_count": int(train_mask.sum()),
                "evaluation_count": int(eval_mask.sum()),
            }
        )
    return {
        "fit_scope": "training_fold",
        "evaluation_scope": "validation_fold",
        "n_bands": N_BANDS,
        "prototype_estimator": prototype_estimator,
        "prototype_class_ids": class_ids.astype(int).tolist(),
        "prototypes": prototypes.tolist(),
        "pairs": reports,
    }


def _select_threshold(scores, positive):
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    unique = np.unique(scores)
    if unique.size == 1:
        candidates = unique
    else:
        candidates = np.concatenate(([np.nextafter(unique[0], -np.inf)], (unique[:-1] + unique[1:]) / 2, [np.nextafter(unique[-1], np.inf)]))
    accuracies = np.asarray([np.mean((scores >= threshold) == positive) for threshold in candidates])
    best = np.flatnonzero(accuracies == accuracies.max())
    chosen = best[np.argmin(np.abs(candidates[best]))]
    return float(candidates[chosen]), float(accuracies[chosen])


def _binary_auc(positive, scores):
    positive = np.asarray(positive, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive_scores = scores[positive]
    negative_scores = scores[~positive]
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise ValueError("AUC requires both classes")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size


def _distribution_overlap(first, second, bins=20):
    pooled = np.concatenate((first, second))
    if pooled.min() == pooled.max():
        return 1.0
    edges = np.linspace(pooled.min(), pooled.max(), bins + 1)
    first_hist = np.histogram(first, bins=edges)[0] / len(first)
    second_hist = np.histogram(second, bins=edges)[0] / len(second)
    return np.minimum(first_hist, second_hist).sum()


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _spectra(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != N_BANDS or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite N x 16 array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _labels(values, expected, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != expected:
        raise ValueError(f"{name} must have one entry per spectrum")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer class IDs")
    return array.astype(np.int64)


def _class_items(class_names):
    if isinstance(class_names, dict):
        return class_names.items()
    return enumerate(class_names)


def _resolve_class(value, name_to_id):
    if isinstance(value, (int, np.integer)):
        return int(value)
    key = _class_key(value)
    if key not in name_to_id:
        raise ValueError(f"Unknown class name: {value}")
    return name_to_id[key]


def _class_key(value):
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--class-names", required=True)
    parser.add_argument("--pairs", nargs="*", default=["Apple:Apple Plastic", "Egg:Egg Plastic"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training = np.load(args.training)
    evaluation = np.load(args.evaluation)
    class_names = json.loads(Path(args.class_names).read_text())
    pairs = [tuple(value.split(":", 1)) for value in args.pairs]
    report = evaluate_material_pairs(
        training["spectra"],
        training["labels"],
        evaluation["spectra"],
        evaluation["labels"],
        class_names,
        pairs,
    )
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
