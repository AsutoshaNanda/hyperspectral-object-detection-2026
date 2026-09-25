"""Reproducible 16-band hyperspectral detector experiments and submission builder.

Simple explanation:

This script:
1. Reads the 16-band hyperspectral images.
2. Reads the correct object boxes from XML files.
3. Keeps some training images hidden for validation and holdout testing.
4. Converts the data into the format YOLO expects.
5. Trains one YOLO26m or RT-DETR-L object-detection model.
6. Uses that model on the competition test images.
7. Creates submission.csv only when holdout mAP50-95 is at least 0.50.

Important terms:
- YOLO26m: an object detector. It predicts class, confidence, and box.
- 16 bands: the 16 spectral channels in each competition image.
- Validation: hidden training images used to check model progress.
- Holdout: another hidden local set used as a safer final check.
- Confidence: how sure the model is about one prediction.
- Bounding box: rectangle around an object.

This version can read the competition raw PNG mosaics directly and applies the organizer-provided X2Cube conversion internally. It supports controlled aspect-ratio, cross-validation, spectral-augmentation, and same-checkpoint TTA experiments.
"""

from pathlib import Path
import argparse
import json
import random
import hashlib
import warnings
import importlib.metadata
import xml.etree.ElementTree as ET

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold, train_test_split



def X2Cube(img, cellSize=4):
    if img.ndim != 2 or any(n % cellSize for n in img.shape):
        raise ValueError("Raw image sides must be divisible by 4")
    B = [cellSize, cellSize]
    skip = [cellSize, cellSize]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(
        img,
        start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel(),
    )
    out = np.transpose(out)
    return out.reshape(M // cellSize, N // cellSize, cellSize * cellSize)


def read_classes(path):
    names = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(names) != 18:
        raise ValueError(f"Expected 18 classes, found {len(names)}")
    return names


def read_cube(path):
    path = Path(path)
    if path.suffix.lower() == ".png":
        raw = np.array(Image.open(path))
        if raw.ndim != 2:
            raise ValueError(f"{path} must be a 2D raw mosaic PNG, got {raw.shape}")
        x = X2Cube(raw)
    elif path.suffix.lower() == ".npy":
        x = np.load(path)
    elif path.suffix.lower() == ".npz":
        z = np.load(path)
        x = z[z.files[0]]
    elif path.suffix.lower() in {".tif", ".tiff"}:
        ok, frames = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
        if not ok:
            raise ValueError(f"Cannot read {path}")
        x = np.stack(frames, axis=-1)
    else:
        raise ValueError(f"Unsupported cube format: {path.suffix}")
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite image values in {path}")
    if x.ndim != 3:
        raise ValueError(f"{path} must be 3D, got {x.shape}")
    if x.shape[-1] == 16:
        return x.astype(np.float32)
    if x.shape[0] == 16:
        return np.moveaxis(x, 0, -1).astype(np.float32)
    raise ValueError(f"{path} does not contain 16 bands: {x.shape}")


def collect_cubes(root):
    root = Path(root)
    files = []
    for ext in ("*.png", "*.npy", "*.npz", "*.tif", "*.tiff"):
        files.extend(root.rglob(ext))
    files = sorted({p.resolve() for p in files})
    if not files:
        raise FileNotFoundError(
            "No 16-band cubes found. Run the organizer-compatible X2Cube conversion first."
        )
    return files


def band_limits(files, samples_per_image=256, seed=42):
    rng = np.random.default_rng(seed)
    chunks = []
    for p in files:
        x = read_cube(p).reshape(-1, 16)
        n = min(samples_per_image, len(x))
        idx = rng.choice(len(x), n, replace=False)
        chunks.append(x[idx])
    sample = np.concatenate(chunks, axis=0)
    lo = np.percentile(sample, 0.5, axis=0)
    hi = np.percentile(sample, 99.5, axis=0)
    hi = np.maximum(hi, lo + 1e-6)
    return lo, hi


def model_geometry(height, width, input_policy):
    if input_policy == "native":
        return {
            "original_height": height,
            "original_width": width,
            "model_height": height,
            "model_width": width,
            "pad_top": 0,
            "pad_left": 0,
        }
    if input_policy != "square_pad":
        raise ValueError(f"Unknown input policy: {input_policy}")
    side = max(height, width)
    return {
        "original_height": height,
        "original_width": width,
        "model_height": side,
        "model_width": side,
        "pad_top": (side - height) // 2,
        "pad_left": (side - width) // 2,
    }


def resolve_input_policy(family, input_policy):
    if input_policy != "auto":
        return input_policy
    return "square_pad" if family == "rtdetr" else "native"


def prepare_cube(cube, lo, hi, input_policy, pipeline_state=None):
    if pipeline_state is None:
        x = np.clip((cube - lo) / (hi - lo), 0, 1)
        x = np.rint(x * 255).astype(np.uint8)
    else:
        try:
            from Experiments.preprocessing.pipeline import apply_pipeline
        except ModuleNotFoundError:
            from pipeline import apply_pipeline
        x = apply_pipeline(cube, pipeline_state, output="uint8")
    height, width = x.shape[:2]
    geometry = model_geometry(height, width, input_policy)
    if input_policy == "square_pad":
        padded = np.zeros(
            (geometry["model_height"], geometry["model_width"], 16),
            dtype=np.uint8,
        )
        top = geometry["pad_top"]
        left = geometry["pad_left"]
        padded[top : top + height, left : left + width] = x
        x = padded
    return x, geometry


def write_tiff(cube, path, lo, hi, input_policy, pipeline_state=None):
    x, geometry = prepare_cube(cube, lo, hi, input_policy, pipeline_state)
    chw = np.moveaxis(x, -1, 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwritemulti(str(path), chw)
    if not ok:
        raise IOError(f"Failed to write {path}")
    return geometry


def transform_rows(rows, geometry):
    source_width = geometry["original_width"]
    source_height = geometry["original_height"]
    target_width = geometry["model_width"]
    target_height = geometry["model_height"]
    pad_left = geometry["pad_left"]
    pad_top = geometry["pad_top"]
    transformed = []
    for cls, cx, cy, bw, bh in rows:
        transformed.append(
            (
                cls,
                (cx * source_width + pad_left) / target_width,
                (cy * source_height + pad_top) / target_height,
                bw * source_width / target_width,
                bh * source_height / target_height,
            )
        )
    return transformed


def restore_boxes(boxes, geometry):
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    boxes[:, [0, 2]] -= geometry["pad_left"]
    boxes[:, [1, 3]] -= geometry["pad_top"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, geometry["original_width"])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, geometry["original_height"])
    return boxes


def spectral_augment(cube, rng, gain_limit, tilt_limit, noise_std):
    cube = cube.astype(np.float32, copy=True)
    gain = rng.uniform(1.0 - gain_limit, 1.0 + gain_limit)
    tilt = rng.uniform(-tilt_limit, tilt_limit)
    wavelength_axis = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    cube *= gain * (1.0 + tilt * wavelength_axis[None, None, :])
    if noise_std > 0:
        scale = np.maximum(cube.std(axis=(0, 1)), 1e-6)
        noise = rng.normal(0.0, noise_std, size=cube.shape).astype(np.float32)
        cube += noise * scale[None, None, :]
    return cube


def stable_rng(seed, stem, copy_index):
    digest = hashlib.sha256(f"{seed}:{stem}:{copy_index}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def expand_rgb_stem(target_model, source_model):
    import torch

    target_layers = [layer for layer in target_model.modules() if isinstance(layer, torch.nn.Conv2d)]
    source_layers = [layer for layer in source_model.modules() if isinstance(layer, torch.nn.Conv2d)]
    target = next((layer for layer in target_layers if layer.in_channels == 16), None)
    source = next(
        (
            layer
            for layer in source_layers
            if layer.in_channels == 3
            and target is not None
            and layer.out_channels == target.out_channels
            and layer.kernel_size == target.kernel_size
        ),
        None,
    )
    if target is None or source is None:
        raise RuntimeError("Could not identify compatible 3-channel and 16-channel input stems")
    expanded = source.weight.detach().mean(dim=1, keepdim=True).repeat(1, 16, 1, 1)
    expanded *= 3.0 / 16.0
    target.weight.data.copy_(expanded.to(device=target.weight.device, dtype=target.weight.dtype))
    if target.bias is not None and source.bias is not None:
        target.bias.data.copy_(source.bias.detach().to(device=target.bias.device, dtype=target.bias.dtype))
    print("Initialized the 16-channel input stem from RGB weights with activation-scale correction")


def trainer_class(args):
    if args.stem_init == "random":
        return None
    if args.family == "rtdetr":
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        class HSITrainer(RTDETRTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
                if weights is None:
                    raise RuntimeError("rgb_mean stem initialization requires pretrained weights")
                expand_rgb_stem(model, weights)
                return model

        return HSITrainer
    from ultralytics.models.yolo.detect import DetectionTrainer

    class HSITrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if weights is None:
                raise RuntimeError("rgb_mean stem initialization requires pretrained weights")
            expand_rgb_stem(model, weights)
            return model

    return HSITrainer


def parse_voc(xml_path, class_to_id):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    rows = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip()
        if name not in class_to_id:
            raise ValueError(f"Unknown class {name} in {xml_path}")
        b = obj.find("bndbox")
        x1 = float(b.findtext("xmin"))
        y1 = float(b.findtext("ymin"))
        x2 = float(b.findtext("xmax"))
        y2 = float(b.findtext("ymax"))
        x1 = min(max(x1, 0.0), width)
        x2 = min(max(x2, 0.0), width)
        y1 = min(max(y1, 0.0), height)
        y2 = min(max(y2, 0.0), height)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        rows.append((class_to_id[name], cx, cy, bw, bh))
    return width, height, rows


def annotation_map(xml_root):
    xmls = list(Path(xml_root).rglob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No VOC XML files found under {xml_root}")
    return unique_stems(xmls)


def rarest_image_label(stems, boxes_by_stem):
    counts = {}
    for stem in stems:
        for row in boxes_by_stem[stem]:
            cls = int(row[0])
            counts[cls] = counts.get(cls, 0) + 1
    labels = []
    for stem in stems:
        classes = {int(row[0]) for row in boxes_by_stem[stem]}
        if not classes:
            labels.append(-1)
        else:
            labels.append(min(classes, key=lambda c: counts[c]))
    return labels


def safe_split(stems, boxes_by_stem, seed):
    labels = rarest_image_label(stems, boxes_by_stem)
    try:
        train, temp = train_test_split(
            stems,
            test_size=0.20,
            random_state=seed,
            stratify=labels,
        )
        temp_labels = rarest_image_label(temp, boxes_by_stem)
        val, holdout = train_test_split(
            temp,
            test_size=0.50,
            random_state=seed,
            stratify=temp_labels,
        )
    except ValueError as exc:
        warnings.warn(f"Class-balanced split failed; using random split: {exc}")
        train, temp = train_test_split(
            stems,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
        )
        val, holdout = train_test_split(
            temp,
            test_size=0.50,
            random_state=seed,
            shuffle=True,
        )
    return sorted(train), sorted(val), sorted(holdout)


def read_groups(path, stems):
    if not path:
        return {stem: stem for stem in stems}, "image_id_only"
    frame = pd.read_csv(path, dtype=str)
    expected = {"image_id", "group_id"}
    if set(frame.columns) != expected:
        raise ValueError(f"Groups file columns must be {sorted(expected)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Groups file contains duplicate image_id values")
    groups = dict(zip(frame["image_id"], frame["group_id"]))
    missing = sorted(set(stems) - set(groups))
    if missing:
        raise ValueError(f"Groups file is missing {len(missing)} training IDs")
    return {stem: groups[stem] for stem in stems}, str(Path(path))


def cv_split(stems, boxes_by_stem, groups_by_stem, seed, folds, fold_index):
    if folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if not 0 <= fold_index < folds:
        raise ValueError(f"cv_fold_index must be in 0..{folds - 1}")
    labels = np.asarray(rarest_image_label(stems, boxes_by_stem))
    groups = np.asarray([groups_by_stem[stem] for stem in stems])
    holdout_splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    design_idx, holdout_idx = next(holdout_splitter.split(stems, labels, groups))
    design_stems = np.asarray(stems)[design_idx]
    design_labels = labels[design_idx]
    design_groups = groups[design_idx]
    fold_splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    selected = list(fold_splitter.split(design_stems, design_labels, design_groups))[fold_index]
    train_idx, val_idx = selected
    return (
        sorted(design_stems[train_idx].tolist()),
        sorted(design_stems[val_idx].tolist()),
        sorted(np.asarray(stems)[holdout_idx].tolist()),
    )


def load_split_manifest(path, stems):
    split_map = json.loads(Path(path).read_text())
    if set(split_map) != {"train", "val", "test"}:
        raise ValueError("Split manifest must contain train, val, and test")
    flattened = [str(stem) for split in split_map.values() for stem in split]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Split manifest contains duplicate IDs across splits")
    if set(flattened) != set(stems):
        raise ValueError("Split manifest IDs do not exactly match usable training IDs")
    return {key: sorted(map(str, value)) for key, value in split_map.items()}


def assert_group_isolation(split_map, groups_by_stem):
    group_splits = {}
    for split, stems in split_map.items():
        for stem in stems:
            group = groups_by_stem[stem]
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f"Group {group} leaks across {group_splits[group]} and {split}")
            group_splits[group] = split


def write_label(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        f"{int(c)} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"
        for c, cx, cy, w, h in rows
    )
    Path(path).write_text(text + ("\n" if text else ""))


def image_id_value(stem):
    return int(stem) if stem.isdigit() else stem


def parse_class_ids(value):
    if not value:
        return []
    ids = sorted({int(item) for item in value.split(",")})
    if any(class_id < 0 or class_id > 17 for class_id in ids):
        raise argparse.ArgumentTypeError("Class IDs must be between 0 and 17")
    return ids


def build_dataset(args):
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else work
    prepared = cache_root / "dataset"
    if prepared.exists():
        raise FileExistsError(f"Use a fresh workdir; keeping existing data: {prepared}")
    names = read_classes(args.class_file)
    class_to_id = {name: i for i, name in enumerate(names)}
    train_cubes = collect_cubes(args.train_cubes)
    design_only = getattr(args, "design_only", False)
    test_cubes = [] if design_only else collect_cubes(args.test_cubes)
    ranking_cubes = collect_cubes(args.ranking_cubes) if args.ranking_cubes and not design_only else []
    xmls = annotation_map(args.train_xml)
    cube_by_stem = unique_stems(train_cubes)
    inference_by_stem = unique_stems(test_cubes + ranking_cubes)
    if set(cube_by_stem) != set(xmls):
        raise ValueError("Training image and XML IDs do not match")
    stems = sorted(set(cube_by_stem) & set(xmls))
    bad_stems = {"1227", "1836", "1855"}
    stems = [s for s in stems if s not in bad_stems]
    print(f"Skipping mismatched images: {sorted(bad_stems)}")

    if not stems:
        raise RuntimeError("No matching training cube/XML stems found")

    boxes_by_stem = {}
    dims_by_stem = {}
    for stem in stems:
        w, h, rows = parse_voc(xmls[stem], class_to_id)
        boxes_by_stem[stem] = rows
        dims_by_stem[stem] = (w, h)

    groups_by_stem, groups_source = read_groups(args.groups_file, stems)
    if args.split_manifest:
        split_map = load_split_manifest(args.split_manifest, stems)
    elif args.cv_fold_index is not None:
        train_stems, val_stems, holdout_stems = cv_split(
            stems,
            boxes_by_stem,
            groups_by_stem,
            args.seed,
            args.cv_folds,
            args.cv_fold_index,
        )
        split_map = {"train": train_stems, "val": val_stems, "test": holdout_stems}
    else:
        train_stems, val_stems, holdout_stems = safe_split(stems, boxes_by_stem, args.seed)
        split_map = {"train": train_stems, "val": val_stems, "test": holdout_stems}
    assert_group_isolation(split_map, groups_by_stem)
    if getattr(args, "train_on_all", False):
        # Phase-2 final fit: train on every labelled image. val/test stay as monitoring
        # sets but are now also in train, so their scores are no longer unbiased.
        split_map["train"] = sorted(set(split_map["train"]) | set(split_map["val"]) | set(split_map["test"]))
    train_stems = split_map["train"]
    val_stems = split_map["val"]
    holdout_stems = split_map["test"]
    normalization = getattr(args, "normalization", "N0")
    spectral_preprocess = getattr(args, "spectral_preprocess", "SP0")
    pipeline_state = None
    if normalization == "N0" and spectral_preprocess == "SP0":
        lo, hi = band_limits([cube_by_stem[s] for s in train_stems], seed=args.seed)
    else:
        try:
            from Experiments.preprocessing.pipeline import fit_pipeline, save_pipeline_state
        except ModuleNotFoundError:
            from pipeline import fit_pipeline, save_pipeline_state
        pipeline_state = fit_pipeline(
            (read_cube(cube_by_stem[s]) for s in train_stems),
            normalization,
            spectral_method=spectral_preprocess,
            seed=args.seed,
            samples_per_cube=getattr(args, "normalization_samples_per_image", 256),
            mnf_components=getattr(args, "mnf_components", None),
            savgol_window=getattr(args, "savgol_window", 5),
            savgol_order=getattr(args, "savgol_order", 2),
        )
        lo = hi = None
        save_pipeline_state(pipeline_state, work / "spectral_pipeline_state.json")
    input_policy = resolve_input_policy(args.family, args.input_policy)
    (work / "split_manifest.json").write_text(json.dumps(split_map, indent=2))
    train_images_written = 0
    for split, split_stems in split_map.items():
        if design_only and split == "test":
            continue
        for stem in split_stems:
            cube = read_cube(cube_by_stem[stem])
            h, w = cube.shape[:2]
            ann_w, ann_h = dims_by_stem[stem]
            if (w, h) != (ann_w, ann_h):
                raise ValueError(
                    f"Spatial mismatch for {stem}: cube {(w, h)} vs annotation {(ann_w, ann_h)}"
                )
            geometry = write_tiff(
                cube,
                prepared / "images" / split / f"{stem}.tiff",
                lo,
                hi,
                input_policy,
                pipeline_state,
            )
            rows = transform_rows(boxes_by_stem[stem], geometry)
            write_label(prepared / "labels" / split / f"{stem}.txt", rows)
            if split != "train":
                continue
            train_images_written += 1
            classes = {int(row[0]) for row in boxes_by_stem[stem]}
            selected = not args.spectral_aug_classes or classes.intersection(args.spectral_aug_classes)
            if not selected:
                continue
            for copy_index in range(args.spectral_aug_copies):
                suffix = f"{stem}_hsaug{copy_index + 1}"
                augmented = spectral_augment(
                    cube,
                    stable_rng(args.seed, stem, copy_index),
                    args.spectral_gain,
                    args.spectral_tilt,
                    args.spectral_noise,
                )
                write_tiff(
                    augmented,
                    prepared / "images" / split / f"{suffix}.tiff",
                    lo,
                    hi,
                    input_policy,
                    pipeline_state,
                )
                write_label(prepared / "labels" / split / f"{suffix}.txt", rows)
                train_images_written += 1

    test_out = cache_root / "test_tiff"
    if test_out.exists():
        raise FileExistsError(f"Use a fresh workdir; keeping existing data: {test_out}")
    test_out.mkdir(parents=True, exist_ok=True)
    inference_geometry = {}
    for stem, path in inference_by_stem.items():
        inference_geometry[stem] = write_tiff(
            read_cube(path),
            test_out / f"{stem}.tiff",
            lo,
            hi,
            input_policy,
            pipeline_state,
        )
    (work / "inference_geometry.json").write_text(json.dumps(inference_geometry, indent=2))

    data = {
        "path": str(prepared),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "channels": 16,
        "names": {i: n for i, n in enumerate(names)},
    }
    yaml_path = work / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))

    report = {
        "train_images_total": len(stems),
        "train_split": len(train_stems),
        "val_split": len(val_stems),
        "holdout_split": len(holdout_stems),
        "prepared_train_images": train_images_written,
        "competition_test_images": len(test_cubes),
        "competition_ranking_images": len(ranking_cubes),
        "class_names": names,
        "excluded_ids": sorted(bad_stems),
        "split_ids": split_map,
        "split_mode": "manifest" if args.split_manifest else ("cv" if args.cv_fold_index is not None else "legacy"),
        "cv_folds": args.cv_folds if args.cv_fold_index is not None else None,
        "cv_fold_index": args.cv_fold_index,
        "groups_source": groups_source,
        "group_count": len(set(groups_by_stem.values())),
        "input_policy": input_policy,
        "spectral_augmentation": {
            "copies": args.spectral_aug_copies,
            "classes": args.spectral_aug_classes,
            "gain_limit": args.spectral_gain,
            "tilt_limit": args.spectral_tilt,
            "noise_std": args.spectral_noise,
        },
        "class_counts": {split: np.bincount([int(row[0]) for stem in ids for row in boxes_by_stem[stem]], minlength=18).tolist() for split, ids in split_map.items()},
        "band_low_0_5pct": lo.tolist() if lo is not None else None,
        "band_high_99_5pct": hi.tolist() if hi is not None else None,
        "normalization": normalization,
        "spectral_preprocess": spectral_preprocess,
        "spectral_pipeline_state": pipeline_state.to_dict() if pipeline_state is not None else None,
    }
    (work / "data_audit.json").write_text(json.dumps(report, indent=2))
    return yaml_path, test_out, inference_geometry


def train(args, yaml_path):
    model = model_class(args)(args.model)
    result = model.train(
        trainer=trainer_class(args),
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.workdir) / "runs"),
        name=args.run_name,
        seed=args.seed,
        deterministic=args.family == "yolo",
        amp=args.family == "yolo",
        rect=args.family == "yolo",
        multi_scale=args.multi_scale,
        patience=25,
        close_mosaic=15,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=5.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        augmentations=[],
        plots=False,
        save=True,
        verbose=True,
        pretrained=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    return best


def localization_finetune(args, yaml_path, checkpoint):
    if args.localization_epochs <= 0:
        return None
    model = model_class(args)(str(checkpoint))
    result = model.train(
        data=str(yaml_path),
        epochs=args.localization_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.workdir) / "runs"),
        name=f"{args.run_name}_localize",
        seed=args.seed,
        deterministic=args.family == "yolo",
        amp=args.family == "yolo",
        rect=args.family == "yolo",
        multi_scale=0.0,
        patience=args.localization_epochs,
        lr0=args.localization_lr0,
        optimizer="AdamW",
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=0.0,
        warmup_bias_lr=0.0,
        warmup_momentum=0.0,
        close_mosaic=0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.02,
        scale=0.10,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        augmentations=[],
        plots=False,
        save=True,
        verbose=True,
        pretrained=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    return best


def predict(args, best, test_out, inference_geometry, tta):
    model = model_class(args)(str(best))
    results = model.predict(
        source=str(test_out),
        imgsz=args.imgsz,
        conf=args.predict_conf,
        iou=args.iou,
        max_det=args.predict_max_det,
        device=args.device,
        augment=tta,
        stream=True,
        verbose=False,
    )
    rows = []
    seen = set()
    empty = []
    for r in results:
        seen.add(Path(r.path).stem)
        if r.boxes is None or len(r.boxes) == 0:
            empty.append(Path(r.path).stem)
        stem = Path(r.path).stem
        if r.boxes is None:
            continue
        xyxy = restore_boxes(
            r.boxes.xyxy.detach().cpu().numpy(),
            inference_geometry[stem],
        )
        confs = r.boxes.conf.detach().cpu().numpy()
        classes = r.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, cls in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append(
                {
                    "image_id": image_id_value(stem),
                    "class_id": int(cls),
                    "confidence": float(score),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    expected_ids = set(inference_geometry)
    if seen != expected_ids:
        raise RuntimeError("Some test images were not processed")
    counts = pd.Series([row["image_id"] for row in rows]).value_counts()
    (Path(args.workdir) / "prediction_audit.json").write_text(json.dumps({
        "processed_images": len(seen),
        "no_detection_ids": empty,
        "prediction_confidence": args.predict_conf,
        "prediction_max_det": args.predict_max_det,
        "tta": tta,
        "predictions_total": len(rows),
        "predictions_per_image_min": int(counts.min()) if len(counts) else 0,
        "predictions_per_image_median": float(counts.median()) if len(counts) else 0.0,
        "predictions_per_image_max": int(counts.max()) if len(counts) else 0,
    }, indent=2))
    sub = pd.DataFrame(rows)
    if sub.empty:
        raise RuntimeError("No detections produced")
    sub.insert(0, "id", np.arange(len(sub), dtype=int))
    expected = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    sub = sub[expected]
    if args.sample_submission and pd.read_csv(args.sample_submission, nrows=0).columns.tolist() != expected:
        raise ValueError("Sample submission columns do not match")
    if not np.isfinite(sub[["confidence", "x1", "y1", "x2", "y2"]].to_numpy()).all():
        raise ValueError("Non-finite submission values")
    if not sub["id"].is_unique:
        raise RuntimeError("Submission id is not unique")
    if not sub["class_id"].between(0, 17).all():
        raise RuntimeError("class_id outside 0..17")
    if not sub["confidence"].between(0, 1).all():
        raise RuntimeError("confidence outside 0..1")
    if not ((sub["x2"] > sub["x1"]) & (sub["y2"] > sub["y1"])).all():
        raise RuntimeError("Invalid boxes")
    out_path = Path(args.workdir) / "submission.csv"
    sub.to_csv(out_path, index=False)
    return out_path


def unique_stems(files):
    result = {}
    for path in files:
        if path.stem in result:
            raise ValueError(f"Duplicate image or XML ID: {path.stem}")
        result[path.stem] = path
    return result


def model_class(args):
    from ultralytics import RTDETR, YOLO

    return RTDETR if args.family == "rtdetr" else YOLO


def metric_summary(metrics):
    class_ids = [int(c) for c in metrics.box.ap_class_index]
    per_class = {}
    for position, class_id in enumerate(class_ids):
        per_class[str(class_id)] = {
            "ap50_95": float(metrics.box.ap[position]),
            "ap50": float(metrics.box.ap50[position]),
            "ap75": float(metrics.box.all_ap[position, 5]),
            "precision": float(metrics.box.p[position]),
            "recall": float(metrics.box.r[position]),
        }
    summary = {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "localization_gap_map50_minus_map75": float(metrics.box.map50 - metrics.box.map75),
        "per_class": per_class,
        "worst_present_class_ap": min((row["ap50_95"] for row in per_class.values()), default=None),
        "absent_class_ids": sorted(set(range(18)) - set(class_ids)),
    }
    matrix = np.asarray(metrics.confusion_matrix.matrix)
    if matrix.shape == (19, 19) and matrix.sum() > 0:
        true_positive = np.diag(matrix)[:18]
        summary["false_positives_per_class"] = (
            matrix[:18, :].sum(axis=1) - true_positive
        ).astype(int).tolist()
        summary["false_negatives_per_class"] = (
            matrix[:, :18].sum(axis=0) - true_positive
        ).astype(int).tolist()
        summary["confusion_matrix_predicted_rows_true_columns"] = matrix.astype(int).tolist()
    else:
        summary["false_positives_per_class"] = None
        summary["false_negatives_per_class"] = None
        summary["confusion_matrix_predicted_rows_true_columns"] = None
        summary["confusion_status"] = "unavailable; compute from saved predictions at an explicit operating point"
    values = [summary["map50_95"], summary["map50"], summary["map75"]]
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite evaluation metrics")
    return summary


def evaluate(args, best, yaml_path, splits, output_name, tta=False):
    model = model_class(args)(str(best))
    report = {
        "checkpoint": str(best),
        "sha256": hashlib.sha256(Path(best).read_bytes()).hexdigest(),
        "tta": tta,
    }
    for split in splits:
        metrics = model.val(data=str(yaml_path), split=split, imgsz=args.imgsz,
                            batch=args.batch if args.batch > 0 else 4,
                            device=args.device, workers=args.workers, conf=args.conf,
                            max_det=args.max_det, iou=args.iou, plots=False,
                            augment=tta,
                            project=str(Path(args.workdir) / "evaluation"),
                            name=f"{Path(output_name).stem}_{split}")
        report[split] = metric_summary(metrics)
        (Path(args.workdir) / output_name).write_text(json.dumps(report, indent=2))
    return report


def promote_candidate(base_report, candidate_report, min_gain):
    base = base_report["val"]
    candidate = candidate_report["val"]
    gain = candidate["map50_95"] - base["map50_95"]
    map75_gain = candidate["map75"] - base["map75"]
    worst_regression = candidate["worst_present_class_ap"] - base["worst_present_class_ap"]
    passed = (gain >= min_gain or (gain >= 0 and map75_gain >= min_gain)) and worst_regression >= -0.03
    return {
        "passed": passed,
        "map50_95_gain": gain,
        "map75_gain": map75_gain,
        "worst_present_class_ap_change": worst_regression,
        "minimum_gain": min_gain,
    }


def check_submission_gate(args, report):
    holdout_map = report["test"]["map50_95"]
    gate = {
        "metric": "holdout_mAP50_95",
        "value": holdout_map,
        "minimum": args.min_holdout_map,
        "passed": holdout_map >= args.min_holdout_map,
    }
    (Path(args.workdir) / "submission_gate.json").write_text(json.dumps(gate, indent=2))
    return gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cubes", required=True)
    p.add_argument("--test-cubes", required=True)
    p.add_argument("--ranking-cubes")
    p.add_argument("--train-xml", required=True)
    p.add_argument("--class-file", required=True)
    p.add_argument("--workdir", default="/kaggle/working/hsi_submission01")
    p.add_argument("--cache-dir")
    p.add_argument("--groups-file")
    p.add_argument("--split-manifest")
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--cv-fold-index", type=int)
    p.add_argument("--model", default="yolo26m.pt")
    p.add_argument("--checkpoint")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-policy", choices=["auto", "native", "square_pad"], default="auto")
    p.add_argument("--stem-init", choices=["random", "rgb_mean"], default="rgb_mean")
    p.add_argument("--normalization", choices=["N0", "N1", "N2", "N3", "N4"], default="N0")
    p.add_argument("--spectral-preprocess", choices=["SP0", "SP1", "SP2", "SP3"], default="SP0")
    p.add_argument("--normalization-samples-per-image", type=int, default=256)
    p.add_argument("--mnf-components", type=int)
    p.add_argument("--savgol-window", type=int, default=5)
    p.add_argument("--savgol-order", type=int, default=2)
    p.add_argument("--multi-scale", type=float, default=0.0)
    p.add_argument("--spectral-aug-copies", type=int, default=0)
    p.add_argument("--spectral-aug-classes", type=parse_class_ids, default=[])
    p.add_argument("--spectral-gain", type=float, default=0.05)
    p.add_argument("--spectral-tilt", type=float, default=0.03)
    p.add_argument("--spectral-noise", type=float, default=0.01)
    p.add_argument("--localization-epochs", type=int, default=0)
    p.add_argument("--localization-lr0", type=float, default=0.0001)
    p.add_argument("--promotion-min-gain", type=float, default=0.005)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--tta-min-gain", type=float, default=0.002)
    p.add_argument("--design-only", action="store_true")
    p.add_argument("--train-on-all", action="store_true")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--predict-conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.65)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--predict-max-det", type=int, default=300)
    p.add_argument("--min-holdout-map", type=float, default=0.50)
    p.add_argument("--family", choices=["yolo", "rtdetr"], default="yolo")
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--run-name", default="submission01")
    p.add_argument("--sample-submission")
    p.add_argument("--evaluate-holdout", action="store_true")
    args = p.parse_args()
    if args.spectral_aug_copies < 0:
        raise ValueError("spectral_aug_copies must be non-negative")
    if args.spectral_preprocess == "SP1" and args.mnf_components is None:
        raise ValueError("SP1 requires --mnf-components selected on validation")
    if not 0 <= args.multi_scale <= 0.9:
        raise ValueError("multi_scale must be in 0..0.9")
    if args.tta and args.family != "yolo":
        raise ValueError("Built-in validated TTA is enabled only for YOLO in this runner")
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable. Enable a Kaggle GPU and keep its CUDA PyTorch installation.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    versions = {name: importlib.metadata.version(name) for name in ("torch", "ultralytics", "numpy")}
    run_config = {
        "args": vars(args),
        "resolved_input_policy": resolve_input_policy(args.family, args.input_policy),
        "versions": versions,
    }
    (Path(args.workdir) / "run_config.json").write_text(json.dumps(run_config, indent=2))
    yaml_path, test_out, inference_geometry = build_dataset(args)
    if args.checkpoint:
        base_checkpoint = Path(args.checkpoint)
        if not base_checkpoint.exists():
            raise FileNotFoundError(base_checkpoint)
    else:
        base_checkpoint = train(args, yaml_path)
    base_report = evaluate(
        args,
        base_checkpoint,
        yaml_path,
        ["val"],
        "evaluation_base.json",
    )
    selected_checkpoint = base_checkpoint
    selected_report = base_report
    selection = {"base_checkpoint": str(base_checkpoint), "localization_finetune": None, "tta": None}
    finetuned_checkpoint = localization_finetune(args, yaml_path, base_checkpoint)
    if finetuned_checkpoint:
        finetuned_report = evaluate(
            args,
            finetuned_checkpoint,
            yaml_path,
            ["val"],
            "evaluation_localized.json",
        )
        decision = promote_candidate(base_report, finetuned_report, args.promotion_min_gain)
        decision["checkpoint"] = str(finetuned_checkpoint)
        selection["localization_finetune"] = decision
        if decision["passed"]:
            selected_checkpoint = finetuned_checkpoint
            selected_report = finetuned_report
    selected_tta = False
    if args.tta:
        tta_report = evaluate(
            args,
            selected_checkpoint,
            yaml_path,
            ["val"],
            "evaluation_tta.json",
            tta=True,
        )
        decision = promote_candidate(selected_report, tta_report, args.tta_min_gain)
        selection["tta"] = decision
        if decision["passed"]:
            selected_report = tta_report
            selected_tta = True
    selection["selected_checkpoint"] = str(selected_checkpoint)
    selection["selected_checkpoint_sha256"] = hashlib.sha256(Path(selected_checkpoint).read_bytes()).hexdigest()
    selection["selected_tta"] = selected_tta
    (Path(args.workdir) / "selection.json").write_text(json.dumps(selection, indent=2))
    if args.design_only:
        print(selection)
        print("Design-only run complete. Holdout and competition test prediction were not used.")
        return
    holdout_report = evaluate(
        args,
        selected_checkpoint,
        yaml_path,
        ["test"],
        "evaluation_holdout.json",
        tta=selected_tta,
    )
    report = {
        "checkpoint": str(selected_checkpoint),
        "sha256": selection["selected_checkpoint_sha256"],
        "tta": selected_tta,
        "val": selected_report["val"],
        "test": holdout_report["test"],
    }
    (Path(args.workdir) / "evaluation.json").write_text(json.dumps(report, indent=2))
    gate = check_submission_gate(args, report)
    print(gate)
    if not gate["passed"]:
        print("Submission blocked: holdout mAP50-95 is below minimum.")
        print(selected_checkpoint)
        return
    submission = predict(args, selected_checkpoint, test_out, inference_geometry, selected_tta)
    print(selected_checkpoint)
    print(submission)


if __name__ == "__main__":
    main()
