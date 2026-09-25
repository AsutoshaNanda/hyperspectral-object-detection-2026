from pathlib import Path
import hashlib
import json
import time

import numpy as np

import hsi_runner as runner
from evaluate_material_pairs import evaluate_material_pairs, extract_box_spectrum


OUT = Path("/kaggle/working/sam1_diagnostic")
OUT.mkdir(parents=True, exist_ok=True)


def one(root, pattern):
    matches = list(Path(root).rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {len(matches)}")
    return matches[0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    data = one("/kaggle/input", "hyperspectral-2026")
    plan = one("/kaggle/input", "hsi_plan_c_ratio")
    folds = Path("/kaggle/working/folds")
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "/kaggle/working/build_folds.py",
        "--xml-dir", str(data / "data_train/data_train/Annotations/VIS"),
        "--cube-dir", str(data / "data_train/data_train/VIS"),
        "--class-file", str(data / "class.txt"),
        "--legacy-split", str(plan / "split_manifest.json"),
        "--output-dir", str(folds),
        "--seed", "42",
    ], check=True)
    names = runner.read_classes(data / "class.txt")
    class_to_id = {name: index for index, name in enumerate(names)}
    cubes = runner.unique_stems(runner.collect_cubes(data / "data_train/data_train/VIS"))
    xmls = runner.annotation_map(data / "data_train/data_train/Annotations/VIS")
    development = sorted(set().union(*(set(json.loads((folds / f"fold{i}.json").read_text())["val"]) for i in range(3))))
    spectra_by_image = {}
    labels_by_image = {}
    counts = 0
    for image_index, stem in enumerate(development, 1):
        cube = runner.read_cube(cubes[stem])
        width, height, rows = runner.parse_voc(xmls[stem], class_to_id)
        spectra = []
        labels = []
        for label, cx, cy, bw, bh in rows:
            bbox = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
            spectra.append(extract_box_spectrum(cube, bbox, shrink_fraction=0.1, estimator="median"))
            labels.append(int(label))
        spectra_by_image[stem] = np.stack(spectra) if spectra else np.empty((0, 16))
        labels_by_image[stem] = np.asarray(labels, dtype=int)
        counts += len(labels)
        if image_index % 250 == 0:
            print(f"Extracted {image_index}/{len(development)} images", flush=True)
    reports = []
    for fold in range(3):
        split = json.loads((folds / f"fold{fold}.json").read_text())
        train_x = np.concatenate([spectra_by_image[stem] for stem in split["train"]])
        train_y = np.concatenate([labels_by_image[stem] for stem in split["train"]])
        val_x = np.concatenate([spectra_by_image[stem] for stem in split["val"]])
        val_y = np.concatenate([labels_by_image[stem] for stem in split["val"]])
        report = evaluate_material_pairs(
            train_x,
            train_y,
            val_x,
            val_y,
            names,
            pairs=(("apple", "apple_plastic"), ("egg", "egg_plastic")),
        )
        report["fold"] = fold
        report["split_hash"] = sha(folds / f"fold{fold}.json")
        reports.append(report)
        (OUT / f"fold{fold}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    aggregate = {}
    for pair_index, pair_name in enumerate(("apple_vs_apple_plastic", "egg_vs_egg_plastic")):
        rows = [report["pairs"][pair_index] for report in reports]
        aggregate[pair_name] = {
            "mean_auc": float(np.mean([row["evaluation_auc"] for row in rows])),
            "worst_auc": float(np.min([row["evaluation_auc"] for row in rows])),
            "mean_accuracy": float(np.mean([row["evaluation_accuracy"] for row in rows])),
            "worst_accuracy": float(np.min([row["evaluation_accuracy"] for row in rows])),
            "mean_margin_overlap": float(np.mean([row["evaluation_margin_overlap"] for row in rows])),
            "folds": rows,
        }
    summary = {
        "status": "measured_three_fold_diagnostic",
        "code_hash": sha(__file__),
        "fold_audit_hash": sha(folds / "fold_audit.json"),
        "fit_scope": "training_fold_only",
        "evaluation_scope": "validation_fold_only",
        "holdout_used": False,
        "test_or_ranking_used": False,
        "shrink_fraction": 0.1,
        "spectrum_estimator": "median",
        "prototype_estimator": "median",
        "objects": counts,
        "runtime_seconds": time.monotonic() - started,
        "pairs": aggregate,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
