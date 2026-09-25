import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_dataset(args):
    from hsi_runner import (
        annotation_map,
        band_limits,
        collect_cubes,
        parse_voc,
        prepare_cube,
        read_classes,
        read_cube,
        transform_rows,
        unique_stems,
    )

    cube_by_stem = unique_stems(collect_cubes(args.cube_dir))
    xml_by_stem = annotation_map(args.xml_dir)
    names = read_classes(args.class_file)
    class_to_id = {name: index for index, name in enumerate(names)}
    split_map = json.loads(Path(args.split_manifest).read_text())
    if set(split_map) != {"train", "val", "test"}:
        raise ValueError("Split manifest must contain train, val, and test")
    selected = sorted(set(split_map["train"]) | set(split_map["val"]))
    missing = sorted(set(selected) - set(cube_by_stem))
    if missing:
        raise ValueError(f"Missing {len(missing)} cubes from split manifest")
    missing_xml = sorted(set(selected) - set(xml_by_stem))
    if missing_xml:
        raise ValueError(f"Missing {len(missing_xml)} XML files from split manifest")

    train_files = [cube_by_stem[str(stem)] for stem in split_map["train"]]
    lo, hi = band_limits(train_files, seed=args.seed)
    output = Path(args.output_dir)
    annotations_dir = output / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    categories = [{"id": index, "name": name} for index, name in enumerate(names)]
    evidence = {
        "split_manifest": str(Path(args.split_manifest)),
        "split_sha256": sha256(args.split_manifest),
        "class_file_sha256": sha256(args.class_file),
        "input_policy": "square_pad",
        "normalization": "train-only percentile 0.5/99.5 to uint8",
        "train_images": len(split_map["train"]),
        "val_images": len(split_map["val"]),
        "holdout_images_not_used": len(split_map["test"]),
        "band_lo": lo.tolist(),
        "band_hi": hi.tolist(),
    }

    for split in ("train", "val"):
        image_dir = output / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        images = []
        annotations = []
        for image_index, stem in enumerate(map(str, split_map[split])):
            cube = read_cube(cube_by_stem[stem])
            xml_width, xml_height, rows = parse_voc(xml_by_stem[stem], class_to_id)
            if cube.shape[:2] != (xml_height, xml_width):
                raise ValueError(
                    f"{stem}: cube {cube.shape[:2]} differs from XML {(xml_height, xml_width)}"
                )
            prepared, geometry = prepare_cube(cube, lo, hi, "square_pad")
            prepared = np.ascontiguousarray(np.moveaxis(prepared, -1, 0))
            file_name = f"{stem}.npy"
            np.save(image_dir / file_name, prepared, allow_pickle=False)
            image_id = image_index + 1
            height, width = prepared.shape[-2:]
            images.append(
                {"id": image_id, "file_name": file_name, "height": height, "width": width}
            )
            for cls, cx, cy, bw, bh in transform_rows(rows, geometry):
                box_width = bw * width
                box_height = bh * height
                x = cx * width - box_width / 2
                y = cy * height - box_height / 2
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": int(cls),
                        "bbox": [float(x), float(y), float(box_width), float(box_height)],
                        "area": float(box_width * box_height),
                        "iscrowd": 0,
                    }
                )
        payload = {
            "info": {"description": "HSI 16-band fixed fold architecture screen"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        annotation_path = annotations_dir / f"instances_{split}.json"
        annotation_path.write_text(json.dumps(payload, separators=(",", ":")))
        evidence[f"{split}_annotations"] = len(annotations)
        evidence[f"{split}_annotation_sha256"] = sha256(annotation_path)

    evidence_path = output / "dataset_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube-dir", required=True)
    parser.add_argument("--xml-dir", required=True)
    parser.add_argument("--class-file", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args), indent=2))


if __name__ == "__main__":
    main()
