import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


EXCLUDED_IDS = ("1227", "1836", "1855")
AREA_EDGES = (0, 32**2, 96**2)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def unique_files(root, suffixes):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise ValueError(f"Duplicate image ID {path.stem}: {result[path.stem]}, {path}")
            result[path.stem] = path
    if not result:
        raise ValueError(f"No matching files in {root}")
    return result


def non_object_metadata(root):
    result = defaultdict(list)

    def visit(node, prefix):
        if node.tag.lower() == "object":
            return
        key = f"{prefix}/{node.tag}" if prefix else node.tag
        result[key].append({"text": (node.text or "").strip(), "attributes": dict(node.attrib)})
        for child in node:
            visit(child, key)

    visit(root, "")
    return dict(result)


def read_annotation(path, names):
    root = ET.parse(path).getroot()
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError(f"Invalid original dimensions in {path}")
    counts = np.zeros(len(names), dtype=np.int64)
    areas = []
    class_sizes = np.zeros((len(names), 3), dtype=np.int64)
    clipped = discarded = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in names:
            raise ValueError(f"Unknown class {name!r} in {path}")
        box = np.array([float(obj.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")])
        if not np.isfinite(box).all():
            raise ValueError(f"Non-finite box in {path}")
        bounded = np.clip(box, 0, [width, height, width, height])
        clipped += int(not np.array_equal(box, bounded))
        x1, y1, x2, y2 = bounded
        if x2 <= x1 or y2 <= y1:
            discarded += 1
            continue
        area = float((x2 - x1) * (y2 - y1))
        class_id = names.index(name)
        counts[class_id] += 1
        areas.append(area)
        class_sizes[class_id, int(np.searchsorted(AREA_EDGES[1:], area, side="right"))] += 1
    return {
        "width": width, "height": height, "class_counts": counts.tolist(),
        "class_presence": (counts > 0).astype(int).tolist(), "box_areas": areas,
        "class_size_counts": class_sizes.tolist(), "size_counts": class_sizes.sum(0).tolist(),
        "clipped_box_count": clipped, "discarded_box_count": discarded,
        "metadata": non_object_metadata(root),
    }


def load_groups(ids, groups_file=None, groups_evidence=None):
    if groups_file is None:
        if groups_evidence is not None:
            raise ValueError("--groups-evidence requires --groups-file")
        return {stem: stem for stem in ids}, {
            "mode": "image_id_only",
            "relationship_origin": "identity_fallback",
            "source_grouping_verified": False,
            "verification_status": "no_source_relationship_claimed",
        }
    if groups_evidence is None or not Path(groups_evidence).read_text().strip():
        raise ValueError("A group mapping requires a non-empty --groups-evidence file documenting the relationship")
    with Path(groups_file).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"image_id", "group_id"}:
            raise ValueError("Groups CSV must contain exactly image_id,group_id")
        groups = {}
        for row in reader:
            stem, group = row["image_id"].strip(), row["group_id"].strip()
            if not stem or not group or stem in groups:
                raise ValueError("Empty or duplicate image ID/group in groups CSV")
            groups[stem] = group
    if set(ids) - groups.keys():
        raise ValueError(f"Groups CSV missing IDs: {sorted(set(ids) - groups.keys())[:10]}")
    return {stem: groups[stem] for stem in ids}, {
        "mode": "provided_group_mapping",
        "relationship_origin": "caller_provided_csv",
        "source_grouping_verified": False,
        "verification_status": "requires_evidence_review",
        "groups_file_sha256": digest_file(groups_file),
        "groups_evidence_sha256": digest_file(groups_evidence),
        "groups_evidence": Path(groups_evidence).read_text(),
        "unused_group_ids": sorted(set(groups) - set(ids)),
    }


def check_split(split, ids, groups):
    flat = [stem for part in ("train", "val", "test") for stem in split[part]]
    if len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError("Split must cover every usable image exactly once")
    seen = {}
    for part in ("train", "val", "test"):
        if not split[part]:
            raise ValueError(f"Empty {part} partition")
        for stem in split[part]:
            group = groups[stem]
            if group in seen and seen[group] != part:
                raise ValueError(f"Group {group} leaks across partitions")
            seen[group] = part


def assign_groups(ids, groups, records, seed):
    members = defaultdict(list)
    for stem in sorted(ids):
        members[groups[stem]].append(stem)
    if len(members) < 3:
        raise ValueError("Three-fold CV requires at least three development groups")
    keys = sorted(members)
    vectors = np.asarray([
        np.sum([records[s]["class_presence"] + records[s]["class_counts"] + records[s]["size_counts"] + [1] for s in members[g]], axis=0).tolist() + [1]
        for g in keys
    ], dtype=float)
    total = vectors.sum(0)
    active = total > 0
    normalized = vectors[:, active] / total[active]
    rng = np.random.default_rng(seed)
    ties = rng.random(len(keys))
    order = sorted(range(len(keys)), key=lambda i: (-float(np.max(normalized[i])), -float(np.sum(normalized[i])), ties[i]))
    loads = np.zeros((3, normalized.shape[1]))
    counts = np.zeros(3, dtype=int)
    chosen = {}
    priority = rng.permutation(3).tolist()
    for i in order:
        scores = []
        for fold in range(3):
            trial = loads.copy()
            trial[fold] += normalized[i]
            scores.append(float(np.square(trial - trial.mean(0)).sum()))
        fold = min(range(3), key=lambda f: (scores[f], counts[f], priority.index(f)))
        loads[fold] += normalized[i]
        counts[fold] += len(members[keys[i]])
        chosen[keys[i]] = fold
    return {stem: chosen[groups[stem]] for stem in ids}


def summarize(ids, records, groups, names):
    counts = np.sum([records[s]["class_counts"] for s in ids], axis=0).astype(int)
    presence = np.sum([records[s]["class_presence"] for s in ids], axis=0).astype(int)
    areas = [area for s in ids for area in records[s]["box_areas"]]
    histogram, _ = np.histogram(areas, bins=[0, 256, 1024, 4096, 9216, 16384, 65536, np.inf])
    return {
        "image_count": len(ids), "group_count": len({groups[s] for s in ids}),
        "object_count": int(counts.sum()), "class_object_counts": dict(zip(names, counts.tolist())),
        "class_image_counts": dict(zip(names, presence.tolist())),
        "missing_classes": [name for name, count in zip(names, counts) if count == 0],
        "size_counts": dict(zip(("small", "medium", "large"), np.sum([records[s]["size_counts"] for s in ids], axis=0).astype(int).tolist())),
        "class_size_counts": dict(zip(names, np.sum([records[s]["class_size_counts"] for s in ids], axis=0).astype(int).tolist())),
        "box_area_histogram": {"edges": [0, 256, 1024, 4096, 9216, 16384, 65536, "infinity"], "counts": histogram.tolist()},
    }


def build_folds(xml_dir, cube_dir, class_file, legacy_split, output_dir, groups_file=None, groups_evidence=None, seed=42):
    names = [line.strip() for line in Path(class_file).read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Class names must be nonempty and unique")
    xmls = unique_files(xml_dir, {".xml"})
    cubes = unique_files(cube_dir, {".png", ".npy", ".npz", ".tif", ".tiff"})
    if set(xmls) != set(cubes):
        raise ValueError("Annotation and cube image IDs differ")
    ids = sorted(set(xmls) - set(EXCLUDED_IDS))
    legacy = json.loads(Path(legacy_split).read_text())
    if set(legacy) != {"train", "val", "test"}:
        raise ValueError("Legacy split requires train, val, test")
    legacy = {key: list(map(str, value)) for key, value in legacy.items()}
    check_split(legacy, ids, {stem: stem for stem in ids})
    records = {stem: read_annotation(xmls[stem], names) for stem in ids}
    groups, group_info = load_groups(ids, groups_file, groups_evidence)
    holdout_groups = {groups[stem] for stem in legacy["test"]}
    holdout = sorted(stem for stem in ids if groups[stem] in holdout_groups)
    development = sorted(set(ids) - set(holdout))
    assignments = assign_groups(development, groups, records, seed)
    folds = []
    for index in range(3):
        split = {"train": [s for s in development if assignments[s] != index], "val": [s for s in development if assignments[s] == index], "test": holdout}
        check_split(split, ids, groups)
        folds.append(split)
    if sorted(s for split in folds for s in split["val"]) != development:
        raise AssertionError("Development images must validate exactly once")
    fields = defaultdict(Counter)
    inventory = {}
    for stem, record in records.items():
        for key, values in record["metadata"].items():
            fields[key].update(json.dumps(v, sort_keys=True) for v in values)
        inventory[stem] = {
            "xml": str(xmls[stem].relative_to(xml_dir)), "cube": str(cubes[stem].relative_to(cube_dir)),
            "xml_sha256": digest_file(xmls[stem]), "cube_size_bytes": cubes[stem].stat().st_size,
            "xml_metadata": record["metadata"], "original_width": record["width"], "original_height": record["height"],
            "original_box_areas": record["box_areas"], "clipped_box_count": record["clipped_box_count"],
            "discarded_box_count": record["discarded_box_count"],
        }
    grouping_fields = [key for key in fields if any(word in key.lower() for word in ("capture", "session", "source", "folder", "sequence", "scene", "video"))]
    candidate_fields = [key for key in grouping_fields if len(fields[key]) > 1]
    non_informative_fields = [key for key in grouping_fields if len(fields[key]) <= 1]
    config = {"seed": seed, "cv_folds": 3, "excluded_ids": list(EXCLUDED_IDS), "area_bins": list(AREA_EDGES),
              "legacy_split_sha256": digest_file(legacy_split), "class_file_sha256": digest_file(class_file),
              "group_mapping_sha256": digest_json(groups), "annotation_inventory_sha256": digest_json(inventory),
              "groups_evidence_sha256": group_info.get("groups_evidence_sha256"),
              "algorithm": "seeded_greedy_multilabel_presence_object_counts_original_area_bins_image_and_group_counts"}
    audit = {
        "status": "folds_built_not_training_results", "code_hash": digest_file(__file__), "config": config,
        "config_hash": digest_json(config), "groups": group_info,
        "excluded_ids": {stem: {"present": stem in xmls,
                                "reason": "Existing runner exclusion retained for comparability; the historical mismatch rationale was not re-verified",
                                "evidence_status": "historical_policy_unverified"} for stem in EXCLUDED_IDS},
        "holdout": {"status": "previously_inspected", "legacy_count": len(legacy["test"]), "group_closed_count": len(holdout),
                    "added_for_group_closure": sorted(set(holdout) - set(legacy["test"])),
                    "previously_trained_by_legacy_checkpoint": sorted(set(holdout) & set(legacy["train"])),
                    "legacy_checkpoint_declared_group_overlap": sorted(holdout_groups & {groups[s] for s in legacy["train"]}),
                    "use": "Freeze new designs before evaluation; cannot claim an untouched final holdout. Existing checkpoints cannot provide new CV evidence."},
        "metadata_audit": {"non_object_fields": {key: {"unique_values": len(values), "value_counts": dict(sorted(values.items()))} for key, values in sorted(fields.items())},
                           "candidate_group_fields": candidate_fields,
                           "non_informative_group_fields": non_informative_fields,
                           "unresolved_metadata_candidates": candidate_fields if groups_file is None else [],
                           "xml_folders": dict(Counter(str(path.parent.relative_to(xml_dir)) for path in xmls.values())),
                           "cube_folders": dict(Counter(str(path.parent.relative_to(cube_dir)) for path in cubes.values())),
                           "numeric_filename_count": sum(stem.isdigit() for stem in ids),
                           "numeric_name_policy": "Numbers alone are not capture/session evidence; no numeric or visual grouping inferred",
                           "conclusion": ("No varying official capture/session/source field exists; image_id_only is the supported fallback" if not candidate_fields else "Only image identities are currently supported; varying source/session candidates require provenance review") if groups_file is None else "Caller-provided mapping used for isolation; no source relationship is claimed until attached provenance is reviewed"},
        "coordinate_policy": "Original XML pixel coordinates; clip to XML image bounds; discard nonpositive boxes as existing runner does; no resize before areas",
        "assertions": {"image_disjoint": True, "declared_group_mapping_disjoint": True,
                       "verified_source_group_disjoint": True if group_info["source_grouping_verified"] else None,
                       "all_usable_ids_covered": True, "validation_once_per_development_id": True,
                       "legacy_holdout_preserved": True},
        "folds": [{"fold": i, "split_hash": digest_json(split), "partitions": {part: summarize(stems, records, groups, names) for part, stems in split.items()}} for i, split in enumerate(folds)],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"fold{i}.json" for i in range(3)] + [out / "groups.csv", out / "fold_audit.json", out / "metadata_inventory.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Use a fresh output directory; frozen fold artifacts cannot be overwritten")
    for i, split in enumerate(folds):
        (out / f"fold{i}.json").write_text(json.dumps(split, indent=2) + "\n")
    with (out / "groups.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "group_id"])
        writer.writerows((stem, groups[stem]) for stem in ids)
    (out / "fold_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    (out / "metadata_inventory.json").write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", required=True)
    parser.add_argument("--cube-dir", required=True)
    parser.add_argument("--class-file", required=True)
    parser.add_argument("--legacy-split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups-file")
    parser.add_argument("--groups-evidence")
    parser.add_argument("--seed", type=int, default=42)
    audit = build_folds(**vars(parser.parse_args()))
    print(json.dumps({"status": audit["status"], "grouping": audit["groups"]["mode"], "fold_sizes": [{part: value["image_count"] for part, value in fold["partitions"].items()} for fold in audit["folds"]], "holdout": audit["holdout"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
