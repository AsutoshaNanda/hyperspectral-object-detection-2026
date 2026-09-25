import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "Experiments" / "build_folds.py"
SPEC = importlib.util.spec_from_file_location("build_folds", MODULE_PATH)
FOLDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FOLDS)


def dataset(root, image_ids):
    xml_dir = root / "xml"
    cube_dir = root / "cubes"
    xml_dir.mkdir()
    cube_dir.mkdir()
    xml = """<annotation><source><session>same-session</session></source><size><width>200</width><height>100</height></size><object><name>item</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>33</xmax><ymax>33</ymax></bndbox></object></annotation>"""
    for image_id in image_ids:
        (xml_dir / f"{image_id}.xml").write_text(xml)
        (cube_dir / f"{image_id}.npy").write_bytes(b"cube")
    class_file = root / "classes.txt"
    class_file.write_text("item\n")
    return xml_dir, cube_dir, class_file


class GroupAuditTests(unittest.TestCase):
    def test_identity_fallback_never_infers_relationships(self):
        groups, audit = FOLDS.load_groups(["100", "101", "102"])
        self.assertEqual(groups, {"100": "100", "101": "101", "102": "102"})
        self.assertEqual(audit["mode"], "image_id_only")
        self.assertIs(audit["source_grouping_verified"], False)
        self.assertEqual(audit["verification_status"], "no_source_relationship_claimed")

    def test_caller_mapping_is_not_reported_as_verified_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups_file = root / "groups.csv"
            evidence_file = root / "evidence.txt"
            with groups_file.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_id", "group_id"])
                writer.writerows((("100", "capture-a"), ("101", "capture-a"), ("102", "capture-b")))
            evidence_file.write_text("Grouping supplied by the dataset owner; manual review is pending.\n")
            groups, audit = FOLDS.load_groups(["100", "101", "102"], groups_file, evidence_file)
        self.assertEqual(groups["100"], groups["101"])
        self.assertEqual(audit["mode"], "provided_group_mapping")
        self.assertIs(audit["source_grouping_verified"], False)
        self.assertEqual(audit["verification_status"], "requires_evidence_review")

    def test_metadata_candidates_are_audited_without_becoming_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output"
            image_ids = [str(index) for index in range(1, 8)]
            xml_dir, cube_dir, class_file = dataset(root, image_ids)
            legacy_split = root / "legacy.json"
            legacy_split.write_text(json.dumps({"train": image_ids[:5], "val": ["6"], "test": ["7"]}))
            audit = FOLDS.build_folds(xml_dir, cube_dir, class_file, legacy_split, output_dir)
            with (output_dir / "groups.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertIn("annotation/source", audit["metadata_audit"]["non_informative_group_fields"])
        self.assertEqual(audit["metadata_audit"]["candidate_group_fields"], [])
        self.assertEqual({row["image_id"]: row["group_id"] for row in rows}, {stem: stem for stem in image_ids})
        self.assertEqual(audit["groups"]["mode"], "image_id_only")
        self.assertIs(audit["groups"]["source_grouping_verified"], False)
        self.assertIsNone(audit["assertions"]["verified_source_group_disjoint"])
        self.assertEqual(audit["excluded_ids"]["1227"]["evidence_status"], "historical_policy_unverified")
        self.assertEqual(audit["coordinate_policy"], "Original XML pixel coordinates; clip to XML image bounds; discard nonpositive boxes as existing runner does; no resize before areas")

    def test_provided_mapping_closes_holdout_without_claiming_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_ids = [str(index) for index in range(1, 10)]
            xml_dir, cube_dir, class_file = dataset(root, image_ids)
            legacy_split = root / "legacy.json"
            legacy_split.write_text(json.dumps({"train": image_ids[:7], "val": ["8"], "test": ["9"]}))
            groups_file = root / "groups.csv"
            with groups_file.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_id", "group_id"])
                writer.writerows((image_id, "shared" if image_id in {"1", "9"} else image_id) for image_id in image_ids)
            evidence_file = root / "evidence.txt"
            evidence_file.write_text("Dataset-owner mapping awaiting manual review.\n")
            output_dir = root / "output"
            audit = FOLDS.build_folds(
                xml_dir,
                cube_dir,
                class_file,
                legacy_split,
                output_dir,
                groups_file,
                evidence_file,
            )
            splits = [json.loads((output_dir / f"fold{index}.json").read_text()) for index in range(3)]
        self.assertEqual(audit["holdout"]["added_for_group_closure"], ["1"])
        self.assertEqual(audit["holdout"]["previously_trained_by_legacy_checkpoint"], ["1"])
        self.assertIs(audit["groups"]["source_grouping_verified"], False)
        for split in splits:
            self.assertEqual(split["test"], ["1", "9"])
            self.assertNotIn("1", split["train"] + split["val"])


if __name__ == "__main__":
    unittest.main()
