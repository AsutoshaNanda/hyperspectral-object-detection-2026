import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from build_registry import SOURCE, build_registry, validate_registry


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = build_registry()
        cls.experiments = {row["id"]: row for row in cls.registry["experiments"]}

    def test_source_is_exact_and_complete(self):
        self.assertEqual(self.registry["source"]["sha256"], hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.assertEqual(set(self.registry["source_sections"]), {str(index) for index in range(1, 43)})
        lines = SOURCE.read_text().splitlines(keepends=True)
        for section in self.registry["source_sections"].values():
            self.assertEqual(section["text"], "".join(lines[section["start_line"] - 1:section["end_line"]]))

    def test_all_numbered_and_bold_experiments_are_covered(self):
        source = SOURCE.read_text()
        ids = set(re.findall(r"^## ((?:N|SP|DF|T|SAM|PL|P|E1|SN)\d+[a-j]?)\s+[—-]", source, re.M))
        ids.update(re.findall(r"\*\*((?:A1|H1|S1|G1|L1|M1)[a-d]):\*\*", source))
        ids.update(["V1", "CR1", "S1-ViT-random", "S1-ViT-rgb_expanded", "D1-YOLO26m", "D1-RT-DETR-L", "A1-audit"])
        self.assertFalse(ids - set(self.experiments))
        for family in ("SN1", "SN2"):
            self.assertEqual(self.experiments[family]["variants"], [family + "-" + variant for variant in ("normal", "linear", "gaussian")])

    def test_35_exact_combination_formulas(self):
        source = SOURCE.read_text()
        expected = dict(re.findall(r"^## ([A-G]-C[1-5])\s*\n```text\n([^`]+)```", source, re.M))
        self.assertEqual(len(expected), 35)
        for identifier, formula in expected.items():
            self.assertEqual(self.experiments[identifier]["settings"]["expression"], formula.strip())
        self.assertEqual(len(self.experiments["G-C5"]["prerequisites"]), 25)

    def test_stages_and_gate_counts(self):
        validate_registry(self.registry)
        requirements = self.registry["execution_requirements"]
        ids = [row["id"] for row in requirements]
        self.assertEqual(len(ids), len(set(ids)))
        for letter in "abcdefghij":
            self.assertIn("E1" + letter + ":E1-0", ids)
            self.assertIn("E1" + letter + ":E1-1", ids)
        self.assertEqual(sum(row["stage"] == "C2" for row in requirements), 30)
        self.assertEqual(sum(row["stage"] == "C3" for row in requirements), 3)

    def test_registry_contains_no_fabricated_results(self):
        for collection in ("experiments", "execution_requirements", "definition_of_done"):
            for row in self.registry[collection]:
                self.assertEqual(row["status"], "pending")
                self.assertIsNone(row["evidence"])
        self.assertEqual(self.registry["coverage"]["definition_of_done_requirements"], 33)

    def test_regeneration_matches_saved_manifest(self):
        saved = json.loads(Path(__file__).with_name("registry.json").read_text())
        self.assertEqual(saved, self.registry)


if __name__ == "__main__":
    unittest.main()
