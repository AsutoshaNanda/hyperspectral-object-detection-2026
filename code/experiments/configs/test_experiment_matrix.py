import importlib.util
import json
import re
import unittest
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent
ROOT = CONFIG_DIR.parents[1]
SPEC = importlib.util.spec_from_file_location("build_experiment_matrix", CONFIG_DIR / "build_experiment_matrix.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class ExperimentMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = BUILDER.build_matrix()
        cls.saved = json.loads((CONFIG_DIR / "experiment_matrix.json").read_text())
        cls.registry = json.loads((ROOT / "Experiments" / "registry.json").read_text())
        cls.roadmap = (ROOT / "Resources" / "Hyperspectral_5_Day_Experiment_Roadmap.md").read_text()

    def test_saved_matrix_is_current_and_valid(self):
        self.assertEqual(self.saved, self.matrix)
        self.assertEqual(BUILDER.validate_matrix(self.matrix)["C0"], 35)

    def test_no_roadmap_experiment_or_combination_is_omitted(self):
        expected_individuals = {row["id"] for row in self.registry["experiments"] if row["kind"] != "combination"}
        actual_individuals = {row["id"] for row in self.matrix["individual_experiments"]}
        self.assertEqual(actual_individuals, expected_individuals)
        expected_formulas = dict(re.findall(r"^## ([A-G]-C[1-5])\s*\n```text\n([^`]+)```", self.roadmap, re.M))
        actual_formulas = {row["id"]: row["expression"] for row in self.matrix["combination_experiments"]}
        self.assertEqual(len(expected_formulas), 35)
        self.assertEqual(actual_formulas, {key: value.strip() for key, value in expected_formulas.items()})

    def test_all_required_variant_families_have_matched_controls(self):
        ids = {row["id"] for row in self.matrix["individual_experiments"]}
        required = {
            "N0", "N1", "N2", "N3", "N4",
            "SP0", "SP1", "SP2", "SP3", "SP4",
            "H1a", "H1b", "H1c", "H1d",
            "S1a", "S1b", "S1-ViT-random", "S1-ViT-rgb_expanded",
            "G1a", "G1b", "G1c",
            "L1a", "L1b", "L1c", "L1d",
            "DF1", "DF2", "DF3", "DF1-random", "DF1-rgb_expanded",
            "M1a", "M1b", "M1c", "M1d",
            "T1", "T2", "T3", "T4",
            "SN1-normal", "SN1-linear", "SN1-gaussian",
            "SN2-normal", "SN2-linear", "SN2-gaussian",
            "SAM1", "SAM2", "SAM3",
            "PL1", "PL2", "PL3", "CR1",
            "P1", "P2", "P3", "P4", "P5",
            *(f"E1{x}" for x in "abcdefghij"),
        }
        self.assertFalse(required - ids)
        for row in self.matrix["individual_experiments"]:
            self.assertIn("matched_control", row)
            self.assertTrue(row["matched_control"]["same_seed"])
            self.assertTrue(row["matched_control"]["same_v1_fold"])
            self.assertTrue(row["matched_control"]["same_budget"])

    def test_vit_stem_variants_expand_to_every_patch_projection_candidate(self):
        slots = self.matrix["expanded_variant_runs"]["vit_patch_projection"]
        expected = {
            f"{stem}@{candidate}"
            for stem in ("S1-ViT-random", "S1-ViT-rgb_expanded")
            for candidate in BUILDER.PATCH_PROJECTION_CANDIDATES
        }
        self.assertEqual({row["slot_id"] for row in slots}, expected)

    def test_unresolved_result_selected_configs_are_not_runnable(self):
        rows = [*self.matrix["individual_experiments"], *self.matrix["combination_experiments"]]
        for row in rows:
            for reference in row["unresolved_references"]:
                self.assertIn(reference["type"], {"verified_result_selection", "validated_parameter_selection", "verified_artifact"})
                self.assertIsNone(reference.get("evidence_record_hash"))
            if row["unresolved_references"]:
                self.assertFalse(row["runnable"], row["id"])
        self.assertTrue(all(not row["runnable"] for row in self.matrix["combination_experiments"]))

    def test_stage_slots_and_budgets_are_complete(self):
        e1 = self.matrix["e1_stages"]
        stages = self.matrix["combination_stages"]
        self.assertEqual(len(e1["smoke"]), 10)
        self.assertTrue(all(row["epochs"] == 1 for row in e1["smoke"]))
        self.assertEqual(len(e1["screen"]), 10)
        self.assertTrue(all(row["epochs"] == 10 for row in e1["screen"]))
        self.assertEqual(len(e1["screen_controls"]), 2)
        self.assertTrue(all(row["epochs"] == 10 for row in e1["screen_controls"]))
        self.assertEqual(len(e1["full_cv_top3"]), 9)
        self.assertEqual(len(stages["C0_all_35"]), 35)
        self.assertTrue(all(row["epochs"] == 10 for row in stages["C0_all_35"]))
        self.assertEqual(len(stages["C1_top15"]), 15)
        self.assertTrue(all(row["epochs"] == 10 for row in stages["C1_top15"]))
        self.assertEqual(len(stages["C2_top10_three_folds"]), 30)
        self.assertEqual(len(stages["C3_top3_holdout"]), 3)
        self.assertEqual(self.matrix["control_profiles"]["transformer_control"]["epochs"], 50)
        self.assertEqual(self.matrix["control_profiles"]["yolo_control"]["epochs"], 80)

    def test_holdout_is_never_used_by_a_selection_stage(self):
        groups = [*self.matrix["e1_stages"].values(), *self.matrix["combination_stages"].values()]
        for group in groups:
            for slot in group:
                if slot["purpose"] == "selection":
                    self.assertNotIn("holdout", slot["data_roles"], slot["slot_id"])
        for slot in self.matrix["combination_stages"]["C3_top3_holdout"]:
            self.assertEqual(slot["purpose"], "final_evaluation")
            self.assertFalse(slot["selection_eligible"])
            self.assertFalse(slot["runnable"])

    def test_no_cross_model_fusion_anywhere(self):
        groups = [
            self.matrix["individual_experiments"],
            self.matrix["combination_experiments"],
            self.matrix["expanded_variant_runs"]["vit_patch_projection"],
            *self.matrix["e1_stages"].values(),
            *self.matrix["combination_stages"].values(),
        ]
        for group in groups:
            for row in group:
                contract = row["model_contract"]
                self.assertEqual(contract["trained_model_count"], 1)
                self.assertEqual(contract["checkpoint_count"], 1)
                self.assertTrue(contract["same_checkpoint_inference_views_only"])
                self.assertFalse(contract["cross_model_prediction_fusion"])

    def test_configuration_contains_no_experiment_results(self):
        self.assertIsNone(self.matrix["results"])
        self.assertTrue(all(row["result"] is None for row in self.matrix["individual_experiments"]))
        self.assertTrue(all(row["result"] is None for row in self.matrix["combination_experiments"]))


if __name__ == "__main__":
    unittest.main()
