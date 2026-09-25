import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "Experiments"
sys.path.insert(0, str(EXPERIMENTS))

from build_registry import build_registry
from orchestration.ledger import CONCLUSION_FIELDS, LedgerError, append_record, verify_ledger
from orchestration.queue import promote_stage, ready_obligations
from orchestration.ranking import build_rankings, write_rankings
from orchestration.report import write_report
from orchestration.state import StateError, initialize_state, load_state, save_state, transition


def result_record(registry, experiment_id="V1", fold=0, score=0.5, status="measured", stage=None, requirement_id=None):
    record = {field: None for field in registry["required_ledger_fields"]}
    record.update(
        {
            "experiment_id": experiment_id,
            "code_hash": "code",
            "config_hash": "config",
            "split_hash": "split",
            "fold": fold,
            "validation_map50_95": score,
            "validation_map50": score + 0.1,
            "validation_map75": score - 0.1,
            "AP_small": score - 0.2,
            "AP_medium": score,
            "AP_large": score + 0.1,
            "worst_present_class_AP": score - 0.25,
            "holdout_map50_95": None,
            "status": status,
            "decision": "retain for controlled comparison",
            "observation": "measured on fixed evidence",
            "evidence_source": "result.json",
            "controlled_change": "one registry change",
            "cv_result": score,
            "worst_fold": score,
            "map75": score - 0.1,
            "ap_small": score - 0.2,
            "worst_class": score - 0.25,
            "material_pair_result": None,
            "next_experiment": "next registered obligation",
            "stage": stage,
            "obligation_id": requirement_id or experiment_id,
        }
    )
    for field in CONCLUSION_FIELDS:
        record.setdefault(field, None)
    return record


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry()

    def test_append_only_hash_chain_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            first = append_record(path, result_record(self.registry), self.registry["required_ledger_fields"])
            second = append_record(path, result_record(self.registry, fold=1), self.registry["required_ledger_fields"])
            self.assertNotEqual(first["record_hash"], second["record_hash"])
            self.assertEqual(verify_ledger(path, self.registry["required_ledger_fields"])["chained_records"], 2)
            lines = path.read_text().splitlines()
            changed = json.loads(lines[0])
            changed["decision"] = "tampered"
            lines[0] = json.dumps(changed)
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaises(LedgerError):
                verify_ledger(path, self.registry["required_ledger_fields"])

    def test_measurement_rejects_missing_required_or_nonfinite_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            missing = result_record(self.registry)
            del missing["split_hash"]
            with self.assertRaises(LedgerError):
                append_record(path, missing, self.registry["required_ledger_fields"])
            nonfinite = result_record(self.registry)
            nonfinite["validation_map50_95"] = float("nan")
            with self.assertRaises(LedgerError):
                append_record(path, nonfinite, self.registry["required_ledger_fields"])


class StateTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry()
        self.state = initialize_state(self.registry)

    def test_evidence_is_required_for_measurement_and_verification(self):
        transition(self.state, "V1", "implemented", "i", config_hash="cfg")
        transition(self.state, "V1", "running", "r")
        with self.assertRaises(StateError):
            transition(self.state, "V1", "measured")
        transition(self.state, "V1", "measured", "m")
        with self.assertRaises(StateError):
            transition(self.state, "V1", "verified")
        transition(self.state, "V1", "verified", "v")
        self.assertEqual(self.state["obligations"]["V1"]["status"], "verified")

    def test_dependencies_block_runs_but_not_implementation(self):
        transition(self.state, "N0", "implemented", "i", config_hash="cfg")
        with self.assertRaises(StateError):
            transition(self.state, "N0", "running", "r")
        transition(self.state, "V1", "implemented", "i", config_hash="cfg")
        transition(self.state, "V1", "running", "r")
        transition(self.state, "V1", "measured", "m")
        transition(self.state, "V1", "verified", "v")
        transition(self.state, "N0", "running", "r")

    def test_valid_stop_reason_is_mandatory(self):
        with self.assertRaises(StateError):
            transition(self.state, "V1", "stopped", "s", stop_reason="paper_was_rgb")
        transition(self.state, "V1", "stopped", "s", stop_reason="invalid_predictions")

    def test_rank_slots_are_not_ready_before_evidence_selection(self):
        ids = {row["id"] for row in ready_obligations(self.state)}
        self.assertIn("V1", ids)
        self.assertIn("A-C1:C0", ids)
        self.assertNotIn("C1-rank1-second_fold", ids)

    def test_state_write_rejects_stale_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            first = initialize_state(self.registry)
            save_state(path, first)
            stale = load_state(path, self.registry)
            current = load_state(path, self.registry)
            save_state(path, current)
            with self.assertRaises(StateError):
                save_state(path, stale)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry()

    def records(self):
        records = []
        for experiment_id, base in (("N0", 0.60), ("N1", 0.65), ("A-C1", 0.70)):
            for fold in range(3):
                record = result_record(self.registry, experiment_id, fold, base + fold * 0.001, "verified")
                record["event_schema_version"] = 1
                record["record_hash"] = f"{experiment_id}-{fold}"
                record["public_score"] = 0.99 if experiment_id == "N0" else 0.01
                records.append(record)
        holdout = result_record(self.registry, "A-C1", "untouched_holdout", 0.0, "verified")
        holdout["validation_map50_95"] = None
        holdout["holdout_map50_95"] = 0.68
        holdout["event_schema_version"] = 1
        holdout["record_hash"] = "A-C1-holdout"
        records.append(holdout)
        return records

    def test_separate_rankings_use_cv_not_public_score(self):
        tables = build_rankings(self.registry, self.records())
        self.assertEqual([row["id"] for row in tables["individual"]], ["N1", "N0"])
        self.assertEqual([row["id"] for row in tables["combination"]], ["A-C1"])
        self.assertEqual([row["id"] for row in tables["finalist"]], ["A-C1"])

    def test_csvs_exist_even_before_results(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            output = Path(directory) / "results"
            paths, tables = write_rankings(self.registry, ledger, output)
            self.assertTrue(all(path.exists() for path in paths.values()))
            self.assertTrue(all(not rows for rows in tables.values()))
            with paths["individual"].open() as stream:
                self.assertIn("cv_mean_map50_95", next(csv.reader(stream)))

    def test_stage_promotion_refuses_insufficient_measured_candidates(self):
        state = initialize_state(self.registry)
        for obligation in state["obligations"].values():
            if obligation.get("stage") == "C0":
                obligation["status"] = "verified"
        with self.assertRaises(StateError):
            promote_stage(state, self.registry, [], "C0", "C1")
        records = []
        combinations = [row["id"] for row in self.registry["experiments"] if row["kind"] == "combination"]
        for index, experiment_id in enumerate(combinations):
            record = result_record(self.registry, experiment_id, "fixed_design_fold", 0.5 + index / 1000, "verified", "C0", experiment_id + ":C0")
            record["event_schema_version"] = 1
            record["record_hash"] = "hash-" + experiment_id
            records.append(record)
        selected = promote_stage(state, self.registry, records, "C0", "C1")
        self.assertEqual(len(selected), 15)
        self.assertEqual(state["obligations"]["C1-rank1-second_fold"]["bound_experiment_id"], selected[0]["id"])

    def test_report_never_claims_empty_plan_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            state = initialize_state(self.registry)
            output = Path(directory) / "results"
            result = write_report(self.registry, state, output / "ledger.jsonl", output, output / "report.md")
            self.assertFalse(result["complete"])
            self.assertIn("Roadmap incomplete", (output / "report.md").read_text())


class CliTests(unittest.TestCase):
    def test_full_lifecycle_requires_matching_implementation_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            ledger = root / "ledger.jsonl"
            config = root / "config.json"
            result = root / "result.json"
            command = [
                sys.executable,
                str(EXPERIMENTS / "run_experiment.py"),
                "--state",
                str(state),
                "--ledger",
                str(ledger),
            ]
            subprocess.run(command + ["init"], check=True, capture_output=True, text=True)
            config.write_text(
                json.dumps(
                    {
                        "obligation_id": "V1",
                        "experiment_id": "V1",
                        "code_paths": [str(EXPERIMENTS / "run_experiment.py")],
                        "command": [sys.executable, "-c", "pass"],
                        "result_path": str(result),
                    }
                )
            )
            subprocess.run(command + ["implement", "V1", "--config", str(config)], check=True, capture_output=True, text=True)
            subprocess.run(command + ["start", "V1"], check=True, capture_output=True, text=True)
            current = json.loads(state.read_text())["obligations"]["V1"]
            record = result_record(build_registry())
            record["config_hash"] = current["config_hash"]
            record["code_hash"] = current["code_hash"]
            result.write_text(json.dumps(record))
            subprocess.run(command + ["measure", "V1", "--record", str(result)], check=True, capture_output=True, text=True)
            subprocess.run(command + ["verify", "V1"], check=True, capture_output=True, text=True)
            final = json.loads(state.read_text())["obligations"]["V1"]
            self.assertEqual(final["status"], "verified")
            self.assertEqual(verify_ledger(ledger, build_registry()["required_ledger_fields"])["chained_records"], 4)


if __name__ == "__main__":
    unittest.main()
