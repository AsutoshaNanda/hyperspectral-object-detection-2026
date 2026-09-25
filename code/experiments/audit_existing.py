from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Experiments/results"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def append_record(record):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "experiment_ledger.jsonl"
    payload_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(row.get("record_hash") == payload_hash for row in existing):
        return
    record = dict(record, record_hash=payload_hash, recorded_at=datetime.now(timezone.utc).isoformat())
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def audit_csv(path, geometry):
    frame = pd.read_csv(path)
    columns = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    assert frame.columns.tolist() == columns
    assert np.isfinite(frame.to_numpy()).all()
    assert frame.id.is_unique and np.array_equal(frame.id.to_numpy(), np.arange(len(frame)))
    assert frame.class_id.between(0, 17).all()
    assert np.equal(frame.class_id, np.floor(frame.class_id)).all()
    assert frame.confidence.between(0, 1).all()
    assert ((frame.x2 > frame.x1) & (frame.y2 > frame.y1)).all()
    ids = frame.image_id.astype(str)
    assert set(ids) <= set(geometry)
    width = ids.map(lambda stem: geometry[stem]["original_width"])
    height = ids.map(lambda stem: geometry[stem]["original_height"])
    assert ((frame.x1 >= 0) & (frame.y1 >= 0) & (frame.x2 <= width + 1e-4) & (frame.y2 <= height + 1e-4)).all()
    counts = frame.groupby("image_id").size()
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "rows": len(frame),
            "images_with_predictions": len(counts), "expected_images": len(geometry),
            "images_without_predictions": sorted(set(geometry) - set(ids)),
            "minimum_confidence": float(frame.confidence.min()),
            "maximum_predictions_per_image": int(counts.max()), "format_audit": "passed",
            "remote_byte_identity": "not_verified"}


def main():
    plan_c = ROOT / "Submission/working_backup_plan_C/hsi_plan_c_ratio"
    plan_b = ROOT / "Submission/plan_b_latest/hsi_plan_b_v2"
    a = json.loads((plan_c / "evaluation.json").read_text())
    b = json.loads((plan_b / "evaluation.json").read_text())
    config = json.loads((plan_c / "run_config.json").read_text())
    audit = json.loads((plan_c / "data_audit.json").read_text())
    checkpoint = plan_c / "runs/rtdetr_ratio_control/weights/best.pt"
    checkpoint_hash = digest(checkpoint)
    saved_hash = json.loads((plan_c / "evaluation_final_val.json").read_text())["sha256"]
    assert checkpoint_hash == saved_hash
    assert json.loads((plan_c / "split_manifest.json").read_text()) == json.loads((plan_b / "split_manifest.json").read_text())
    geometry = json.loads((plan_c / "inference_geometry.json").read_text())
    csv_audits = [audit_csv(plan_c / "submission.csv", geometry), audit_csv(ROOT / "Submission/submission_conf0001_max300.csv", geometry)]
    scores = json.loads((RESULTS / "source_snapshots/kaggle_submissions_2026-09-19.json").read_text())
    public = {row["description"]: row["public_score"] for row in scores["submissions"]}
    deltas = {split: {key: a[split][key] - b[split][key] for key in ("map50_95", "map50", "map75", "worst_present_class_ap")} for split in ("val", "test")}
    classes = audit["class_names"]
    per_class = []
    for split in ("val", "test"):
        for cid, row in a[split]["per_class"].items():
            previous = b[split]["per_class_ap"][cid]
            per_class.append({"split": split, "class_id": int(cid), "class_name": classes[int(cid)],
                              "plan_b_AP": previous, "plan_c_AP": row["ap50_95"],
                              "delta_AP": row["ap50_95"] - previous, "plan_c_AP75": row["ap75"]})
    with (RESULTS / "plan_c_class_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)
    invalid_confusion = {split: not np.asarray(a[split].get("confusion_matrix_predicted_rows_true_columns", [])).any() for split in ("val", "test")}
    record = {
        "experiment_id": "plan_c_existing_evidence_audit", "parent_experiment_id": "plan_b_v2_rtdetr_l_16b_50e",
        "code_hash": digest(ROOT / "Submission/working_backup_plan_C/hsi_runner.py"),
        "config_hash": digest(plan_c / "run_config.json"), "checkpoint_hash": checkpoint_hash,
        "split_hash": digest(plan_c / "split_manifest.json"),
        "pretrained_weight_name": config["args"]["model"], "pretrained_weight_source": None,
        "pretrained_dataset": None, "pretrained_license": None,
        "train_runtime": None, "gpu": None, "seed": config["args"]["seed"], "fold": "legacy_single_split",
        "validation_map50_95": a["val"]["map50_95"], "validation_map50": a["val"]["map50"],
        "validation_map75": a["val"]["map75"], "AP_small": None, "AP_medium": None, "AP_large": None,
        "per_class_AP50_95": {cid: row["ap50_95"] for cid, row in a["val"]["per_class"].items()},
        "per_class_AP50": {cid: row["ap50"] for cid, row in a["val"]["per_class"].items()},
        "per_class_AP75": {cid: row["ap75"] for cid, row in a["val"]["per_class"].items()},
        "precision_per_class": {cid: row["precision"] for cid, row in a["val"]["per_class"].items()},
        "recall_per_class": {cid: row["recall"] for cid, row in a["val"]["per_class"].items()},
        "false_positives_per_class": None, "false_negatives_per_class": None, "confusion_matrix": None,
        "worst_present_class": min(a["val"]["per_class"], key=lambda cid: a["val"]["per_class"][cid]["ap50_95"]),
        "worst_present_class_AP": a["val"]["worst_present_class_ap"],
        "localization_gap": a["val"]["map50"] - a["val"]["map75"],
        "holdout_map50_95": a["test"]["map50_95"], "holdout_map75": a["test"]["map75"],
        "public_score": public["RT-DETR-L Plan C, 50 epochs, square-padding preprocessing, random 16-band stem."],
        "public_low_confidence_score": public["RT-DETR-L Plan C : conf=0.001 and max_det=300."],
        "public_plan_d_score": public["Plan D RT-DETR Low Confidence"],
        "status": "historical_evidence_verified_not_cv_complete", "deltas": deltas,
        "observation": "Plan-C improves local AP on the same split; its public scores do not exceed Plan-D. Saved confusion matrices contain only zero entries.",
        "evidence_quality": "saved checkpoint hash verified; saved metrics and split verified; public scores observed in authenticated browser; remote CSV bytes not matched",
        "hypothesis": "square padding helps on the legacy split; generalization across folds is unproven",
        "decision": "retain native and square-pad controls; reopen unconditional geometry promotion; invalidate saved zero confusion counts; do not claim three-fold promotion",
        "smallest_next_test": "V1 fixed folds and D1 original-coordinate diagnostics, then matched G1 geometry runs on all three folds",
        "holdout_status": "previously_inspected_and_used_for_decisions_not_untouched",
        "invalid_saved_confusion": invalid_confusion, "submission_audits": csv_audits,
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in [plan_c / "evaluation.json", plan_b / "evaluation.json", RESULTS / "source_snapshots/kaggle_submissions_2026-09-19.json"]},
        "missing_metadata": ["verified pretrained source/dataset/license", "complete uninterrupted runtime/GPU provenance", "size AP", "valid confusion counts", "three-fold results"],
    }
    append_record(record)
    (RESULTS / "plan_c_audit.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    print(json.dumps({"validation": record["validation_map50_95"], "holdout": record["holdout_map50_95"], "public": record["public_score"], "low_conf_public": record["public_low_confidence_score"], "csv_audits": csv_audits}, indent=2))


if __name__ == "__main__":
    main()
