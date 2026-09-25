import hashlib
import fcntl
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


LIFECYCLE_STATUSES = ("planned", "implemented", "running", "measured", "verified", "stopped")
CONCLUSION_FIELDS = (
    "observation",
    "evidence_source",
    "controlled_change",
    "cv_result",
    "worst_fold",
    "map75",
    "ap_small",
    "worst_class",
    "material_pair_result",
    "decision",
    "next_experiment",
)
STOP_REASONS = (
    "invalid_predictions",
    "resource_impossible",
    "organizer_rule_violation",
    "architecture_incompatible",
    "prerequisite_diagnostic_rejected",
)


class LedgerError(ValueError):
    pass


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_finite(value, location="record"):
    if isinstance(value, float) and not math.isfinite(value):
        raise LedgerError(f"Non-finite number at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{location}.{key}")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{location}[{index}]")


def load_records(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise LedgerError(f"Invalid JSON on ledger line {line_number}: {error}") from error
    return records


def validate_record(record, required_fields, expected_experiment_id=None):
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise LedgerError("Missing required ledger fields: " + ", ".join(missing))
    if record["status"] not in LIFECYCLE_STATUSES:
        raise LedgerError(f"Unknown lifecycle status: {record['status']}")
    if expected_experiment_id and record["experiment_id"] != expected_experiment_id:
        raise LedgerError(
            f"Record experiment_id {record['experiment_id']!r} does not match {expected_experiment_id!r}"
        )
    _assert_finite(record)
    status = record["status"]
    if status in ("measured", "verified"):
        for field in ("code_hash", "config_hash", "split_hash", "fold"):
            if record.get(field) in (None, ""):
                raise LedgerError(f"{field} is required for {status} evidence")
        missing_conclusion = [field for field in CONCLUSION_FIELDS if field not in record]
        if missing_conclusion:
            raise LedgerError("Missing evidence-rule fields: " + ", ".join(missing_conclusion))
        if not record.get("observation") or not record.get("evidence_source") or not record.get("decision"):
            raise LedgerError("Measured evidence requires observation, evidence_source and decision")
        metric_present = any(
            record.get(field) is not None
            for field in (
                "validation_map50_95",
                "validation_map50",
                "validation_map75",
                "AP_small",
                "AP_medium",
                "AP_large",
                "holdout_map50_95",
                "holdout_map75",
            )
        )
        diagnostic_present = bool(record.get("measurements")) or bool(record.get("artifacts"))
        if not metric_present and not diagnostic_present:
            raise LedgerError("Measured evidence requires a metric, measurements, or artifacts")
    if status == "stopped":
        if record.get("stop_reason") not in STOP_REASONS:
            raise LedgerError("Stopped evidence must use a roadmap section 38 stop reason")
        if not record.get("evidence_source") or not record.get("observation") or not record.get("decision"):
            raise LedgerError("Stopped evidence requires an evidence source, observation and decision")
    return record


def _last_raw_hash(content):
    lines = [line for line in content.splitlines() if line.strip()]
    return sha256_bytes(lines[-1]) if lines else None


def append_record(path, record, required_fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        content = b""
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            content += block
        value = dict(record)
        value.setdefault("event_schema_version", 1)
        value.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        value["previous_line_hash"] = _last_raw_hash(content)
        value.pop("record_hash", None)
        validate_record(value, required_fields)
        value["record_hash"] = sha256_bytes(canonical_json(value).encode("utf-8"))
        encoded = canonical_json(value) + "\n"
        os.lseek(descriptor, 0, os.SEEK_END)
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return value


def verify_ledger(path, required_fields):
    path = Path(path)
    if not path.exists():
        return {"records": 0, "chained_records": 0, "legacy_records": 0}
    previous_raw_hash = None
    chained = 0
    legacy = 0
    records = 0
    for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
        if not raw.strip():
            continue
        records += 1
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise LedgerError(f"Invalid JSON on ledger line {line_number}: {error}") from error
        if record.get("event_schema_version") != 1:
            legacy += 1
            previous_raw_hash = sha256_bytes(raw)
            continue
        validate_record(record, required_fields)
        expected_previous = record.get("previous_line_hash")
        if expected_previous != previous_raw_hash:
            raise LedgerError(f"Broken ledger chain on line {line_number}")
        claimed_hash = record.get("record_hash")
        payload = dict(record)
        payload.pop("record_hash", None)
        actual_hash = sha256_bytes(canonical_json(payload).encode("utf-8"))
        if claimed_hash != actual_hash:
            raise LedgerError(f"Invalid record hash on line {line_number}")
        chained += 1
        previous_raw_hash = sha256_bytes(raw)
    return {"records": records, "chained_records": chained, "legacy_records": legacy}


def record_by_hash(path, record_hash):
    for record in load_records(path):
        if record.get("record_hash") == record_hash:
            return record
    raise LedgerError(f"Unknown ledger record hash: {record_hash}")
