import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from orchestration.ledger import (
    CONCLUSION_FIELDS,
    LedgerError,
    append_record,
    canonical_json,
    load_records,
    record_by_hash,
    sha256_file,
    validate_record,
    verify_ledger,
)
from orchestration.queue import blocked_obligations, promote_stage, ready_obligations, status_summary
from orchestration.report import write_report
from orchestration.state import (
    StateError,
    complete_requirement,
    initialize_state,
    load_state,
    save_state,
    set_gate,
    transition,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_REGISTRY = HERE / "registry.json"
DEFAULT_STATE = HERE / "results/experiment_state.json"
DEFAULT_LEDGER = HERE / "results/experiment_ledger.jsonl"
DEFAULT_RESULTS = HERE / "results"


def load_json(path):
    return json.loads(Path(path).read_text())


def hash_paths(paths):
    digest = hashlib.sha256()
    for path in sorted(Path(value).resolve() for value in paths):
        if not path.is_file():
            raise StateError(f"Code path is not a file: {path}")
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def config_hash(config):
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def lifecycle_record(registry, obligation, status, decision, **fields):
    record = {field: None for field in registry["required_ledger_fields"]}
    record.update(
        {
            "experiment_id": obligation.get("bound_experiment_id") or obligation.get("experiment_id") or obligation["id"],
            "parent_experiment_id": None,
            "status": status,
            "decision": decision,
            "obligation_id": obligation["id"],
            "stage": obligation.get("stage"),
        }
    )
    record.update(fields)
    return record


def require_verified_hashes(ledger_path, hashes):
    records = {row.get("record_hash"): row for row in load_records(ledger_path)}
    missing = [value for value in hashes if value not in records]
    if missing:
        raise StateError("Unknown evidence hashes: " + ", ".join(missing))
    unverified = [value for value in hashes if records[value].get("status") != "verified"]
    if unverified:
        raise StateError("Evidence is not verified: " + ", ".join(unverified))


def validate_config(config, obligation):
    if config.get("obligation_id") != obligation["id"]:
        raise StateError("Configuration obligation_id does not match")
    expected = obligation.get("bound_experiment_id") or obligation.get("experiment_id") or obligation["id"]
    if config.get("experiment_id") != expected:
        raise StateError(f"Configuration experiment_id must be {expected!r}")
    if not isinstance(config.get("code_paths"), list) or not config["code_paths"]:
        raise StateError("Configuration requires non-empty code_paths")
    command = config.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
        raise StateError("Configuration command must be a non-empty string list")
    if not config.get("result_path"):
        raise StateError("Configuration requires result_path")
    return config


def append_lifecycle(ledger_path, registry, obligation, status, decision, **fields):
    return append_record(
        ledger_path,
        lifecycle_record(registry, obligation, status, decision, **fields),
        registry["required_ledger_fields"],
    )


def preflight_transition(state, obligation_id, target, **fields):
    trial = copy.deepcopy(state)
    transition(trial, obligation_id, target, **fields)


def cmd_init(args, registry):
    if args.state.exists() and not args.force:
        raise StateError(f"State already exists: {args.state}")
    state = initialize_state(registry)
    if args.state.exists():
        existing = load_json(args.state)
        progressed = [
            row["id"]
            for row in existing.get("obligations", {}).values()
            if row.get("status") != "planned"
        ]
        completed = [
            row["id"]
            for row in existing.get("definition_of_done", {}).values()
            if row.get("status") != "planned"
        ]
        if progressed or completed:
            raise StateError("Refusing to replace state containing execution or completion progress")
        state["revision"] = int(existing.get("revision", 0))
    save_state(args.state, state)
    print(json.dumps(status_summary(state), indent=2))


def cmd_implement(args, registry, state):
    obligation = state["obligations"].get(args.obligation_id)
    if not obligation:
        raise StateError(f"Unknown obligation: {args.obligation_id}")
    config = validate_config(load_json(args.config), obligation)
    digest = config_hash(config)
    code_digest = hash_paths(config["code_paths"])
    preflight_transition(
        state,
        obligation["id"],
        "implemented",
        record_hash="pending",
        config_hash=digest,
        code_hash=code_digest,
    )
    record = append_lifecycle(
        args.ledger,
        registry,
        obligation,
        "implemented",
        "Configuration and code paths recorded; no result claimed.",
        code_hash=code_digest,
        config_hash=digest,
        config_path=str(args.config),
    )
    transition(
        state,
        obligation["id"],
        "implemented",
        record["record_hash"],
        config_hash=digest,
        code_hash=code_digest,
    )
    save_state(args.state, state)
    print(json.dumps(record, indent=2))


def cmd_start(args, registry, state):
    obligation = state["obligations"].get(args.obligation_id)
    if not obligation:
        raise StateError(f"Unknown obligation: {args.obligation_id}")
    preflight_transition(state, obligation["id"], "running", record_hash="pending")
    record = append_lifecycle(
        args.ledger,
        registry,
        obligation,
        "running",
        "Run started; no measurement claimed.",
        code_hash=obligation.get("code_hash"),
        config_hash=obligation.get("config_hash"),
    )
    transition(state, obligation["id"], "running", record["record_hash"])
    save_state(args.state, state)
    return record


def enrich_measurement(record, obligation, config=None, result_path=None):
    value = dict(record)
    expected = obligation.get("bound_experiment_id") or obligation.get("experiment_id") or obligation["id"]
    value["experiment_id"] = expected
    value["obligation_id"] = obligation["id"]
    value["stage"] = obligation.get("stage")
    value["status"] = "measured"
    if config:
        if not value.get("config_hash"):
            value["config_hash"] = config_hash(config)
        if not value.get("code_hash"):
            value["code_hash"] = hash_paths(config["code_paths"])
        split_path = config.get("split_path")
        if split_path and not value.get("split_hash"):
            value["split_hash"] = sha256_file(split_path)
        if value.get("fold") is None:
            value["fold"] = config.get("fold", obligation.get("fold"))
    if result_path and not value.get("evidence_source"):
        value["evidence_source"] = str(result_path)
    for field in CONCLUSION_FIELDS:
        value.setdefault(field, None)
    return value


def measure(args, registry, state, config=None):
    obligation = state["obligations"].get(args.obligation_id)
    if not obligation:
        raise StateError(f"Unknown obligation: {args.obligation_id}")
    if obligation["status"] != "running":
        raise StateError("Measurement can only complete a running obligation")
    raw = load_json(args.record)
    value = enrich_measurement(raw, obligation, config=config, result_path=args.record)
    validate_record(value, registry["required_ledger_fields"], value["experiment_id"])
    if value["config_hash"] != obligation.get("config_hash"):
        raise LedgerError("Measured config_hash differs from the implemented configuration")
    if value["code_hash"] != obligation.get("code_hash"):
        raise LedgerError("Measured code_hash differs from the implemented code")
    preflight_transition(state, obligation["id"], "measured", record_hash="pending")
    appended = append_record(args.ledger, value, registry["required_ledger_fields"])
    transition(state, obligation["id"], "measured", appended["record_hash"])
    save_state(args.state, state)
    print(json.dumps(appended, indent=2))
    return appended


def cmd_verify(args, registry, state):
    obligation = state["obligations"].get(args.obligation_id)
    if not obligation:
        raise StateError(f"Unknown obligation: {args.obligation_id}")
    measured_hash = obligation.get("measurement_record_hash")
    if not measured_hash:
        raise StateError("No measured evidence exists for this obligation")
    measured = record_by_hash(args.ledger, measured_hash)
    verified = {
        key: value
        for key, value in measured.items()
        if key not in ("record_hash", "recorded_at", "previous_line_hash", "event_schema_version")
    }
    verified["status"] = "verified"
    verified["verified_from_record_hash"] = measured_hash
    if args.decision:
        verified["decision"] = args.decision
    preflight_transition(state, obligation["id"], "verified", record_hash="pending")
    appended = append_record(args.ledger, verified, registry["required_ledger_fields"])
    transition(state, obligation["id"], "verified", appended["record_hash"])
    save_state(args.state, state)
    print(json.dumps(appended, indent=2))


def cmd_stop(args, registry, state):
    obligation = state["obligations"].get(args.obligation_id)
    if not obligation:
        raise StateError(f"Unknown obligation: {args.obligation_id}")
    raw = load_json(args.record)
    value = lifecycle_record(
        registry,
        obligation,
        "stopped",
        raw.get("decision"),
        **{key: item for key, item in raw.items() if key not in registry["required_ledger_fields"]},
    )
    for field in registry["required_ledger_fields"]:
        if field in raw:
            value[field] = raw[field]
    value["experiment_id"] = obligation.get("bound_experiment_id") or obligation.get("experiment_id") or obligation["id"]
    value["status"] = "stopped"
    value["stop_reason"] = args.reason
    value.setdefault("evidence_source", str(args.record))
    preflight_transition(
        state,
        obligation["id"],
        "stopped",
        record_hash="pending",
        stop_reason=args.reason,
    )
    appended = append_record(args.ledger, value, registry["required_ledger_fields"])
    transition(state, obligation["id"], "stopped", appended["record_hash"], stop_reason=args.reason)
    save_state(args.state, state)
    print(json.dumps(appended, indent=2))


def record_failure(args, registry, state, obligation, message, log_path):
    preflight_transition(
        state,
        obligation["id"],
        "implemented",
        record_hash="pending",
        error=message,
    )
    record = append_lifecycle(
        args.ledger,
        registry,
        obligation,
        "implemented",
        "Run failed before valid measurement; retain for diagnosis and retry.",
        code_hash=obligation.get("code_hash"),
        config_hash=obligation.get("config_hash"),
        error=message,
        failure_log=str(log_path),
    )
    transition(state, obligation["id"], "implemented", record["record_hash"], error=message)
    save_state(args.state, state)


def cmd_run(args, registry, state):
    obligation = state["obligations"].get(args.obligation_id)
    if not obligation:
        raise StateError(f"Unknown obligation: {args.obligation_id}")
    config_path = Path(args.config)
    config = validate_config(load_json(config_path), obligation)
    digest = config_hash(config)
    if obligation["status"] == "planned":
        implementation_args = argparse.Namespace(
            obligation_id=args.obligation_id,
            config=config_path,
            ledger=args.ledger,
            state=args.state,
        )
        cmd_implement(implementation_args, registry, state)
    if obligation.get("config_hash") != digest:
        raise StateError("Run configuration differs from implemented configuration")
    cmd_start(args, registry, state)
    workdir = Path(config.get("working_directory", ROOT))
    log_path = Path(config.get("log_path", DEFAULT_RESULTS / "logs" / f"{args.obligation_id}.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(config["command"], cwd=workdir, text=True, capture_output=True)
    log_path.write_text(result.stdout + result.stderr)
    fresh = load_state(args.state, registry)
    state.clear()
    state.update(fresh)
    obligation = state["obligations"][args.obligation_id]
    if result.returncode:
        message = f"Command exited with status {result.returncode}"
        record_failure(args, registry, state, obligation, message, log_path)
        raise StateError(message)
    result_path = Path(config["result_path"])
    if not result_path.is_absolute():
        result_path = workdir / result_path
    if not result_path.is_file():
        message = f"Successful command did not produce result record: {result_path}"
        record_failure(args, registry, state, obligation, message, log_path)
        raise StateError(message)
    measure_args = argparse.Namespace(
        obligation_id=args.obligation_id,
        record=result_path,
        ledger=args.ledger,
        state=args.state,
    )
    measure(measure_args, registry, state, config=config)


def cmd_retry(args, registry, state):
    obligation = state["obligations"].get(args.obligation_id)
    if not obligation:
        raise StateError(f"Unknown obligation: {args.obligation_id}")
    failure = load_json(args.record)
    message = failure.get("error") or failure.get("observation")
    if not message:
        raise StateError("Retry record requires error or observation")
    preflight_transition(
        state,
        obligation["id"],
        "implemented",
        record_hash="pending",
        error=message,
    )
    record = append_lifecycle(
        args.ledger,
        registry,
        obligation,
        "implemented",
        failure.get("decision", "Transient run failure recorded; obligation returned to the runnable queue."),
        code_hash=obligation.get("code_hash"),
        config_hash=obligation.get("config_hash"),
        error=message,
        evidence_source=str(args.record),
    )
    transition(state, obligation["id"], "implemented", record["record_hash"], error=message)
    save_state(args.state, state)
    print(json.dumps(record, indent=2))


def cmd_gate(args, state):
    known_external = {
        dependency
        for obligation in state["obligations"].values()
        for dependency in obligation["prerequisites"]
        if dependency not in state["obligations"]
    }
    if args.gate_id not in known_external:
        raise StateError(f"Gate is not a registry prerequisite: {args.gate_id}")
    require_verified_hashes(args.ledger, args.evidence_hash)
    set_gate(state, args.gate_id, args.evidence_hash)
    save_state(args.state, state)


def cmd_complete(args, state):
    require_verified_hashes(args.ledger, args.evidence_hash)
    complete_requirement(state, args.requirement_id, args.evidence_hash)
    save_state(args.state, state)


def cmd_promote(args, registry, state):
    selected = promote_stage(
        state,
        registry,
        load_records(args.ledger),
        args.source_stage,
        args.destination_stage,
    )
    save_state(args.state, state)
    print(json.dumps(selected, indent=2))


def cmd_report(args, registry, state):
    result = write_report(registry, state, args.ledger, args.output_dir, args.report_path)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--force", action="store_true")
    subparsers.add_parser("status")
    subparsers.add_parser("ready")
    subparsers.add_parser("blocked")
    implement = subparsers.add_parser("implement")
    implement.add_argument("obligation_id")
    implement.add_argument("--config", type=Path, required=True)
    start = subparsers.add_parser("start")
    start.add_argument("obligation_id")
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("obligation_id")
    measure_parser.add_argument("--record", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("obligation_id")
    verify.add_argument("--decision")
    stop = subparsers.add_parser("stop")
    stop.add_argument("obligation_id")
    stop.add_argument("--reason", required=True)
    stop.add_argument("--record", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("obligation_id")
    run.add_argument("--config", type=Path, required=True)
    retry = subparsers.add_parser("retry")
    retry.add_argument("obligation_id")
    retry.add_argument("--record", type=Path, required=True)
    gate = subparsers.add_parser("gate")
    gate.add_argument("gate_id")
    gate.add_argument("--evidence-hash", action="append", required=True)
    complete = subparsers.add_parser("complete")
    complete.add_argument("requirement_id")
    complete.add_argument("--evidence-hash", action="append", required=True)
    promote = subparsers.add_parser("promote-stage")
    promote.add_argument("source_stage")
    promote.add_argument("destination_stage")
    audit = subparsers.add_parser("audit-ledger")
    report = subparsers.add_parser("report")
    report.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    report.add_argument("--report-path", type=Path, default=DEFAULT_RESULTS / "FINAL_REPORT.md")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    registry = load_json(args.registry)
    if args.command == "init":
        cmd_init(args, registry)
        return
    if args.command == "audit-ledger":
        print(json.dumps(verify_ledger(args.ledger, registry["required_ledger_fields"]), indent=2))
        return
    state = load_state(args.state, registry)
    if args.command == "status":
        print(json.dumps(status_summary(state), indent=2))
    elif args.command == "ready":
        print(json.dumps(ready_obligations(state), indent=2))
    elif args.command == "blocked":
        print(json.dumps(blocked_obligations(state), indent=2))
    elif args.command == "implement":
        cmd_implement(args, registry, state)
    elif args.command == "start":
        print(json.dumps(cmd_start(args, registry, state), indent=2))
    elif args.command == "measure":
        measure(args, registry, state)
    elif args.command == "verify":
        cmd_verify(args, registry, state)
    elif args.command == "stop":
        cmd_stop(args, registry, state)
    elif args.command == "run":
        cmd_run(args, registry, state)
    elif args.command == "retry":
        cmd_retry(args, registry, state)
    elif args.command == "gate":
        cmd_gate(args, state)
    elif args.command == "complete":
        cmd_complete(args, state)
    elif args.command == "promote-stage":
        cmd_promote(args, registry, state)
    elif args.command == "report":
        cmd_report(args, registry, state)


if __name__ == "__main__":
    try:
        main()
    except (StateError, LedgerError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
