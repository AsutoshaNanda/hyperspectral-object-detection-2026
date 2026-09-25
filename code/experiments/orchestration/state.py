import hashlib
import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .ledger import LIFECYCLE_STATUSES, STOP_REASONS


TRANSITIONS = {
    "planned": {"implemented", "stopped"},
    "implemented": {"running", "stopped"},
    "running": {"implemented", "measured", "stopped"},
    "measured": {"running", "verified", "stopped"},
    "verified": set(),
    "stopped": set(),
}


class StateError(ValueError):
    pass


def registry_hash(registry):
    encoded = json.dumps(registry, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def initialize_state(registry):
    obligations = {}
    for experiment in registry["experiments"]:
        obligations[experiment["id"]] = {
            "id": experiment["id"],
            "obligation_type": "experiment",
            "experiment_id": experiment["id"],
            "stage": None,
            "status": "planned",
            "prerequisites": list(experiment["prerequisites"]),
            "bound_experiment_id": experiment["id"],
            "config_hash": None,
            "code_hash": None,
            "implementation_record_hash": None,
            "measurement_record_hash": None,
            "verification_record_hash": None,
            "stop_record_hash": None,
            "attempts": 0,
            "last_error": None,
        }
    for requirement in registry["execution_requirements"]:
        if requirement["id"] in obligations:
            raise StateError(f"Duplicate obligation ID: {requirement['id']}")
        obligations[requirement["id"]] = {
            "id": requirement["id"],
            "obligation_type": "execution_requirement",
            "experiment_id": requirement.get("experiment_id"),
            "stage": requirement["stage"],
            "status": "planned",
            "prerequisites": list(requirement["prerequisites"]),
            "bound_experiment_id": requirement.get("experiment_id"),
            "selection": requirement.get("selection"),
            "fold": requirement.get("fold"),
            "config_hash": None,
            "code_hash": None,
            "implementation_record_hash": None,
            "measurement_record_hash": None,
            "verification_record_hash": None,
            "stop_record_hash": None,
            "selection_evidence_hashes": [],
            "attempts": 0,
            "last_error": None,
        }
    completion = {
        row["id"]: {
            "id": row["id"],
            "requirement": row["requirement"],
            "status": "planned",
            "evidence_record_hashes": [],
        }
        for row in registry["definition_of_done"]
    }
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "registry_hash": registry_hash(registry),
        "registry_source_hash": registry["source"]["sha256"],
        "created_at": now,
        "updated_at": now,
        "revision": 0,
        "obligations": obligations,
        "gates": {},
        "definition_of_done": completion,
    }


def load_state(path, registry=None):
    path = Path(path)
    if not path.exists():
        if registry is None:
            raise StateError(f"State does not exist: {path}")
        return initialize_state(registry)
    state = json.loads(path.read_text())
    if registry is not None and state.get("registry_hash") != registry_hash(registry):
        raise StateError("State registry hash does not match registry.json")
    return state


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        expected_revision = int(state.get("revision", 0))
        if path.exists():
            current_revision = int(json.loads(path.read_text()).get("revision", 0))
            if current_revision != expected_revision:
                raise StateError(
                    f"State changed concurrently: expected revision {expected_revision}, found {current_revision}"
                )
        value = dict(state)
        value["revision"] = expected_revision + 1
        value["updated_at"] = datetime.now(timezone.utc).isoformat()
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        state.clear()
        state.update(value)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def resolve_dependency(state, dependency):
    if dependency in state["obligations"]:
        return state["obligations"][dependency]["status"] == "verified"
    return state["gates"].get(dependency, {}).get("status") == "verified"


def unmet_dependencies(state, obligation_id):
    obligation = state["obligations"].get(obligation_id)
    if obligation is None:
        raise StateError(f"Unknown obligation: {obligation_id}")
    return [dependency for dependency in obligation["prerequisites"] if not resolve_dependency(state, dependency)]


def transition(state, obligation_id, target, record_hash=None, config_hash=None, code_hash=None, stop_reason=None, error=None):
    if target not in LIFECYCLE_STATUSES:
        raise StateError(f"Unknown target status: {target}")
    obligation = state["obligations"].get(obligation_id)
    if obligation is None:
        raise StateError(f"Unknown obligation: {obligation_id}")
    current = obligation["status"]
    if target not in TRANSITIONS[current]:
        raise StateError(f"Invalid transition: {current} -> {target}")
    if target == "running":
        missing = unmet_dependencies(state, obligation_id)
        if missing:
            raise StateError("Unmet prerequisites: " + ", ".join(missing))
        if not obligation.get("config_hash") and not config_hash:
            raise StateError("Running requires an implemented configuration hash")
        obligation["attempts"] += 1
    if target == "implemented":
        if current == "planned" and not config_hash:
            raise StateError("Implementation requires a configuration hash")
        if current == "running" and not error:
            raise StateError("Returning a run to implemented requires recorded failure evidence")
        if config_hash:
            obligation["config_hash"] = config_hash
        if code_hash:
            obligation["code_hash"] = code_hash
        if record_hash:
            obligation["implementation_record_hash"] = record_hash
        obligation["last_error"] = error
    if target == "measured":
        if not record_hash:
            raise StateError("Measured status requires a validated ledger record")
        obligation["measurement_record_hash"] = record_hash
    if target == "verified":
        if not obligation.get("measurement_record_hash"):
            raise StateError("Verification requires measured evidence")
        if not record_hash:
            raise StateError("Verification requires a verification ledger record")
        obligation["verification_record_hash"] = record_hash
    if target == "stopped":
        if stop_reason not in STOP_REASONS:
            raise StateError("Stopped status requires a roadmap section 38 reason")
        if not record_hash:
            raise StateError("Stopped status requires a validated ledger record")
        obligation["stop_record_hash"] = record_hash
        obligation["stop_reason"] = stop_reason
    obligation["status"] = target
    return obligation


def set_gate(state, gate_id, evidence_record_hashes):
    if not evidence_record_hashes:
        raise StateError("A gate requires verified evidence record hashes")
    state["gates"][gate_id] = {
        "status": "verified",
        "evidence_record_hashes": sorted(set(evidence_record_hashes)),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def bind_selection(state, obligation_id, experiment_id, evidence_record_hashes):
    obligation = state["obligations"].get(obligation_id)
    if not obligation or obligation["obligation_type"] != "execution_requirement":
        raise StateError(f"Not a staged execution obligation: {obligation_id}")
    if obligation["status"] != "planned":
        raise StateError("Selection can only bind a planned obligation")
    if obligation.get("experiment_id") and obligation["experiment_id"] != experiment_id:
        raise StateError("Cannot replace a registry-fixed experiment selection")
    if experiment_id not in state["obligations"]:
        raise StateError(f"Unknown selected experiment: {experiment_id}")
    if not evidence_record_hashes:
        raise StateError("Selection requires measured ranking evidence")
    obligation["bound_experiment_id"] = experiment_id
    obligation["selection_evidence_hashes"] = sorted(set(evidence_record_hashes))


def complete_requirement(state, requirement_id, evidence_record_hashes):
    requirement = state["definition_of_done"].get(requirement_id)
    if not requirement:
        raise StateError(f"Unknown definition-of-done requirement: {requirement_id}")
    if not evidence_record_hashes:
        raise StateError("Completion requires verified evidence records")
    requirement["status"] = "verified"
    requirement["evidence_record_hashes"] = sorted(set(evidence_record_hashes))
