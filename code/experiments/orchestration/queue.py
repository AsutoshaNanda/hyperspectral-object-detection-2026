from collections import Counter
import re

from .ranking import stage_candidates
from .state import StateError, bind_selection
from .state import unmet_dependencies


def ready_obligations(state):
    ready = []
    for obligation in state["obligations"].values():
        status = obligation["status"]
        if status == "planned":
            if obligation.get("stage") in ("E1-2", "E1-3", "C1", "C2", "C3", "C4") and not obligation.get("selection_evidence_hashes"):
                continue
            ready.append({"id": obligation["id"], "action": "implement", "stage": obligation.get("stage")})
        elif status in ("implemented", "measured"):
            missing = unmet_dependencies(state, obligation["id"])
            if not missing:
                action = "run" if status == "implemented" else "verify_or_rerun"
                ready.append({"id": obligation["id"], "action": action, "stage": obligation.get("stage")})
    return sorted(ready, key=lambda row: ((row["stage"] or ""), row["id"]))


def blocked_obligations(state):
    blocked = []
    for obligation in state["obligations"].values():
        if obligation["status"] not in ("implemented", "measured"):
            continue
        missing = unmet_dependencies(state, obligation["id"])
        if missing:
            blocked.append({"id": obligation["id"], "status": obligation["status"], "missing": missing})
    return sorted(blocked, key=lambda row: row["id"])


def status_summary(state):
    total = Counter(row["status"] for row in state["obligations"].values())
    by_stage = {}
    for row in state["obligations"].values():
        stage = row.get("stage") or "individual"
        by_stage.setdefault(stage, Counter())[row["status"]] += 1
    return {
        "total": dict(sorted(total.items())),
        "by_stage": {stage: dict(sorted(counts.items())) for stage, counts in sorted(by_stage.items())},
        "definition_of_done": dict(
            sorted(Counter(row["status"] for row in state["definition_of_done"].values()).items())
        ),
    }


PROMOTIONS = {
    ("E1-1", "E1-2"): {"source_stages": ("E1-1",), "count": 3, "family": "E1"},
    ("C0", "C1"): {"source_stages": ("C0",), "count": 15, "kind": "combination"},
    ("C1", "C2"): {"source_stages": ("C0", "C1"), "count": 10, "kind": "combination"},
    ("C2", "C3"): {"source_stages": ("C2",), "count": 3, "kind": "combination", "full_cv": True},
    ("C3", "C4"): {"source_stages": ("C2", "C3"), "count": 1, "kind": "combination", "full_cv": True, "holdout": True},
}


def _terminal_source(state, source_stage):
    rows = [row for row in state["obligations"].values() if row.get("stage") == source_stage]
    pending = [row["id"] for row in rows if row["status"] not in ("verified", "stopped")]
    if pending:
        raise StateError(f"Source stage {source_stage} is incomplete: " + ", ".join(pending))


def promote_stage(state, registry, records, source_stage, destination_stage):
    policy = PROMOTIONS.get((source_stage, destination_stage))
    if not policy:
        raise StateError(f"Unsupported promotion: {source_stage} -> {destination_stage}")
    _terminal_source(state, source_stage)
    experiments = {row["id"]: row for row in registry["experiments"]}
    eligible = set(experiments)
    if policy.get("family"):
        eligible = {identifier for identifier, row in experiments.items() if row["family"] == policy["family"]}
    if policy.get("kind"):
        eligible = {identifier for identifier, row in experiments.items() if row["kind"] == policy["kind"]}
    rows = stage_candidates(registry, records, policy["source_stages"], eligible)
    rows = [row for row in rows if row["cv_mean_map50_95"] is not None]
    if policy.get("full_cv"):
        rows = [row for row in rows if row["fold_count"] == 3]
    if policy.get("holdout"):
        rows = [row for row in rows if row["holdout_map50_95"] is not None]
    if len(rows) < policy["count"]:
        raise StateError(
            f"Promotion requires {policy['count']} measured verified candidates; found {len(rows)}"
        )
    selected = rows[: policy["count"]]
    destination = [row for row in state["obligations"].values() if row.get("stage") == destination_stage]
    if not destination:
        raise StateError(f"No destination obligations for stage {destination_stage}")
    for obligation in destination:
        match = re.search(r"rank(\d+)", obligation["id"])
        rank = int(match.group(1)) if match else 1
        candidate = selected[rank - 1]
        hashes = [value for value in candidate["evidence_records"].split(";") if value]
        bind_selection(state, obligation["id"], candidate["id"], hashes)
    return selected
