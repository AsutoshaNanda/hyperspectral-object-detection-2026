import csv
import statistics
from collections import defaultdict
from pathlib import Path

from .ledger import load_records


CSV_FIELDS = (
    "rank",
    "id",
    "verified_runs",
    "fold_count",
    "cv_mean_map50_95",
    "cv_std_map50_95",
    "worst_fold_map50_95",
    "mean_map75",
    "AP_small",
    "worst_class_AP",
    "holdout_map50_95",
    "difficult_pair_confusion",
    "train_runtime",
    "gpu",
    "decision",
    "evidence_records",
)


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _minimum(values):
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _fold_key(value):
    if isinstance(value, int) and value in (0, 1, 2):
        return value
    if isinstance(value, str) and value in ("0", "1", "2"):
        return int(value)
    return None


def aggregate(records):
    grouped = defaultdict(list)
    for record in records:
        if record.get("event_schema_version") == 1 and record.get("status") == "verified":
            grouped[record["experiment_id"]].append(record)
    rows = []
    for experiment_id, items in grouped.items():
        cv_by_fold = {}
        for item in items:
            fold = _fold_key(item.get("fold"))
            if fold is not None and item.get("validation_map50_95") is not None:
                cv_by_fold[fold] = item
        cv_items = [cv_by_fold[fold] for fold in sorted(cv_by_fold)]
        cv_values = [item["validation_map50_95"] for item in cv_items]
        map75 = [item.get("validation_map75") for item in cv_items]
        ap_small = [item.get("AP_small") for item in cv_items]
        worst_class = [item.get("worst_present_class_AP") for item in cv_items]
        holdout = [item.get("holdout_map50_95") for item in items]
        pair_confusion = [item.get("difficult_pair_confusion") for item in items]
        runtimes = [item.get("train_runtime") for item in items if isinstance(item.get("train_runtime"), (int, float))]
        rows.append(
            {
                "id": experiment_id,
                "verified_runs": len(items),
                "fold_count": len(cv_items),
                "cv_mean_map50_95": _mean(cv_values),
                "cv_std_map50_95": statistics.pstdev(cv_values) if len(cv_values) > 1 else (0.0 if cv_values else None),
                "worst_fold_map50_95": _minimum(cv_values),
                "mean_map75": _mean(map75),
                "AP_small": _mean(ap_small),
                "worst_class_AP": _minimum(worst_class),
                "holdout_map50_95": _mean(holdout),
                "difficult_pair_confusion": _mean(pair_confusion),
                "train_runtime": sum(runtimes) if runtimes else None,
                "gpu": "; ".join(sorted({str(item["gpu"]) for item in items if item.get("gpu")})),
                "decision": items[-1].get("decision"),
                "evidence_records": ";".join(item["record_hash"] for item in items),
            }
        )
    return rows


def _descending(value):
    return -(value if value is not None else float("-inf"))


def individual_sort_key(row):
    return (
        _descending(row["cv_mean_map50_95"]),
        _descending(row["worst_fold_map50_95"]),
        _descending(row["mean_map75"]),
        _descending(row["worst_class_AP"]),
        _descending(row["AP_small"]),
        row["id"],
    )


def finalist_sort_key(row):
    runtime = row["train_runtime"] if row["train_runtime"] is not None else float("inf")
    return (
        _descending(row["cv_mean_map50_95"]),
        _descending(row["worst_fold_map50_95"]),
        _descending(row["holdout_map50_95"]),
        _descending(row["mean_map75"]),
        _descending(row["worst_class_AP"]),
        _descending(row["AP_small"]),
        row["difficult_pair_confusion"] if row["difficult_pair_confusion"] is not None else float("inf"),
        runtime,
        row["id"],
    )


def build_rankings(registry, records):
    kinds = {row["id"]: row["kind"] for row in registry["experiments"]}
    rows = aggregate(records)
    individual = [row for row in rows if kinds.get(row["id"]) != "combination" and row["fold_count"] == 3]
    combination = [row for row in rows if kinds.get(row["id"]) == "combination"]
    finalist = [
        row
        for row in combination
        if row["fold_count"] == 3 and row["holdout_map50_95"] is not None
    ]
    individual.sort(key=individual_sort_key)
    combination.sort(key=individual_sort_key)
    finalist.sort(key=finalist_sort_key)
    for table in (individual, combination, finalist):
        for rank, row in enumerate(table, 1):
            row["rank"] = rank
    return {"individual": individual, "combination": combination, "finalist": finalist}


def write_rankings(registry, ledger_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_rankings(registry, load_records(ledger_path))
    paths = {}
    for name, rows in tables.items():
        path = output_dir / f"{name}_results.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows({field: row.get(field) for field in CSV_FIELDS} for row in rows)
        paths[name] = path
    return paths, tables


def stage_candidates(registry, records, source_stages, eligible_ids=None):
    source_stages = set(source_stages)
    selected = [
        record
        for record in records
        if record.get("event_schema_version") == 1
        and record.get("status") == "verified"
        and record.get("stage") in source_stages
        and (eligible_ids is None or record.get("experiment_id") in eligible_ids)
    ]
    grouped = defaultdict(list)
    for record in selected:
        grouped[record["experiment_id"]].append(record)
    rows = []
    for experiment_id, items in grouped.items():
        values = [item.get("validation_map50_95") for item in items if item.get("validation_map50_95") is not None]
        map75 = [item.get("validation_map75") for item in items]
        ap_small = [item.get("AP_small") for item in items]
        worst_class = [item.get("worst_present_class_AP") for item in items]
        holdout = [item.get("holdout_map50_95") for item in items]
        rows.append(
            {
                "id": experiment_id,
                "verified_runs": len(items),
                "fold_count": len({_fold_key(item.get("fold")) for item in items if _fold_key(item.get("fold")) is not None}),
                "cv_mean_map50_95": _mean(values),
                "cv_std_map50_95": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),
                "worst_fold_map50_95": _minimum(values),
                "mean_map75": _mean(map75),
                "AP_small": _mean(ap_small),
                "worst_class_AP": _minimum(worst_class),
                "holdout_map50_95": _mean(holdout),
                "difficult_pair_confusion": _mean([item.get("difficult_pair_confusion") for item in items]),
                "train_runtime": _mean([item.get("train_runtime") for item in items]),
                "gpu": "; ".join(sorted({str(item["gpu"]) for item in items if item.get("gpu")})),
                "decision": items[-1].get("decision"),
                "evidence_records": ";".join(item["record_hash"] for item in items),
            }
        )
    rows.sort(key=individual_sort_key)
    return rows
