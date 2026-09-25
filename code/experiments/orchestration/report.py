from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .ledger import sha256_file, verify_ledger
from .queue import blocked_obligations, status_summary
from .ranking import write_rankings


def _table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def write_report(registry, state, ledger_path, output_dir, report_path):
    paths, tables = write_rankings(registry, ledger_path, output_dir)
    audit = verify_ledger(ledger_path, registry["required_ledger_fields"])
    summary = status_summary(state)
    stage_rows = []
    for stage, counts in summary["by_stage"].items():
        stage_rows.append((stage, sum(counts.values()), counts.get("implemented", 0), counts.get("running", 0), counts.get("measured", 0), counts.get("verified", 0), counts.get("stopped", 0), counts.get("planned", 0)))
    done = state["definition_of_done"]
    done_rows = [(identifier, row["status"], row["requirement"]) for identifier, row in sorted(done.items())]
    blocked = blocked_obligations(state)
    blocked_rows = [(row["id"], row["status"], ", ".join(row["missing"])) for row in blocked]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment execution report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This report distinguishes code implementation from measured and verified competition evidence. Empty ranking tables mean the required verified evidence does not exist yet.",
        "",
        "## Obligation status",
        "",
        _table(("Scope", "Total", "Implemented", "Running", "Measured", "Verified", "Stopped", "Planned"), stage_rows),
        "",
        "## Evidence ledger",
        "",
        f"Records: {audit['records']}; chained records: {audit['chained_records']}; legacy records: {audit['legacy_records']}.",
        "",
        "## Rankings",
        "",
        f"Individual rows: {len(tables['individual'])}; combination rows: {len(tables['combination'])}; finalist rows: {len(tables['finalist'])}.",
        "",
        _table(("Table", "Path", "SHA-256"), [(name, str(path), sha256_file(path)) for name, path in paths.items()]),
        "",
        "## Definition of done",
        "",
        _table(("ID", "Status", "Requirement"), done_rows),
        "",
        "## Blocked obligations",
        "",
        _table(("ID", "Status", "Missing verified prerequisites"), blocked_rows) if blocked_rows else "None.",
        "",
        "## Completion statement",
        "",
    ]
    completed = all(row["status"] == "verified" for row in done.values())
    if completed:
        lines.append("All roadmap definition-of-done requirements have verified evidence.")
    else:
        counts = Counter(row["status"] for row in done.values())
        lines.append(f"Roadmap incomplete: {counts.get('verified', 0)} of {len(done)} definition-of-done requirements are verified.")
    report_path.write_text("\n".join(lines) + "\n")
    return {"report": report_path, "rankings": paths, "complete": completed, "ledger": audit}
