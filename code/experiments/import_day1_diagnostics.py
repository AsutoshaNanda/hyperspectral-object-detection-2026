from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "Experiments/results/kaggle/day1_v2/roadmap_day1"
STATE = ROOT / "Experiments/results/experiment_state.json"
REGISTRY = ROOT / "Experiments/registry.json"
OUTPUT = ROOT / "Experiments/results/imports"


JOBS = {
    "D1-YOLO26m": {
        "report": "yolo_plan_a_size_ap.json",
        "checkpoint": "52800d6c16b33155e34d6e53ecdb31bb009ed0e1356d1777ca67b29d0509e291",
        "weight": "yolo26m.pt",
        "public": 0.58967,
        "model": "YOLO26m",
        "comparison": None,
    },
    "D1-RT-DETR-L": {
        "report": "rtdetr_plan_c_size_ap.json",
        "checkpoint": "b4a54b0f5ad05a25fae96daf070e53286852f15c7091dd3fa13af2113afb9012",
        "weight": "rtdetr-l.pt",
        "public": 0.60691,
        "model": "RT-DETR-L square-pad",
        "comparison": "rtdetr_plan_b_size_ap.json",
    },
}


def metric_map(class_rows, key):
    return {str(row["category_id"]): row["sizes"]["all"][key] for row in class_rows}


def result(identifier, details, state, fields):
    report = json.loads((REPORTS / details["report"]).read_text())
    strict = report["roadmap_strict"]
    sizes = strict["size_metrics"]
    class_rows = strict["class_by_size"]
    weakest = min(class_rows, key=lambda row: row["sizes"]["all"]["AP50_95"] if row["sizes"]["all"]["AP50_95"] is not None else 2)
    confusion = report["confusion_at_fixed_diagnostic_threshold"]
    obligation = state["obligations"][identifier]
    record = {field: None for field in fields}
    record.update(
        experiment_id=identifier,
        parent_experiment_id=None,
        code_hash=obligation["code_hash"],
        config_hash=obligation["config_hash"],
        checkpoint_hash=details["checkpoint"],
        split_hash=report["provenance"]["split_hash"],
        pretrained_weight_name=details["weight"],
        train_runtime=None,
        gpu=report["provenance"]["gpu"],
        seed=42,
        fold="legacy_validation_diagnostic",
        validation_map50_95=sizes["all"]["AP50_95"],
        validation_map50=sizes["all"]["AP50"],
        validation_map75=sizes["all"]["AP75"],
        AP_small=sizes["small"]["AP50_95"],
        AP_medium=sizes["medium"]["AP50_95"],
        AP_large=sizes["large"]["AP50_95"],
        per_class_AP50_95=metric_map(class_rows, "AP50_95"),
        per_class_AP50=metric_map(class_rows, "AP50"),
        per_class_AP75=metric_map(class_rows, "AP75"),
        precision_per_class={str(row["category_id"]): row["precision"] for row in confusion["per_class"]},
        recall_per_class={str(row["category_id"]): row["recall"] for row in confusion["per_class"]},
        false_positives_per_class={str(row["category_id"]): row["false_positives"] for row in confusion["per_class"]},
        false_negatives_per_class={str(row["category_id"]): row["false_negatives"] for row in confusion["per_class"]},
        confusion_matrix=confusion["matrix"],
        worst_present_class={"id": weakest["category_id"], "name": weakest["category_name"]},
        worst_present_class_AP=weakest["sizes"]["all"]["AP50_95"],
        localization_gap=sizes["all"]["AP50"] - sizes["all"]["AP75"],
        public_score=details["public"],
        status="measured",
        decision="Use measured size support to prioritize higher-resolution and sliced-inference tests after fixed-fold baselines; do not promote from this legacy split alone.",
    )
    comparison = None
    if details["comparison"]:
        comparator = json.loads((REPORTS / details["comparison"]).read_text())["roadmap_strict"]["size_metrics"]
        comparison = {size: sizes[size]["AP50_95"] - comparator[size]["AP50_95"] for size in ("all", "small", "medium", "large")}
    record.update(
        observation=f"{details['model']} strict original-pixel AP-small={sizes['small']['AP50_95']:.6f}, AP-medium={sizes['medium']['AP50_95']:.6f}, AP-large={sizes['large']['AP50_95']:.6f}.",
        evidence_source=str((REPORTS / details["report"]).relative_to(ROOT)),
        controlled_change="Diagnostic evaluation only; checkpoint and legacy validation split are unchanged.",
        cv_result=None,
        worst_fold=None,
        map75=sizes["all"]["AP75"],
        ap_small=sizes["small"]["AP50_95"],
        worst_class={"id": weakest["category_id"], "name": weakest["category_name"], "AP": weakest["sizes"]["all"]["AP50_95"]},
        material_pair_result=None,
        next_experiment="P1 higher resolution and P3 same-checkpoint sliced inference after fixed-fold baseline completion",
        measurements={
            "strict_original_pixel_size_metrics": sizes,
            "strict_minus_native_rtdetr_AP50_95": comparison,
            "box_area_histogram": report["box_area_histogram"],
            "confusion_operating_point": {"confidence": confusion["confidence_threshold"], "iou": confusion["iou_threshold"]},
            "historical_split_only": True,
        },
        artifacts=[details["report"], details["report"].replace("size_ap", "gt_original"), details["report"].replace("size_ap", "predictions_original")],
    )
    return record


def main():
    state = json.loads(STATE.read_text())
    fields = json.loads(REGISTRY.read_text())["required_ledger_fields"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for identifier, details in JOBS.items():
        path = OUTPUT / f"{identifier}.json"
        path.write_text(json.dumps(result(identifier, details, state, fields), indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
