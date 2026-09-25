import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Resources/Hyperspectral_5_Day_Experiment_Roadmap.md"


def build_registry(source=SOURCE):
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    starts = [(i, re.match(r"# (\d+)\. (.*)", line)) for i, line in enumerate(lines)]
    starts = [(i, match) for i, match in starts if match]
    sections = {}
    for index, (start, match) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(lines)
        sections[match[1]] = {"heading": lines[start].strip(), "start_line": start + 1, "end_line": end, "text": "".join(lines[start:end])}
    experiments = []

    def add(identifier, title, section, family=None, kind="individual", prerequisites=(), settings=None, unresolved=(), gate=None, variants=()):
        experiments.append({"id": identifier, "title": title, "family": family or identifier, "kind": kind, "status": "pending", "source_sections": [str(section)], "prerequisites": list(prerequisites), "settings": settings or {}, "unresolved_settings": list(unresolved), "gate": gate, "variants": list(variants), "evidence": None})

    add("V1", "Group-aware 3-fold CV", 5, kind="infrastructure", settings={"cv_folds": 3, "design_only": True, "fold_indices": [0, 1, 2]}, unresolved=["Verified capture/session grouping or documented image_id_only fallback"])
    for model in ("YOLO26m", "RT-DETR-L"):
        add("D1-" + model, "Current checkpoint size diagnostics: " + model, 6, "D1", "diagnostic", ("V1",), {"area_coordinates": "original_pixels", "small_area_lt": 1024, "medium_area_gte": 1024, "medium_area_lt": 9216, "large_area_gte": 9216}, ["Checkpoint path and hash"])
    add("A1-audit", "Training-only missing-annotation candidate discovery and review", 7, "A1", "diagnostic", ("V1",), unresolved=["Confidence threshold selected from validation precision", "Low-IoU candidate threshold", "Manual candidate reviews"])
    for suffix, title in [("a", "Official annotations only"), ("b", "Official plus confirmed manually reviewed missing boxes"), ("c", "A1b plus uncertain-region ignore handling")]:
        add("A1" + suffix, title, 7, "A1", prerequisites=("V1",) if suffix == "a" else ("A1-audit", "A1a"), gate=None if suffix == "a" else "Only reviewed confirmed boxes enter training-copy labels; validation/holdout GT unchanged")
    normalization = [("N0", "Existing normalization baseline", {"percentile_low": 0.5, "percentile_high": 99.5, "fit_scope": "training_fold", "clip": True, "scale": [0, 1], "storage": "8-bit TIFF"}, []), ("N1", "Training-only per-band z-score", {}, ["Numeric storage preserving z-score values"]), ("N2", "Training-only robust min-max", {}, ["Robust statistics definition and limits"]), ("N3", "Per-vector SNV after train-derived clipping", {}, ["Fixed clipping parameters", "Numeric storage preserving SNV values"]), ("N4", "Area-under-spectrum normalization after fixed clipping", {}, ["Fixed clipping parameters", "Spectral sum or wavelength-integrated area definition"])]
    for identifier, title, settings, unresolved in normalization:
        add(identifier, title, 8, "N", prerequisites=("V1",) if identifier == "N0" else ("N0",), settings=settings, unresolved=unresolved)
    for identifier, title in [("SP0", "No extra spectral denoising"), ("SP1", "Train-only MNF fit and reconstruction"), ("SP2", "Savitzky-Golay across wavelength axis"), ("SP3", "MSC with train-only reference spectrum"), ("SP4", "Winning spectral preprocessing plus train-only spectral augmentation")]:
        dependencies = ("N0", "N1", "N2", "N3", "N4") if identifier == "SP0" else ("SP0",)
        if identifier == "SP4":
            dependencies = ("SP0", "SP1", "SP2", "SP3", "H1b")
        unresolved = {"SP1": ["Validation-selected noise component removal"], "SP2": ["Conservative valid window and polynomial order"], "SP4": ["Measured spectral preprocessing winner"]}.get(identifier, [])
        add(identifier, title, 9, "SP", prerequisites=dependencies, unresolved=unresolved, gate="Normalization winner chosen from real CV results")
    for suffix, title in [("a", "No spectral augmentation"), ("b", "One conservative spectral-augmentation copy"), ("c", "Class-targeted spectral copy for measured weak classes"), ("d", "Winning SP preprocessing plus H1b")]:
        dependencies = ("V1",) if suffix == "a" else ("H1a",)
        if suffix == "d":
            dependencies = ("SP1", "SP2", "SP3", "H1b")
        add("H1" + suffix, title, 10, "H1", prerequisites=dependencies, settings={"copies": 1} if suffix == "b" else {}, unresolved=["Conservative augmentation parameters"] if suffix in "bc" else [], gate="Target classes from measured per-class AP/confusion" if suffix == "c" else None)
    add("S1a", "Random 16-band convolutional input stem", 11, "S1", prerequisites=("V1",), settings={"initialization": "random", "input_channels": 16})
    add("S1b", "RGB mean 16-band convolutional input stem", 11, "S1", prerequisites=("S1a",), settings={"initialization": "rgb_mean", "scale_correction": "3/16", "input_channels": 16})
    for variant in ("random", "rgb_expanded"):
        add("S1-ViT-" + variant, "Actual ViT patch projection: " + variant, 11, "S1", prerequisites=("V1",), settings={"input_channels": 16, "initialization": variant}, gate="Apply to each ViT-like E1 candidate with an actual patch-embedding projection", unresolved=["Activation-scale correction for RGB-expanded projection"] if variant == "rgb_expanded" else [])
    for suffix, title in [("a", "Native direct square stretch"), ("b", "Center-pad to square then resize"), ("c", "Native aspect-preserving rectangular training")]:
        add("G1" + suffix, title, 12, "G1", prerequisites=("V1",), gate="Use compatible architecture for rectangular training" if suffix == "c" else None)
    for suffix, epochs in [("a", 0), ("b", 10), ("c", 12), ("d", 15)]:
        add("L1" + suffix, "Localization fine-tuning: " + str(epochs) + " tail epochs", 13, "L1", prerequisites=("V1",), settings={"tail_epochs": epochs}, unresolved=[] if not epochs else ["Low learning rate", "Reduced geometric augmentation configuration"])
    for identifier, title in [("DF1", "Full 16-band D-FINE-S"), ("DF2", "Full 16-band D-FINE-M"), ("DF3", "RT-DETR plus isolated FDR-style regression transfer")]:
        dependencies = ("V1",) if identifier == "DF1" else ("DF1",)
        add(identifier, title, 14, "DF", kind="architecture" if identifier != "DF3" else "individual", prerequisites=dependencies, gate="Measured normalization and geometry winners; full D-FINE correctness before DF2/DF3", variants=["DF1-random", "DF1-rgb_expanded"] if identifier == "DF1" else [], unresolved=["Architecture-specific config and pretrained provenance"])
    for variant in ("random", "rgb_expanded"):
        add("DF1-" + variant, "D-FINE-S 16-band initialization: " + variant, 14, "DF1", prerequisites=("V1",), settings={"initialization": variant, "input_channels": 16})
    for suffix, scale in [("a", 0), ("b", 0.10), ("c", 0.20)]:
        add("M1" + suffix, "YOLO multi-scale training " + str(scale), 15, "M1", prerequisites=("V1",), settings={"multi_scale": scale})
    add("M1d", "Best transformer supported multi-scale training policy", 15, "M1", prerequisites=("V1",), unresolved=["Measured best transformer", "Supported multi-scale training policy"])
    for identifier, title in [("T1", "Same-checkpoint horizontal flip TTA"), ("T2", "Same-checkpoint multi-scale TTA"), ("T3", "Same-checkpoint flip plus multi-scale TTA"), ("T4", "Same-checkpoint overlapping crop/sliced TTA")]:
        add(identifier, title, 16, "T", kind="inference", prerequisites=("T1", "T2") if identifier == "T3" else ("V1",), unresolved={"T2": ["Small fixed scale set selected before holdout"], "T4": ["Crop dimensions, overlap and merge policy"]}.get(identifier, []), gate="Identical checkpoint; policies selected on validation/CV before holdout")
    for identifier, model in [("SN1", "YOLO"), ("SN2", "Cascade R-CNN")]:
        variants = [identifier + "-" + name for name in ("normal", "linear", "gaussian")]
        add(identifier, model + " NMS comparison", 17, "SN", "inference", ("V1",) if identifier == "SN1" else ("CR1",), variants=variants)
        for variant in ("normal", "linear", "gaussian"):
            add(identifier + "-" + variant, model + " " + variant + " NMS", 17, identifier, "inference", ("V1",) if identifier == "SN1" else ("CR1",), settings={"suppression": variant}, gate="Same checkpoint and pre-suppression predictions; tune only on validation folds", unresolved=["Suppression parameters"])
    add("SAM1", "Training-fold material-pair spectral separability diagnostic", 18, "SAM", "diagnostic", ("V1",), unresolved=["Fixed inward box shrink percentage", "Median or trimmed-mean spectrum", "Training-selected pair threshold"])
    add("SAM2", "Same-model SAM auxiliary feature", 18, "SAM", prerequisites=("SAM1",), gate="SAM1 demonstrates useful measured separation; never fuse independent classifier predictions", unresolved=["Single-model auxiliary feature integration"])
    add("SAM3", "SAM plus winning spectral preprocessing", 18, "SAM", prerequisites=("SAM1", "SP0", "SP1", "SP2", "SP3"), gate="SP-series and SAM1 results must exist")
    add("PL1", "Hidden known training-GT pseudo-label quality audit", 19, "PL", "diagnostic", ("V1",), unresolved=["Controlled hidden training subset", "Teacher checkpoint trained without hidden subset", "Threshold curve"])
    add("PL2", "Confirmed missing-label training augmentation", 19, "PL", prerequisites=("A1-audit", "A1a"), gate="Only manually confirmed training candidates; unchanged official validation/holdout GT")
    add("PL3", "Teacher/student training-fold code path", 19, "PL", prerequisites=("V1", "PL1"), gate="Training-fold data only; test/ranking pseudo-label expansion requires exact organizer permission", unresolved=["Teacher/student and knowledge-distillation configuration"])
    add("CR1", "16-band Cascade R-CNN", 20, kind="architecture", prerequisites=("V1",), settings={"backbone_control": "ResNet-50-FPN or closest stable MMDetection baseline", "input_channels": 16}, unresolved=["Measured normalization and geometry winners", "Pretrained provenance"])
    for identifier, title in [("P1", "Higher input resolution"), ("P2", "Overlapping patch training"), ("P3", "Same-checkpoint sliced inference"), ("P4", "Patch training plus sliced inference"), ("P5", "High-resolution feature level")]:
        dependencies = ("D1-YOLO26m", "D1-RT-DETR-L") + (("P2", "P3") if identifier == "P4" else ())
        add(identifier, title, 21, "P", "inference" if identifier == "P3" else "individual", dependencies, unresolved={"P1": ["Higher resolution against fixed architecture/settings"], "P2": ["Crop dimensions, overlap and minimum visible-box fraction"], "P3": ["Crop dimensions, overlap and merge policy"], "P5": ["At least one architecture supporting denser feature pyramid"]}.get(identifier, []), gate="D1 size measurements exist; P5 must run on at least one compatible architecture")
    e1 = re.findall(r"^## (E1[a-j])\s+[—-]\s+(.+)$", sections["22"]["text"], re.M)
    for identifier, title in e1:
        add(identifier, title, 22, "E1", "architecture", ("V1",), gate="Public weights, source, pretrained dataset and license recorded; S1 16-band adaptation; fair E1 stages", unresolved=["Official variant and compatible detection head", "Exact pretrained weights and provenance", "Common short budget", "Normalization, geometry and seed"])
    combinations = []
    for section in range(27, 34):
        for identifier, expression in re.findall(r"^## ([A-G]-C[1-5])\s*\n```text\n([^`]+)```", sections[str(section)]["text"], re.M):
            dependencies = []
            if identifier.startswith("G-"):
                letters = {"G-C1": "AB", "G-C2": "AC", "G-C3": "BD", "G-C4": "ABD", "G-C5": "ABCDE"}[identifier]
                dependencies = [f"{letter}-C{index}" for letter in letters for index in range(1, 6)]
            add(identifier, expression.strip(), section, "combination-" + identifier[0], "combination", prerequisites=dependencies, gate="individual-results-complete-and-components-promoted", settings={"expression": expression.strip()}, unresolved=["All components resolved from measured eligible individual results", "Common fixed design fold and shortened budget"])
            combinations.append(identifier)
    stages = {
        "E1-0": {"source_sections": ["23"], "selection": "every E1a-j candidate", "requirements": ["Load pretrained weights", "Accept 16 bands", "Forward pass", "One short smoke epoch", "Valid predictions", "GPU memory/runtime"], "failure_meaning": "implementation failure, not performance rejection"},
        "E1-1": {"source_sections": ["23"], "selection": "every E1a-j candidate and shortened-budget YOLO/RT-DETR controls", "requirements": ["Same fold", "Same fixed short budget", "Same normalization", "Same geometry", "Same seed"]},
        "E1-2": {"source_sections": ["23", "5"], "selection": "best three E1-1 candidates", "candidate_count": 3, "fold_count": 3, "budget": "normal full budget", "gate": "architecture-promotion"},
        "E1-3": {"source_sections": ["23"], "selection": "final E1 winner only", "candidate_count": 1, "evaluation": "holdout"},
        "C0": {"source_sections": ["34"], "selection": "all 35 combinations", "candidate_count": 35, "requirements": ["Exact same fixed design fold", "Exact same shortened budget", "Corresponding shortened-budget baseline"]},
        "C1": {"source_sections": ["34"], "selection": "top 15 C0 by design-fold mAP50-95 with diagnostic verification", "candidate_count": 15, "evaluation": "second fold", "diagnostics": ["mAP75", "AP-small", "worst-class AP", "difficult-pair confusion"]},
        "C2": {"source_sections": ["34"], "selection": "top 10 after second-fold verification", "candidate_count": 10, "fold_count": 3, "budget": "normal full budget"},
        "C3": {"source_sections": ["34"], "selection": "top three full-CV candidates", "candidate_count": 3, "evaluation": "once on untouched holdout", "requirements": ["Freeze settings before holdout"]},
        "C4": {"source_sections": ["34", "35"], "selection": "one final model/checkpoint/config", "candidate_count": 1, "requirements": ["Public Kaggle score confirms and is not selector", "Finalist ranking order preserved"]},
    }
    execution = []

    def requirement(identifier, stage, experiment=None, selection=None, fold=None, prerequisites=()):
        execution.append({"id": identifier, "stage": stage, "experiment_id": experiment, "selection": selection, "fold": fold, "prerequisites": list(prerequisites), "status": "pending", "evidence": None})

    for identifier, _ in e1:
        requirement(identifier + ":E1-0", "E1-0", identifier, prerequisites=("V1",))
        requirement(identifier + ":E1-1", "E1-1", identifier, fold="fixed_design_fold", prerequisites=(identifier + ":E1-0",))
    for model in ("YOLO", "RT-DETR"):
        requirement("E1-1-control-" + model, "E1-1", selection="Current " + model + " retrained with identical shortened budget", fold="fixed_design_fold", prerequisites=("V1",))
    for rank in range(1, 4):
        for fold in range(3):
            requirement(f"E1-2-rank{rank}-fold{fold}", "E1-2", selection=f"E1-1 rank {rank}", fold=fold, prerequisites=tuple(identifier + ":E1-1" for identifier, _ in e1))
    requirement("E1-3-winner", "E1-3", selection="Verified final E1 winner", prerequisites=tuple(row["id"] for row in execution if row["stage"] == "E1-2"))
    for identifier in combinations:
        requirement(identifier + ":C0", "C0", identifier, fold="fixed_design_fold", prerequisites=("individual-results-complete-and-components-promoted",))
    requirement("C0-corresponding-controls", "C0", selection="Each distinct corresponding baseline at the identical short budget", fold="fixed_design_fold")
    for stage, count, folds, selector in [("C1", 15, ["second_fold"], "C0"), ("C2", 10, [0, 1, 2], "C1"), ("C3", 3, ["untouched_holdout"], "C2")]:
        for rank in range(1, count + 1):
            for fold in folds:
                requirement(f"{stage}-rank{rank}-{fold}", stage, selection=f"{selector} selected rank {rank}", fold=fold, prerequisites=tuple(row["id"] for row in execution if row["stage"] == selector))
    requirement("C4-final-selection", "C4", selection="One verified final trained model", prerequisites=tuple(row["id"] for row in execution if row["stage"] == "C3"))
    checklist = [{"id": f"DONE-{index:02d}", "requirement": item, "status": "pending", "evidence": None, "source_sections": ["42"]} for index, item in enumerate(re.findall(r"^- \[ \] (.+)$", sections["42"]["text"], re.M), 1)]
    fields = re.search(r"```text\n(.*?)```", sections["4"]["text"], re.S)[1].strip().splitlines()
    return {"schema_version": 1, "source": {"path": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source), "sha256": hashlib.sha256(raw).hexdigest()}, "status_policy": "This registry describes obligations, not completed runs. Pending changes only with independently recorded execution evidence. Existing scores in section 2 are source claims, not imported results.", "source_sections": sections, "required_ledger_fields": fields, "global_constraints": {"source_sections": ["1", "3", "4", "5", "24", "25", "26", "35", "38", "39"], "single_trained_model": True, "cross_model_prediction_fusion": False, "cv_folds": 3, "small_technique_map_gain_gte": 0.005, "small_technique_map75_gain_gte_if_no_map_loss": 0.005, "worst_present_class_max_regression": 0.03, "architecture_map_gain_gte": 0.01, "architecture_alternative_gate": "Hard-class AP/localization improves materially without reducing overall mAP", "unresolved_gate_definitions": ["Catastrophic fold regression threshold is not numerically specified", "Material hard-class/localization gain is not numerically specified"], "individual_ranking": ["cv_mean_map50_95_desc", "worst_fold_map50_95_desc", "map75_desc", "worst_class_ap_desc", "ap_small_desc"], "finalist_ranking_source": "35", "stop_reasons_source": "38"}, "experiments": experiments, "stages": stages, "execution_requirements": execution, "definition_of_done": checklist, "coverage": {"experiment_records": len(experiments), "combination_experiments": len(combinations), "e1_candidates": len(e1), "execution_requirements": len(execution), "definition_of_done_requirements": len(checklist)}, "limitations": ["No execution or measured metric is claimed by registry generation.", "Unspecified settings remain unresolved; no epoch budgets, seeds, thresholds, crop sizes or architecture variants are invented.", "Alternatives named with slash or 'or' remain explicit candidate choices, not invented mandates to train every size.", "SN2 normal/linear/Gaussian variants apply section 17's suppression comparison to Cascade; exact parameters require validation selection.", "Aggregate family records and their child variants do not represent additional duplicate training mandates.", "SP4/H1d, T4/P3 and A1b/PL2 retain separate obligations; any shared execution must prove identical configuration and satisfy both evidence contracts.", "Conditional verification slots are obligations to select eligible measured candidates, not claims that any candidate has qualified.", "All individual detector comparisons require shared 3-fold evidence under sections 3 and 5, with E1 and combination stages following their explicit staged budgets."]}


def validate_registry(registry):
    ids = [entry["id"] for entry in registry["experiments"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment ID")
    combinations = {entry["id"] for entry in registry["experiments"] if entry["kind"] == "combination"}
    expected = {f"{letter}-C{index}" for letter in "ABCDEFG" for index in range(1, 6)}
    if combinations != expected:
        raise ValueError("Missing or extra combinations")
    if {entry["id"] for entry in registry["experiments"] if entry["family"] == "E1"} != {"E1" + letter for letter in "abcdefghij"}:
        raise ValueError("Missing E1 candidate")
    for entry in registry["experiments"]:
        for dependency in entry["prerequisites"]:
            if dependency not in ids:
                raise ValueError(f"Unknown prerequisite {dependency}")
        if entry["evidence"] is not None or entry["status"] != "pending":
            raise ValueError("Generator must not fabricate execution evidence")
    for stage, count in [("E1-0", 10), ("E1-1", 12), ("E1-2", 9), ("E1-3", 1), ("C0", 36), ("C1", 15), ("C2", 30), ("C3", 3), ("C4", 1)]:
        if sum(row["stage"] == stage for row in registry["execution_requirements"]) != count:
            raise ValueError(f"Incorrect execution obligation count for {stage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("registry.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    registry = build_registry()
    validate_registry(registry)
    encoded = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if args.output.read_text() != encoded:
            raise SystemExit("Registry differs from current roadmap/generator")
    else:
        if args.output.exists():
            existing = json.loads(args.output.read_text())
            for key in ("experiments", "execution_requirements", "definition_of_done"):
                if any(row.get("status") != "pending" or row.get("evidence") is not None for row in existing.get(key, [])):
                    raise SystemExit("Refusing to overwrite registry containing execution progress")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(json.dumps(registry["coverage"], indent=2))


if __name__ == "__main__":
    main()
