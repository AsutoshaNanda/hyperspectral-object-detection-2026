import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "Experiments" / "registry.json"
ROADMAP_PATH = ROOT / "Resources" / "Hyperspectral_5_Day_Experiment_Roadmap.md"
OUTPUT_PATH = Path(__file__).with_name("experiment_matrix.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(token, eligible_ids, evidence_stage, purpose):
    return {
        "type": "verified_result_selection",
        "token": token,
        "eligible_experiment_ids": list(eligible_ids),
        "evidence_stage": evidence_stage,
        "purpose": purpose,
        "required_status": "measured_and_gate_passed",
        "bound_experiment_id": None,
        "evidence_record_hash": None,
    }


def parameter(name, experiment_ids, purpose):
    return {
        "type": "validated_parameter_selection",
        "token": name,
        "eligible_experiment_ids": list(experiment_ids),
        "evidence_stage": "fixed_design_fold_before_holdout",
        "purpose": purpose,
        "required_status": "measured",
        "bound_value": None,
        "evidence_record_hash": None,
    }


def artifact(name, producer, purpose):
    return {
        "type": "verified_artifact",
        "token": name,
        "producer_experiment_id": producer,
        "purpose": purpose,
        "required_status": "measured",
        "artifact_path": None,
        "artifact_sha256": None,
    }


SELECTIONS = {
    "ARCH*": selected("ARCH*", ["E1a", "E1b", "E1c", "E1d", "E1e", "E1f", "E1g", "E1h", "E1i", "E1j", "DF1", "DF2", "CR1"], "verified_full_cv", "best legal single architecture"),
    "ARCH_NMS*": selected("ARCH_NMS*", ["CR1", "E1e", "E1f", "E1g", "E1i"], "verified_full_cv", "best compatible NMS-based architecture"),
    "ARCH_DETR*": selected("ARCH_DETR*", ["E1a", "E1b", "E1c", "E1d", "E1h", "DF1", "DF2"], "verified_full_cv", "best end-to-end detector"),
    "PRE*": selected("PRE*", ["N0", "N1", "N2", "N3", "N4", "SP0", "SP1", "SP2", "SP3"], "verified_full_cv", "best promoted normalization and spectral preprocessing"),
    "STEM*": selected("STEM*", ["S1a", "S1b", "S1-ViT-random", "S1-ViT-rgb_expanded", "DF1-random", "DF1-rgb_expanded"], "verified_full_cv", "best compatible 16-band initialization"),
    "GEO*": selected("GEO*", ["G1a", "G1b", "G1c"], "verified_full_cv", "best compatible geometry policy"),
    "LOC*": selected("LOC*", ["L1a", "L1b", "L1c", "L1d", "DF3"], "verified_full_cv", "best promoted localization method"),
    "AUG*": selected("AUG*", ["H1a", "H1b", "H1c", "H1d", "SP4"], "verified_full_cv", "best promoted augmentation policy"),
    "SCALE*": selected("SCALE*", ["M1a", "M1b", "M1c", "M1d"], "verified_full_cv", "best promoted multi-scale training setting"),
    "SMALL*": selected("SMALL*", ["P1", "P2", "P3", "P4", "P5"], "verified_full_cv", "best promoted small-object technique"),
    "CLASS*": selected("CLASS*", ["SAM2", "SAM3", "H1c"], "verified_full_cv", "best promoted material or class technique"),
    "LABEL*": selected("LABEL*", ["A1a", "A1b", "A1c", "PL2", "PL3"], "verified_full_cv", "best promoted training-label policy"),
    "TTA*": selected("TTA*", ["T1", "T2", "T3", "T4"], "fixed_cv_inference", "best same-checkpoint TTA policy"),
    "NMS*": selected("NMS*", ["SN1-normal", "SN1-linear", "SN1-gaussian", "SN2-normal", "SN2-linear", "SN2-gaussian"], "fixed_cv_inference", "best suppression policy on the selected NMS-based detector"),
    "HIGH_RES*": selected("HIGH_RES*", ["P1"], "verified_full_cv", "promoted higher-resolution setting"),
    "PATCH*": selected("PATCH*", ["P2"], "verified_full_cv", "promoted patch-training setting"),
    "SLICE*": selected("SLICE*", ["P3", "T4"], "fixed_cv_inference", "promoted same-checkpoint sliced inference setting"),
    "E1_RANK_1": selected("E1_RANK_1", [f"E1{x}" for x in "abcdefghij"], "E1-1", "highest ranked E1 design-fold candidate"),
    "E1_RANK_2": selected("E1_RANK_2", [f"E1{x}" for x in "abcdefghij"], "E1-1", "second ranked E1 design-fold candidate"),
    "BEST_A": selected("BEST_A", [f"A-C{i}" for i in range(1, 6)], "C0", "best compatible Set A combination"),
    "BEST_B": selected("BEST_B", [f"B-C{i}" for i in range(1, 6)], "C0", "best compatible Set B combination"),
    "BEST_C": selected("BEST_C", [f"C-C{i}" for i in range(1, 6)], "C0", "best compatible Set C combination"),
    "BEST_D": selected("BEST_D", [f"D-C{i}" for i in range(1, 6)], "C0", "best compatible Set D combination"),
    "BEST_E": selected("BEST_E", [f"E-C{i}" for i in range(1, 6)], "C0", "best compatible Set E combination"),
}


MODEL_CONTRACT = {
    "trained_model_count": 1,
    "checkpoint_count": 1,
    "same_checkpoint_inference_views_only": True,
    "cross_model_prediction_fusion": False,
}


EXISTING_AUGMENTATIONS = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 5.0,
    "translate": 0.08,
    "scale": 0.35,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 0.5,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "custom_augmentations": [],
}


CONTROL_PROFILES = {
    "transformer_control": {
        "evidence_name": "Plan-C RT-DETR-L control",
        "family": "rtdetr",
        "model": "rtdetr-l.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 2,
        "input_policy": "square_pad",
        "stem_initialization": "random",
        "normalization": "N0",
        "spectral_preprocess": "SP0",
        "multi_scale": 0.0,
        "spectral_augmentation_copies": 0,
        "localization_tail_epochs": 0,
        "seed": 42,
        "augmentations": EXISTING_AUGMENTATIONS,
        "source": "Submission/working_backup_plan_C/hsi_plan_c_ratio/run_config.json",
    },
    "yolo_control": {
        "evidence_name": "Plan-A YOLO26m control",
        "family": "yolo",
        "model": "yolo26m.pt",
        "epochs": 80,
        "imgsz": 1024,
        "batch": 4,
        "input_policy": "native",
        "stem_initialization": "random",
        "normalization": "N0",
        "spectral_preprocess": "SP0",
        "multi_scale": 0.0,
        "spectral_augmentation_copies": 0,
        "localization_tail_epochs": 0,
        "seed": 42,
        "augmentations": EXISTING_AUGMENTATIONS,
        "source": "Submission/Plan_A_Recovery/evidence_v1/hsi_plan_a_recovery/run_config.json",
    },
}


E1_VARIANTS = {
    "E1a": "RT-DETRv2-R18",
    "E1b": "D-FINE-S",
    "E1c": "D-FINE-M",
    "E1d": "DINO 4-scale Swin-L",
    "E1e": "Co-DINO ResNet-50",
    "E1f": "InternImage-T Mask R-CNN FPN",
    "E1g": "Swin-T Mask R-CNN FPN",
    "E1h": "FocalNet-T DINO",
    "E1i": "ConvNeXt V2-T FCMAE transfer",
    "E1j": "DINOv2 ViT-S/14 transfer",
}


PATCH_PROJECTION_CANDIDATES = ["E1d", "E1g", "E1h", "E1i", "E1j"]


def full_budget(profile_ref):
    profile = CONTROL_PROFILES[profile_ref]
    return {
        "type": "full",
        "epochs": profile["epochs"],
        "imgsz": profile["imgsz"],
        "batch": profile["batch"],
        "budget_source": profile_ref,
        "resource_assumption": False,
    }


def architecture_budget(experiment_id):
    return {
        "type": "full",
        "epochs": 50,
        "imgsz": 640,
        "batch": 2,
        "budget_source": "transformer_control_parity",
        "resource_assumption": True,
        "assumption": "Use the RT-DETR control budget for architecture verification. Smoke-test memory may reduce microbatch only; preserve effective batch with accumulation and record the change.",
        "architecture_id": experiment_id,
    }


def base_profile_for(identifier):
    if identifier == "D1-YOLO26m" or identifier.startswith("M1") or identifier.startswith("SN1"):
        return "yolo_control"
    return "transformer_control"


def delta_for(identifier):
    exact = {
        "V1": {"cv_folds": 3, "fold_indices": [0, 1, 2], "grouping": "image_id_only_verified", "holdout_ids_excluded": True},
        "D1-YOLO26m": {"evaluation": "size_ap", "checkpoint": "Plan-A", "area_coordinates": "original_pixels"},
        "D1-RT-DETR-L": {"evaluation": "size_ap", "checkpoint": "Plan-C", "area_coordinates": "original_pixels"},
        "A1-audit": {"operation": "training_only_missing_annotation_audit", "review_labels": ["CONFIRMED_OBJECT", "UNCERTAIN", "NOT_OBJECT"]},
        "A1a": {"label_policy": "official_only"},
        "A1b": {"label_policy": "official_plus_manually_confirmed_training_boxes"},
        "A1c": {"label_policy": "A1b_plus_uncertain_ignore_regions"},
        "N0": {"normalization": "N0", "percentiles": [0.5, 99.5], "fit_scope": "training_fold", "storage": "uint8_unit_interval"},
        "N1": {"normalization": "N1", "statistics": "per_band_mean_std", "fit_scope": "training_fold", "numeric_storage": "float32_signed_then_train_fitted_symmetric_encoding"},
        "N2": {"normalization": "N2", "percentiles": [2.0, 98.0], "fit_scope": "training_fold", "storage": "float32_unit_interval"},
        "N3": {"normalization": "N3", "preclip_percentiles": [0.5, 99.5], "vector_transform": "SNV", "fit_scope": "training_fold", "numeric_storage": "float32_signed_then_train_fitted_symmetric_encoding"},
        "N4": {"normalization": "N4", "preclip_percentiles": [0.5, 99.5], "area_definition": "sum_across_16_equally_spaced_bands", "fit_scope": "training_fold", "storage": "float32"},
        "SP0": {"spectral_preprocess": "SP0"},
        "SP1": {"spectral_preprocess": "SP1", "fit_scope": "training_fold", "selection": "validation_selected_mnf_component_count"},
        "SP2": {"spectral_preprocess": "SP2", "axis": "wavelength", "window_length": 5, "polyorder": 2},
        "SP3": {"spectral_preprocess": "SP3", "reference": "training_fold_mean_spectrum"},
        "SP4": {"spectral_preprocess": "PRE*", "spectral_augmentation": "H1b"},
        "H1a": {"spectral_augmentation_copies": 0},
        "H1b": {"spectral_augmentation_copies": 1, "gain_limit": 0.05, "tilt_limit": 0.03, "noise_std": 0.01, "split": "train_only"},
        "H1c": {"spectral_augmentation_copies": 1, "gain_limit": 0.05, "tilt_limit": 0.03, "noise_std": 0.01, "target_classes": "WEAK_CLASSES*", "split": "train_only"},
        "H1d": {"spectral_preprocess": "PRE*", "spectral_augmentation": "H1b"},
        "S1a": {"input_channels": 16, "stem_initialization": "random"},
        "S1b": {"input_channels": 16, "stem_initialization": "rgb_mean", "activation_scale_correction": 0.1875},
        "S1-ViT-random": {"input_channels": 16, "patch_projection_initialization": "random", "apply_to": PATCH_PROJECTION_CANDIDATES},
        "S1-ViT-rgb_expanded": {"input_channels": 16, "patch_projection_initialization": "rgb_mean", "activation_scale_correction": 0.1875, "apply_to": PATCH_PROJECTION_CANDIDATES},
        "G1a": {"input_policy": "native_square_stretch"},
        "G1b": {"input_policy": "square_pad"},
        "G1c": {"input_policy": "native_rectangular", "rectangular_training": True},
        "L1a": {"localization_tail_epochs": 0},
        "L1b": {"localization_tail_epochs": 10, "lr0": 0.0001, "lrf": 0.1, "degrees": 0.0, "translate": 0.02, "scale": 0.10, "mosaic": 0.0},
        "L1c": {"localization_tail_epochs": 12, "lr0": 0.0001, "lrf": 0.1, "degrees": 0.0, "translate": 0.02, "scale": 0.10, "mosaic": 0.0},
        "L1d": {"localization_tail_epochs": 15, "lr0": 0.0001, "lrf": 0.1, "degrees": 0.0, "translate": 0.02, "scale": 0.10, "mosaic": 0.0},
        "DF1": {"architecture": "D-FINE-S", "execution_variants": ["DF1-random", "DF1-rgb_expanded"]},
        "DF2": {"architecture": "D-FINE-M", "stem_initialization": "random"},
        "DF3": {"architecture": "RT-DETR-L", "isolated_change": "FDR_style_regression_transfer"},
        "DF1-random": {"architecture": "D-FINE-S", "input_channels": 16, "stem_initialization": "random"},
        "DF1-rgb_expanded": {"architecture": "D-FINE-S", "input_channels": 16, "stem_initialization": "rgb_mean", "activation_scale_correction": 0.1875},
        "M1a": {"multi_scale": 0.0},
        "M1b": {"multi_scale": 0.10},
        "M1c": {"multi_scale": 0.20},
        "M1d": {"architecture": "ARCH_DETR*", "multi_scale_policy": "SUPPORTED_SCALE_POLICY*"},
        "T1": {"inference_views": ["native", "horizontal_flip"], "merge": "same_checkpoint"},
        "T2": {"inference_scales": [0.75, 1.0, 1.25], "merge": "same_checkpoint"},
        "T3": {"inference_views": ["native", "horizontal_flip"], "inference_scales": [0.75, 1.0, 1.25], "merge": "same_checkpoint"},
        "T4": {"crop_size": [384, 384], "overlap_ratio": 0.25, "include_full_image": True, "coordinate_restore": "original_pixels", "merge": "same_checkpoint"},
        "SN1": {"execution_variants": ["SN1-normal", "SN1-linear", "SN1-gaussian"]},
        "SN1-normal": {"suppression": "normal_nms", "iou_threshold": 0.65},
        "SN1-linear": {"suppression": "soft_nms_linear", "iou_threshold": 0.65, "score_threshold": 0.001},
        "SN1-gaussian": {"suppression": "soft_nms_gaussian", "sigma": 0.5, "score_threshold": 0.001},
        "SN2": {"execution_variants": ["SN2-normal", "SN2-linear", "SN2-gaussian"]},
        "SN2-normal": {"suppression": "normal_nms", "iou_threshold": 0.65},
        "SN2-linear": {"suppression": "soft_nms_linear", "iou_threshold": 0.65, "score_threshold": 0.001},
        "SN2-gaussian": {"suppression": "soft_nms_gaussian", "sigma": 0.5, "score_threshold": 0.001},
        "SAM1": {"operation": "spectral_pair_diagnostic", "box_inward_shrink_fraction": 0.10, "spectrum_reducer": "median", "prototype_scope": "training_fold", "pairs": [["Apple", "Apple Plastic"], ["Egg", "Egg Plastic"]]},
        "SAM2": {"operation": "single_detector_sam_auxiliary_feature", "prototype_scope": "training_fold", "independent_classifier_fusion": False},
        "SAM3": {"operation": "SAM2_plus_PRE*", "prototype_scope": "training_fold"},
        "PL1": {"operation": "pseudo_label_quality_curve", "subset": "CONTROLLED_HIDDEN_TRAINING_GT*", "teacher_excludes_subset": True},
        "PL2": {"label_policy": "official_plus_manually_confirmed_training_boxes", "validation_and_holdout_gt": "unchanged"},
        "PL3": {"operation": "teacher_student_knowledge_distillation", "scope": "training_fold_only", "test_or_ranking_pseudo_labels": False},
        "CR1": {"architecture": "Cascade R-CNN ResNet-50 FPN", "input_channels": 16, "stem_initialization": "random"},
        "P1": {"imgsz": 1280, "architecture_settings": "otherwise_matched"},
        "P2": {"crop_size": [384, 384], "overlap_ratio": 0.25, "minimum_visible_box_fraction": 0.50, "coordinate_restore": "original_pixels"},
        "P3": {"crop_size": [384, 384], "overlap_ratio": 0.25, "include_full_image": True, "coordinate_restore": "original_pixels", "merge": "same_checkpoint"},
        "P4": {"training": "P2", "inference": "P3", "merge": "same_checkpoint"},
        "P5": {"feature_level": "P2_stride4", "architecture_requirement": "supports_dense_feature_pyramid"},
    }
    if identifier in E1_VARIANTS:
        return {
            "architecture": E1_VARIANTS[identifier],
            "catalog_ref": f"Experiments.models.catalog:{identifier}",
            "input_channels": 16,
            "stem_initialization": "random",
            "candidate_metrics_are_project_only": True,
        }
    return exact.get(identifier, {})


def unresolved_for(identifier):
    refs = []
    if identifier == "A1-audit":
        refs.extend([
            parameter("A1_CONFIDENCE_THRESHOLD*", ["A1-audit"], "candidate confidence selected from validation precision"),
            parameter("A1_LOW_IOU_THRESHOLD*", ["A1-audit"], "candidate low-IoU threshold fixed before training-label review"),
        ])
    if identifier == "SP0":
        refs.append(SELECTIONS["PRE*"] | {"token": "NORMALIZATION_WINNER*", "eligible_experiment_ids": ["N0", "N1", "N2", "N3", "N4"], "purpose": "normalization control for SP-series"})
    if identifier == "SP1":
        refs.extend([
            SELECTIONS["PRE*"] | {"token": "NORMALIZATION_WINNER*", "eligible_experiment_ids": ["N0", "N1", "N2", "N3", "N4"], "purpose": "normalization control for SP-series"},
            parameter("MNF_COMPONENTS*", ["SP1"], "component count selected only on design validation"),
        ])
    if identifier in {"SP2", "SP3"}:
        refs.append(SELECTIONS["PRE*"] | {"token": "NORMALIZATION_WINNER*", "eligible_experiment_ids": ["N0", "N1", "N2", "N3", "N4"], "purpose": "normalization control for SP-series"})
    if identifier in {"SP4", "H1d", "SAM3"}:
        refs.append(SELECTIONS["PRE*"])
    if identifier == "H1c":
        refs.append(selected("WEAK_CLASSES*", ["D1-YOLO26m", "D1-RT-DETR-L"], "verified_size_and_per_class_diagnostics", "measured weak classes"))
    if identifier == "M1d":
        refs.extend([SELECTIONS["ARCH_DETR*"], parameter("SUPPORTED_SCALE_POLICY*", ["M1d"], "architecture-supported transformer scale range")])
    if identifier in {"SAM2", "SAM3"}:
        refs.append(selected("SAM1_SEPARABLE*", ["SAM1"], "SAM1", "SAM1 must demonstrate useful pair separation"))
    if identifier == "PL1":
        refs.extend([
            artifact("CONTROLLED_HIDDEN_TRAINING_GT*", "PL1", "fixed hidden training-label subset"),
            artifact("LEAKAGE_FREE_TEACHER*", "PL1", "teacher trained without the hidden subset"),
        ])
    if identifier in {"A1b", "A1c", "PL2"}:
        refs.append(artifact("MANUAL_REVIEWED_LABELS*", "A1-audit", "confirmed boxes and uncertain-region decisions"))
    if identifier == "PL3":
        refs.append(selected("PL1_THRESHOLD*", ["PL1"], "PL1", "validated pseudo-label confidence policy"))
    if identifier == "P4":
        refs.extend([SELECTIONS["PATCH*"], SELECTIONS["SLICE*"]])
    if identifier == "P5":
        refs.append(selected("P5_COMPATIBLE_ARCH*", ["CR1", "E1e", "E1f", "E1g", "E1i"], "architecture_smoke", "architecture supporting a dense feature level"))
    if identifier in {"DF1", "DF2", "DF3", "DF1-random", "DF1-rgb_expanded", "CR1"}:
        refs.extend([
            SELECTIONS["PRE*"] | {"token": "NORMALIZATION_WINNER*", "eligible_experiment_ids": ["N0", "N1", "N2", "N3", "N4"], "purpose": "measured normalization winner"},
            SELECTIONS["GEO*"],
        ])
    if identifier.startswith("SN2"):
        refs.append(artifact("CR1_PRE_NMS_PREDICTIONS*", "CR1", "matched pre-suppression predictions"))
    return refs


def build_individual(registry):
    rows = []
    for source in registry["experiments"]:
        if source["kind"] == "combination":
            continue
        identifier = source["id"]
        profile_ref = base_profile_for(identifier)
        budget = architecture_budget(identifier) if identifier in {"CR1", "DF1", "DF2", "DF3", *E1_VARIANTS} else full_budget(profile_ref)
        refs = unresolved_for(identifier)
        family_manifest = identifier in {"DF1", "SN1", "SN2"}
        row = {
            "id": identifier,
            "title": source["title"],
            "family": source["family"],
            "kind": source["kind"],
            "source_sections": source["source_sections"],
            "prerequisites": source["prerequisites"],
            "control_profile_ref": profile_ref,
            "matched_control": {
                "profile_ref": profile_ref,
                "same_seed": True,
                "same_v1_fold": True,
                "same_budget": True,
                "same_existing_augmentations": True,
                "only_delta": identifier,
            },
            "candidate_delta": delta_for(identifier),
            "budget": budget,
            "folds": [0, 1, 2] if source["kind"] not in {"inference", "diagnostic", "infrastructure"} and not family_manifest else [],
            "selection_data_roles": ["train", "validation"],
            "holdout_allowed": False,
            "model_contract": dict(MODEL_CONTRACT),
            "unresolved_references": refs,
            "execution_mode": "family_manifest" if family_manifest else "experiment",
            "runnable": not refs and not family_manifest,
            "result": None,
        }
        rows.append(row)
    return rows


def expression_components(identifier, expression):
    token_map = {
        "ARCH*": "ARCH*",
        "ARCH_NMS*": "ARCH_NMS*",
        "PRE*": "PRE*",
        "STEM*": "STEM*",
        "GEO*": "GEO*",
        "LOC*": "LOC*",
        "AUG*": "AUG*",
        "SCALE*": "SCALE*",
        "SMALL*": "SMALL*",
        "CLASS*": "CLASS*",
        "LABEL*": "LABEL*",
        "TTA*": "TTA*",
        "NMS*": "NMS*",
    }
    components = []
    for text, token in token_map.items():
        if text in expression:
            components.append(SELECTIONS[token])
    fixed = {
        "C-C1": [("result", "HIGH_RES*")],
        "C-C2": [("result", "PATCH*")],
        "C-C3": [("result", "SLICE*")],
        "C-C4": [("result", "PATCH*"), ("result", "SLICE*")],
        "D-C1": [("fixed", "T1")],
        "D-C2": [("fixed", "T2")],
        "F-C1": [("result", "E1_RANK_1")],
        "F-C2": [("result", "E1_RANK_2")],
        "F-C3": [("result", "E1_RANK_1")],
        "F-C4": [("result", "E1_RANK_1")],
        "F-C5": [("result", "E1_RANK_1")],
        "G-C1": [("result", "BEST_A"), ("result", "BEST_B")],
        "G-C2": [("result", "BEST_A"), ("result", "BEST_C")],
        "G-C3": [("result", "BEST_B"), ("result", "BEST_D")],
        "G-C4": [("result", "BEST_A"), ("result", "BEST_B"), ("result", "BEST_D")],
        "G-C5": [("result", "BEST_A"), ("result", "BEST_B"), ("result", "BEST_C"), ("result", "BEST_D"), ("result", "BEST_E")],
    }
    for component_type, name in fixed.get(identifier, []):
        components.append(SELECTIONS[name] if component_type == "result" else {"type": "fixed_component", "experiment_id": name})
    unique = []
    seen = set()
    for component in components:
        key = component.get("token", component.get("experiment_id"))
        if key not in seen:
            unique.append(component)
            seen.add(key)
    return unique


def build_combinations(registry):
    rows = []
    for source in registry["experiments"]:
        if source["kind"] != "combination":
            continue
        identifier = source["id"]
        expression = source["settings"]["expression"]
        refs = expression_components(identifier, expression)
        rows.append({
            "id": identifier,
            "set": identifier[0],
            "source_sections": source["source_sections"],
            "expression": expression,
            "components": refs,
            "prerequisites": source["prerequisites"],
            "matched_control": {
                "same_resolved_architecture": True,
                "same_seed": True,
                "same_v1_fold": True,
                "same_budget": True,
                "same_existing_augmentations": True,
                "remove_only_listed_components": True,
            },
            "screen_budget": {"type": "screen", "epochs": 10, "fold": 0},
            "selection_data_roles": ["train", "validation"],
            "holdout_allowed": False,
            "model_contract": dict(MODEL_CONTRACT),
            "unresolved_references": [component for component in refs if component["type"] == "verified_result_selection"],
            "runnable": False,
            "result": None,
        })
    return rows


def rank_ref(stage, rank, eligible, purpose):
    return selected(f"{stage}_RANK_{rank}", eligible, stage, purpose)


def build_e1_stages():
    candidate_ids = list(E1_VARIANTS)
    smoke = []
    screen = []
    for identifier in candidate_ids:
        smoke.append({
            "slot_id": f"{identifier}:E1-0",
            "experiment_id": identifier,
            "purpose": "implementation_smoke",
            "epochs": 1,
            "fold": 0,
            "selection_eligible": False,
            "data_roles": ["train", "validation"],
            "model_contract": dict(MODEL_CONTRACT),
            "runnable": True,
        })
        screen.append({
            "slot_id": f"{identifier}:E1-1",
            "experiment_id": identifier,
            "purpose": "selection",
            "epochs": 10,
            "fold": 0,
            "selection_eligible": True,
            "data_roles": ["train", "validation"],
            "model_contract": dict(MODEL_CONTRACT),
            "runnable": True,
        })
    controls = [
        {"slot_id": "E1-1-control-YOLO", "profile_ref": "yolo_control", "purpose": "matched_control", "epochs": 10, "fold": 0, "selection_eligible": False, "data_roles": ["train", "validation"], "model_contract": dict(MODEL_CONTRACT), "runnable": True},
        {"slot_id": "E1-1-control-RT-DETR", "profile_ref": "transformer_control", "purpose": "matched_control", "epochs": 10, "fold": 0, "selection_eligible": False, "data_roles": ["train", "validation"], "model_contract": dict(MODEL_CONTRACT), "runnable": True},
    ]
    full = []
    for rank in range(1, 4):
        reference = rank_ref("E1-1", rank, candidate_ids, "top-three E1 full verification slot")
        for fold in range(3):
            full.append({
                "slot_id": f"E1-2-rank{rank}-fold{fold}",
                "experiment_ref": reference,
                "purpose": "selection",
                "budget": {"type": "candidate_full_budget", "epochs_by_candidate": {identifier: 50 for identifier in candidate_ids}, "resource_assumption": True},
                "fold": fold,
                "selection_eligible": True,
                "data_roles": ["train", "validation"],
                "model_contract": dict(MODEL_CONTRACT),
                "runnable": False,
            })
    winner = {
        "slot_id": "E1-3-winner",
        "experiment_ref": rank_ref("E1-2", 1, candidate_ids, "final verified E1 winner"),
        "purpose": "final_evaluation",
        "epochs": 0,
        "data_roles": ["holdout"],
        "selection_eligible": False,
        "model_contract": dict(MODEL_CONTRACT),
        "runnable": False,
    }
    return {"smoke": smoke, "screen": screen, "screen_controls": controls, "full_cv_top3": full, "winner_holdout": [winner]}


def build_combination_stages(combination_ids):
    c0 = [{
        "slot_id": f"{identifier}:C0",
        "experiment_id": identifier,
        "purpose": "selection",
        "epochs": 10,
        "fold": 0,
        "selection_eligible": True,
        "data_roles": ["train", "validation"],
        "model_contract": dict(MODEL_CONTRACT),
        "runnable": False,
    } for identifier in combination_ids]
    c1 = []
    for rank in range(1, 16):
        c1.append({
            "slot_id": f"C1-rank{rank}-fold1",
            "experiment_ref": rank_ref("C0", rank, combination_ids, "top-15 second-fold verification slot"),
            "purpose": "selection",
            "epochs": 10,
            "fold": 1,
            "selection_eligible": True,
            "data_roles": ["train", "validation"],
            "model_contract": dict(MODEL_CONTRACT),
            "runnable": False,
        })
    c2 = []
    for rank in range(1, 11):
        reference = rank_ref("C1", rank, combination_ids, "top-10 three-fold full-budget verification slot")
        for fold in range(3):
            c2.append({
                "slot_id": f"C2-rank{rank}-fold{fold}",
                "experiment_ref": reference,
                "purpose": "selection",
                "budget": {
                    "type": "resolved_architecture_full_budget",
                    "epochs_by_profile": {"transformer_control": 50, "yolo_control": 80, "new_architecture_assumption": 50},
                    "imgsz_by_profile": {"transformer_control": 640, "yolo_control": 1024, "new_architecture_assumption": 640},
                    "batch_by_profile": {"transformer_control": 2, "yolo_control": 4, "new_architecture_assumption": 2},
                    "resource_assumption": "new_architecture_assumption only",
                },
                "fold": fold,
                "selection_eligible": True,
                "data_roles": ["train", "validation"],
                "model_contract": dict(MODEL_CONTRACT),
                "runnable": False,
            })
    c3 = []
    for rank in range(1, 4):
        c3.append({
            "slot_id": f"C3-rank{rank}-holdout",
            "experiment_ref": rank_ref("C2", rank, combination_ids, "top-three frozen finalist holdout slot"),
            "holdout_ref": artifact("UNTOUCHED_HOLDOUT_OR_REVISED_PROTOCOL*", "V1", "one frozen final evaluation protocol; current legacy holdout is already contaminated"),
            "purpose": "final_evaluation",
            "epochs": 0,
            "selection_eligible": False,
            "data_roles": ["holdout"],
            "model_contract": dict(MODEL_CONTRACT),
            "runnable": False,
        })
    control = {
        "slot_id": "C0-corresponding-controls",
        "purpose": "matched_control",
        "epochs": 10,
        "fold": 0,
        "selection_eligible": False,
        "data_roles": ["train", "validation"],
        "model_contract": dict(MODEL_CONTRACT),
        "runnable": False,
    }
    return {"C0_all_35": c0, "C0_controls": [control], "C1_top15": c1, "C2_top10_three_folds": c2, "C3_top3_holdout": c3}


def build_matrix():
    registry = json.loads(REGISTRY_PATH.read_text())
    individuals = build_individual(registry)
    combinations = build_combinations(registry)
    vit_expansions = []
    for experiment_id, initialization in (("S1-ViT-random", "random"), ("S1-ViT-rgb_expanded", "rgb_mean")):
        for candidate_id in PATCH_PROJECTION_CANDIDATES:
            vit_expansions.append({
                "slot_id": f"{experiment_id}@{candidate_id}",
                "experiment_id": experiment_id,
                "architecture_id": candidate_id,
                "patch_projection_initialization": initialization,
                "activation_scale_correction": None if initialization == "random" else 0.1875,
                "matched_control": "same candidate, fold, budget, and existing augmentations",
                "folds": [0, 1, 2],
                "model_contract": dict(MODEL_CONTRACT),
                "runnable": True,
            })
    return {
        "schema_version": 1,
        "status": "configuration_only_no_results",
        "sources": {
            "roadmap": {"path": str(ROADMAP_PATH.relative_to(ROOT)), "sha256": sha256(ROADMAP_PATH)},
            "registry": {"path": str(REGISTRY_PATH.relative_to(ROOT)), "sha256": sha256(REGISTRY_PATH)},
        },
        "invariants": {
            "seed": 42,
            "input_channels": 16,
            "cv_folds": 3,
            "fixed_fold_source": "V1",
            "fixed_fold_split_hash": "a878b0839dc15b1fbc92b05aa43698831994bc9a1572a542eb35c786eb5b844e",
            "fixed_fold_hashes": ["c25c63d577c6e1a3904376cc9edfef15ec74eaeb94f8ef8371cc85e9f8c69a6e", "75dfe5575d68a68322130e76d5311f9546fd62396645ab44ba9ac4189ab86da7", "0002ef59a3a4a856fe00719757d0d2f4d4c6cec3ff419d274092a72f52624c65"],
            "design_fold": 0,
            "second_fold": 1,
            "smoke_epochs": 1,
            "screen_epochs": 10,
            "selection_data_roles": ["train", "validation"],
            "holdout_selection_forbidden": True,
            "single_trained_model": True,
            "cross_model_prediction_fusion": False,
            "same_checkpoint_tta_only": True,
            "result_fields_must_remain_null_until_execution": True,
        },
        "control_profiles": CONTROL_PROFILES,
        "resource_assumptions": {
            "new_architecture_full_budget": "50 epochs at 640 with effective batch 2 until smoke runtime and memory evidence is recorded",
            "yolo_full_budget": "80 epochs at 1024 with batch 4 from Plan-A",
            "transformer_full_budget": "50 epochs at 640 with batch 2 from Plan-C",
            "screen_budget": "10 epochs on V1 fold 0 for every candidate and matched control",
            "smoke_budget": "1 epoch on V1 fold 0; implementation validity only",
            "memory_rule": "An OOM may reduce microbatch only. Preserve effective batch through accumulation when supported and record the exact change.",
            "runtime_estimates": None,
        },
        "selection_bindings": SELECTIONS,
        "individual_experiments": individuals,
        "expanded_variant_runs": {"vit_patch_projection": vit_expansions},
        "combination_experiments": combinations,
        "e1_stages": build_e1_stages(),
        "combination_stages": build_combination_stages([row["id"] for row in combinations]),
        "results": None,
    }


def validate_matrix(matrix):
    registry = json.loads(REGISTRY_PATH.read_text())
    registry_individuals = {row["id"] for row in registry["experiments"] if row["kind"] != "combination"}
    matrix_individuals = {row["id"] for row in matrix["individual_experiments"]}
    if matrix_individuals != registry_individuals:
        raise ValueError(f"Individual coverage mismatch: {sorted(matrix_individuals ^ registry_individuals)}")
    expected_combinations = {row["id"]: row["settings"]["expression"] for row in registry["experiments"] if row["kind"] == "combination"}
    actual_combinations = {row["id"]: row["expression"] for row in matrix["combination_experiments"]}
    if actual_combinations != expected_combinations or len(actual_combinations) != 35:
        raise ValueError("The exact 35 combination formulas are not preserved")
    for row in [*matrix["individual_experiments"], *matrix["combination_experiments"]]:
        if row["unresolved_references"] and row["runnable"]:
            raise ValueError(f"Unresolved experiment is runnable: {row['id']}")
        if row["result"] is not None:
            raise ValueError(f"Configuration fabricated a result: {row['id']}")
        if row["model_contract"] != MODEL_CONTRACT:
            raise ValueError(f"Illegal model contract: {row['id']}")
    stage_groups = [*matrix["e1_stages"].values(), *matrix["combination_stages"].values()]
    for group in stage_groups:
        for slot in group:
            if slot["purpose"] == "selection" and "holdout" in slot["data_roles"]:
                raise ValueError(f"Holdout appears in selection stage: {slot['slot_id']}")
            if slot["model_contract"] != MODEL_CONTRACT:
                raise ValueError(f"Illegal stage model contract: {slot['slot_id']}")
    counts = {
        "E1_smoke": len(matrix["e1_stages"]["smoke"]),
        "E1_screen": len(matrix["e1_stages"]["screen"]),
        "E1_controls": len(matrix["e1_stages"]["screen_controls"]),
        "E1_full": len(matrix["e1_stages"]["full_cv_top3"]),
        "E1_holdout": len(matrix["e1_stages"]["winner_holdout"]),
        "C0": len(matrix["combination_stages"]["C0_all_35"]),
        "C1": len(matrix["combination_stages"]["C1_top15"]),
        "C2": len(matrix["combination_stages"]["C2_top10_three_folds"]),
        "C3": len(matrix["combination_stages"]["C3_top3_holdout"]),
    }
    if counts != {"E1_smoke": 10, "E1_screen": 10, "E1_controls": 2, "E1_full": 9, "E1_holdout": 1, "C0": 35, "C1": 15, "C2": 30, "C3": 3}:
        raise ValueError(f"Stage slot counts differ: {counts}")
    source = ROADMAP_PATH.read_text()
    roadmap_ids = set(re.findall(r"^## ((?:N|SP|DF|T|SAM|PL|P|E1|SN)\d+[a-j]?)\s+[—-]", source, re.M))
    roadmap_ids.update(re.findall(r"\*\*((?:A1|H1|S1|G1|L1|M1)[a-d]):\*\*", source))
    roadmap_ids.update({"V1", "CR1", "S1-ViT-random", "S1-ViT-rgb_expanded", "D1-YOLO26m", "D1-RT-DETR-L", "A1-audit"})
    if roadmap_ids - matrix_individuals:
        raise ValueError(f"Roadmap IDs omitted: {sorted(roadmap_ids - matrix_individuals)}")
    return counts


def main():
    matrix = build_matrix()
    counts = validate_matrix(matrix)
    OUTPUT_PATH.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"individuals": len(matrix["individual_experiments"]), "combinations": len(matrix["combination_experiments"]), **counts}, indent=2))


if __name__ == "__main__":
    main()
