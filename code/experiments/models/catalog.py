from dataclasses import asdict, dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class CandidateSpec:
    experiment_id: str
    name: str
    role: str
    framework: str
    source_url: str
    evidence_url: str
    checkpoint: str
    pretrained_dataset: str
    code_license: str
    weights_license: str
    config: str
    required_modules: tuple[str, ...]
    integration: str
    input_layer_kind: str
    verified_on: str = "2026-09-19"

    def to_dict(self):
        row = asdict(self)
        row["required_modules"] = list(self.required_modules)
        return row


CANDIDATES = {
    "CR1": CandidateSpec(
        "CR1",
        "Cascade R-CNN ResNet-50 FPN",
        "detector",
        "mmdetection-3.x",
        "https://github.com/open-mmlab/mmdetection",
        "https://github.com/open-mmlab/mmdetection/tree/main/configs/cascade_rcnn",
        "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth",
        "COCO 2017",
        "Apache-2.0",
        "Apache-2.0 code; COCO terms also apply",
        "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
        ("mmdet", "mmengine", "mmcv"),
        "mmdetection",
        "input_stem",
    ),
    "DF1": CandidateSpec(
        "DF1",
        "D-FINE-S",
        "detector",
        "D-FINE official",
        "https://github.com/Peterande/D-FINE",
        "https://github.com/Peterande/D-FINE#model-zoo",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "COCO 2017",
        "Apache-2.0",
        "Apache-2.0 repository checkpoint; COCO terms also apply",
        "configs/dfine/dfine_hgnetv2_s_coco.yml",
        ("torch", "torchvision"),
        "dfine_official",
        "input_stem",
    ),
    "DF2": CandidateSpec(
        "DF2",
        "D-FINE-M",
        "detector",
        "D-FINE official",
        "https://github.com/Peterande/D-FINE",
        "https://github.com/Peterande/D-FINE#model-zoo",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
        "COCO 2017",
        "Apache-2.0",
        "Apache-2.0 repository checkpoint; COCO terms also apply",
        "configs/dfine/dfine_hgnetv2_m_coco.yml",
        ("torch", "torchvision"),
        "dfine_official",
        "input_stem",
    ),
    "DF3": CandidateSpec(
        "DF3",
        "RT-DETR plus isolated FDR-style regression",
        "ablation",
        "project RT-DETR branch",
        "https://github.com/Peterande/D-FINE",
        "https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_decoder.py",
        "parent RT-DETR checkpoint selected by the controlled baseline",
        "parent RT-DETR checkpoint dataset",
        "parent RT-DETR license plus Apache-2.0 D-FINE source",
        "same as parent checkpoint; D-FINE source is Apache-2.0",
        "project-owned RT-DETR config",
        ("torch",),
        "df3_isolated_only",
        "input_stem",
    ),
    "E1a": CandidateSpec(
        "E1a",
        "RT-DETRv2-R18",
        "detector",
        "transformers",
        "https://github.com/lyuwenyu/RT-DETR",
        "https://huggingface.co/PekingU/rtdetr_v2_r18vd",
        "PekingU/rtdetr_v2_r18vd",
        "COCO 2017",
        "Apache-2.0",
        "Apache-2.0 model card; COCO terms also apply",
        "Hugging Face model config",
        ("torch", "transformers"),
        "transformers_rtdetrv2",
        "input_stem",
    ),
    "E1b": CandidateSpec(
        "E1b",
        "D-FINE-S",
        "detector",
        "D-FINE official",
        "https://github.com/Peterande/D-FINE",
        "https://github.com/Peterande/D-FINE#model-zoo",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "COCO 2017",
        "Apache-2.0",
        "Apache-2.0 repository checkpoint; COCO terms also apply",
        "configs/dfine/dfine_hgnetv2_s_coco.yml",
        ("torch", "torchvision"),
        "dfine_official",
        "input_stem",
    ),
    "E1c": CandidateSpec(
        "E1c",
        "D-FINE-M",
        "detector",
        "D-FINE official",
        "https://github.com/Peterande/D-FINE",
        "https://github.com/Peterande/D-FINE#model-zoo",
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
        "COCO 2017",
        "Apache-2.0",
        "Apache-2.0 repository checkpoint; COCO terms also apply",
        "configs/dfine/dfine_hgnetv2_m_coco.yml",
        ("torch", "torchvision"),
        "dfine_official",
        "input_stem",
    ),
    "E1d": CandidateSpec(
        "E1d",
        "DINO 4-scale Swin-L",
        "detector",
        "DINO official",
        "https://github.com/IDEA-Research/DINO",
        "https://github.com/IDEA-Research/DINO#model-zoo",
        "official DINO Google Drive model-zoo checkpoint0011_4scale_swin.pth",
        "ImageNet-22K backbone, COCO 2017 detector",
        "Apache-2.0",
        "Apache-2.0 repository artifacts; ImageNet and COCO terms also apply",
        "config/DINO/DINO_4scale_swin.py",
        ("torch", "torchvision"),
        "external_detector_repo",
        "patch_projection",
    ),
    "E1e": CandidateSpec(
        "E1e",
        "Co-DINO ResNet-50",
        "detector",
        "Co-DETR official MMDetection 2.25.3 fork",
        "https://github.com/Sense-X/Co-DETR",
        "https://github.com/Sense-X/Co-DETR#co-detr-with-resnet-50",
        "official Co-DETR Google Drive Co-DINO R50 checkpoint folder",
        "ImageNet-1K backbone, COCO 2017 detector",
        "MIT",
        "MIT repository artifacts; ImageNet and COCO terms also apply",
        "projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py",
        ("torch", "mmcv"),
        "external_detector_repo_legacy_mmdet",
        "input_stem",
    ),
    "E1f": CandidateSpec(
        "E1f",
        "InternImage-T Mask R-CNN FPN",
        "detector",
        "InternImage official detection stack",
        "https://github.com/OpenGVLab/InternImage",
        "https://github.com/OpenGVLab/InternImage/blob/master/detection/README.md",
        "official mask_rcnn_internimage_t_fpn_1x_coco.pth",
        "ImageNet-1K backbone, COCO 2017 detector",
        "MIT",
        "MIT repository artifacts; ImageNet and COCO terms also apply",
        "detection/configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py",
        ("torch", "mmcv"),
        "external_detector_repo_legacy_mmdet",
        "input_stem",
    ),
    "E1g": CandidateSpec(
        "E1g",
        "Swin-T Mask R-CNN FPN",
        "detector",
        "Swin object-detection official MMDetection fork",
        "https://github.com/SwinTransformer/Swin-Transformer-Object-Detection",
        "https://github.com/SwinTransformer/Swin-Transformer-Object-Detection#main-results-on-coco-with-pretrained-models",
        "official Swin-T Mask R-CNN 3x checkpoint",
        "ImageNet-1K backbone, COCO 2017 detector",
        "Apache-2.0",
        "Apache-2.0 repository artifacts; ImageNet and COCO terms also apply",
        "configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py",
        ("torch", "mmcv"),
        "external_detector_repo_legacy_mmdet",
        "patch_projection",
    ),
    "E1h": CandidateSpec(
        "E1h",
        "FocalNet-T DINO",
        "detector",
        "FocalNet-DINO official",
        "https://github.com/FocalNet/FocalNet-DINO",
        "https://github.com/microsoft/FocalNet#object-detection",
        "official FocalNet-DINO checkpoint listed by the repository",
        "ImageNet-22K and Objects365 backbone/detector pretraining, COCO fine-tuning",
        "MIT",
        "MIT repository artifacts; ImageNet, Objects365, and COCO terms also apply",
        "config/DINO/DINO_4scale_focalnet_tiny.py",
        ("torch", "torchvision"),
        "external_detector_repo",
        "patch_projection",
    ),
    "E1i": CandidateSpec(
        "E1i",
        "ConvNeXt V2-T FCMAE transfer",
        "pretraining_transfer",
        "timm backbone plus verified project detector",
        "https://github.com/facebookresearch/ConvNeXt-V2",
        "https://github.com/facebookresearch/ConvNeXt-V2#pretrained-models",
        "convnextv2_tiny.fcmae_ft_in1k",
        "ImageNet-1K FCMAE pretraining and fine-tuning",
        "MIT",
        "CC-BY-NC for official ImageNet pretrained/fine-tuned weights",
        "timm model convnextv2_tiny.fcmae_ft_in1k plus CR1-compatible detector config",
        ("torch", "timm"),
        "timm_backbone_transfer",
        "patch_projection",
    ),
    "E1j": CandidateSpec(
        "E1j",
        "DINOv2 ViT-S/14 transfer",
        "pretraining_transfer",
        "timm backbone plus verified project detector",
        "https://github.com/facebookresearch/dinov2",
        "https://github.com/facebookresearch/dinov2#pretrained-models",
        "vit_small_patch14_dinov2.lvd142m",
        "LVD-142M self-supervised pretraining",
        "Apache-2.0",
        "Apache-2.0 for standard DINOv2 weights in the official model card",
        "timm model vit_small_patch14_dinov2.lvd142m plus CR1-compatible detector config",
        ("torch", "timm"),
        "timm_backbone_transfer",
        "patch_projection",
    ),
}


def get_candidate(experiment_id: str):
    try:
        return CANDIDATES[experiment_id]
    except KeyError as exc:
        raise KeyError(f"Unknown architecture experiment: {experiment_id}") from exc


def validate_catalog():
    required = {"CR1", "DF1", "DF2", "DF3", *(f"E1{x}" for x in "abcdefghij")}
    errors = []
    if set(CANDIDATES) != required:
        errors.append(f"catalog IDs differ: {sorted(set(CANDIDATES) ^ required)}")
    for key, spec in CANDIDATES.items():
        if key != spec.experiment_id:
            errors.append(f"{key}: experiment_id mismatch")
        for field in (
            "name",
            "role",
            "framework",
            "checkpoint",
            "pretrained_dataset",
            "code_license",
            "weights_license",
            "config",
            "integration",
        ):
            if not getattr(spec, field).strip():
                errors.append(f"{key}: missing {field}")
        for field in ("source_url", "evidence_url"):
            parsed = urlparse(getattr(spec, field))
            if parsed.scheme != "https" or not parsed.netloc:
                errors.append(f"{key}: invalid {field}")
        if spec.input_layer_kind not in {"input_stem", "patch_projection"}:
            errors.append(f"{key}: invalid input_layer_kind")
    return errors
