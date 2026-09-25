from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SAMEvidence:
    report_id: str
    demonstrates_separation: bool
    prototype_source_image_ids: frozenset[str]
    preprocessing_report_id: str | None = None

    @classmethod
    def create(
        cls,
        report_id,
        demonstrates_separation,
        prototype_source_image_ids,
        preprocessing_report_id=None,
    ):
        sources = frozenset(str(value) for value in prototype_source_image_ids)
        if not report_id or not sources:
            raise ValueError("SAM1 report id and prototype training sources are required")
        return cls(str(report_id), bool(demonstrates_separation), sources, preprocessing_report_id)


class SAMDetectorClassificationHead(nn.Module):
    def __init__(self, roi_feature_dim, prototypes, num_classes, *, spectral_preprocessor=None):
        super().__init__()
        prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        if prototypes.ndim != 2 or prototypes.shape[0] < 2 or prototypes.shape[1] < 2:
            raise ValueError("prototypes must be a [classes, spectral_bands] matrix")
        if roi_feature_dim <= 0 or num_classes <= 1:
            raise ValueError("roi_feature_dim and num_classes are invalid")
        self.register_buffer("prototypes", prototypes)
        self.spectral_preprocessor = spectral_preprocessor or nn.Identity()
        self.classifier = nn.Linear(roi_feature_dim + prototypes.shape[0], num_classes)

    def sam_distances(self, roi_spectra):
        spectra = self.spectral_preprocessor(roi_spectra)
        if spectra.ndim != 2 or spectra.shape[1] != self.prototypes.shape[1]:
            raise ValueError("roi_spectra must match the prototype spectral dimension")
        spectra = spectra.to(device=self.prototypes.device, dtype=self.prototypes.dtype)
        numerator = spectra @ self.prototypes.T
        denominator = torch.linalg.vector_norm(spectra, dim=1, keepdim=True) * torch.linalg.vector_norm(
            self.prototypes, dim=1
        ).unsqueeze(0)
        cosine = numerator / denominator.clamp_min(torch.finfo(spectra.dtype).eps)
        return torch.acos(cosine.clamp(-1.0, 1.0))

    def forward(self, roi_features, roi_spectra, *, independent_classifier_logits=None):
        if independent_classifier_logits is not None:
            raise ValueError("Independent classifier fusion is prohibited; SAM must stay inside the detector branch")
        if roi_features.ndim != 2 or roi_features.shape[0] != roi_spectra.shape[0]:
            raise ValueError("roi_features and roi_spectra must share a batch dimension")
        return self.classifier(torch.cat((roi_features, self.sam_distances(roi_spectra)), dim=1))


def build_sam_auxiliary_head(
    experiment_id,
    roi_feature_dim,
    prototypes,
    num_classes,
    evidence,
    *,
    allowed_training_image_ids,
    forbidden_image_ids=(),
    spectral_preprocessor=None,
    fusion_mode="detector_classification_branch",
):
    if experiment_id not in {"SAM2", "SAM3"}:
        raise ValueError("experiment_id must be SAM2 or SAM3")
    if fusion_mode != "detector_classification_branch":
        raise ValueError("SAM auxiliary features must be fused inside one detector classification branch")
    if not isinstance(evidence, SAMEvidence) or not evidence.demonstrates_separation:
        raise ValueError("SAM2 requires measured SAM1 separation evidence")
    allowed = {str(value) for value in allowed_training_image_ids}
    forbidden = {str(value) for value in forbidden_image_ids}
    if not evidence.prototype_source_image_ids <= allowed:
        raise ValueError("SAM prototypes were not derived only from the declared training fold")
    if evidence.prototype_source_image_ids & forbidden:
        raise ValueError("SAM prototypes include validation/holdout/test data")
    if experiment_id == "SAM3" and (not evidence.preprocessing_report_id or spectral_preprocessor is None):
        raise ValueError("SAM3 requires winning spectral-preprocessing evidence and its preprocessing module")
    return SAMDetectorClassificationHead(
        roi_feature_dim,
        prototypes,
        num_classes,
        spectral_preprocessor=spectral_preprocessor,
    )
