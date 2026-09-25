import argparse
import importlib.metadata
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .catalog import get_candidate
from .loaders import (
    DependencyUnavailable,
    IncompatibleArchitecture,
    load_dfine,
    load_mmdetection,
    load_rtdetrv2,
    load_timm_transfer,
)


@dataclass
class SmokeReport:
    experiment_id: str
    status: str
    detail: str
    checks: dict
    elapsed_seconds: float
    peak_gpu_memory_mb: float | None
    device: str
    dependency_versions: dict
    adaptation: dict | None
    metadata: dict

    def to_dict(self):
        return asdict(self)


def _tensor_values(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_values(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensor_values(child)
    elif hasattr(value, "to_tuple"):
        yield from _tensor_values(value.to_tuple())


def _finite_output(value):
    tensors = list(_tensor_values(value))
    return bool(tensors) and all(torch.isfinite(tensor).all().item() for tensor in tensors)


def _prediction_contract(value):
    if hasattr(value, "logits") and hasattr(value, "pred_boxes"):
        logits, boxes = value.logits, value.pred_boxes
    elif isinstance(value, dict) and {"pred_logits", "pred_boxes"} <= set(value):
        logits, boxes = value["pred_logits"], value["pred_boxes"]
    else:
        return False, "Output does not expose detector logits and boxes"
    if logits.ndim != 3 or boxes.ndim != 3 or boxes.shape[-1] != 4:
        return False, "Detector output shapes are invalid"
    if logits.shape[:2] != boxes.shape[:2]:
        return False, "Detector query dimensions disagree"
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        return False, "Detector output contains non-finite values"
    if ((boxes < 0) | (boxes > 1)).any():
        return False, "Normalized detector boxes are outside [0, 1]"
    return True, "Finite logits and normalized boxes"


def _generic_detector_step(model, device, shape, num_classes=18):
    model.to(device)
    pixels = torch.rand(shape, device=device)
    model.eval()
    with torch.no_grad():
        output = model(pixel_values=pixels)
    prediction_ok, prediction_detail = _prediction_contract(output)
    model.train()
    labels = [
        {
            "class_labels": torch.tensor([0], device=device),
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=device),
        }
        for _ in range(shape[0])
    ]
    trained = model(pixel_values=pixels, labels=labels)
    loss = trained.loss
    finite_loss = bool(torch.isfinite(loss).item())
    loss.backward()
    finite_gradients = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "forward_finite": _finite_output(output),
        "prediction_contract": prediction_ok,
        "prediction_detail": prediction_detail,
        "training_loss_finite": finite_loss,
        "backward_gradient_finite": finite_gradients,
        "num_classes": num_classes,
    }


def _dfine_step(model, criterion, device, shape):
    model.to(device)
    criterion.to(device)
    pixels = torch.rand(shape, device=device)
    model.eval()
    with torch.no_grad():
        prediction = model(pixels)
    prediction_ok, prediction_detail = _prediction_contract(prediction)
    targets = [
        {
            "labels": torch.tensor([0], device=device),
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=device),
        }
        for _ in range(shape[0])
    ]
    model.train()
    outputs = model(pixels, targets)
    losses = criterion(outputs, targets)
    loss = sum(losses.values())
    finite_loss = bool(torch.isfinite(loss).item()) and bool(losses)
    loss.backward()
    finite_gradients = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "forward_finite": _finite_output(prediction),
        "prediction_contract": prediction_ok,
        "prediction_detail": prediction_detail,
        "training_loss_finite": finite_loss,
        "backward_gradient_finite": finite_gradients,
        "loss_keys": sorted(losses),
    }


def _mmdet_step(model, device, shape):
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    model.to(device)
    pixels = torch.rand(shape, device=device)
    samples = []
    for _ in range(shape[0]):
        sample = DetDataSample()
        sample.set_metainfo(
            {
                "img_shape": shape[-2:],
                "ori_shape": shape[-2:],
                "batch_input_shape": shape[-2:],
                "scale_factor": (1.0, 1.0),
            }
        )
        ground_truth = InstanceData()
        ground_truth.bboxes = torch.tensor([[8.0, 8.0, 32.0, 32.0]], device=device)
        ground_truth.labels = torch.tensor([0], dtype=torch.long, device=device)
        sample.gt_instances = ground_truth
        samples.append(sample)
    model.eval()
    with torch.no_grad():
        predictions = model(pixels, samples, mode="predict")
    prediction_ok = bool(predictions) and all(
        hasattr(sample, "pred_instances")
        and torch.isfinite(sample.pred_instances.bboxes).all().item()
        and torch.isfinite(sample.pred_instances.scores).all().item()
        for sample in predictions
    )
    model.train()
    losses = model(pixels, samples, mode="loss")
    tensor_losses = [value for value in _tensor_values(losses) if value.ndim == 0]
    loss = sum(tensor_losses)
    finite_loss = bool(tensor_losses) and bool(torch.isfinite(loss).item())
    loss.backward()
    finite_gradients = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "forward_finite": prediction_ok,
        "prediction_contract": prediction_ok,
        "prediction_detail": "MMDetection DetDataSample outputs" if prediction_ok else "Invalid MMDetection predictions",
        "training_loss_finite": finite_loss,
        "backward_gradient_finite": finite_gradients,
        "loss_keys": sorted(losses),
    }


def _versions(modules):
    result = {}
    for name in modules:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run_smoke(
    experiment_id,
    strategy="rgb_mean",
    device="cpu",
    repo_root=None,
    checkpoint=None,
    image_size=64,
):
    spec = get_candidate(experiment_id)
    start = time.perf_counter()
    checks = {}
    adaptation = None
    detail = ""
    status = "implementation_failure"
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    try:
        if spec.integration == "transformers_rtdetrv2":
            model, record = load_rtdetrv2(checkpoint or spec.checkpoint, strategy)
            adaptation = record.to_dict()
            checks = _generic_detector_step(model, device, (1, 16, image_size, image_size))
        elif spec.integration == "dfine_official":
            if not repo_root:
                raise DependencyUnavailable("D-FINE source root is required")
            model, criterion, record, audit = load_dfine(
                experiment_id,
                repo_root,
                checkpoint,
                strategy,
            )
            adaptation = record.to_dict()
            checks = _dfine_step(model, criterion, device, (1, 16, image_size, image_size))
            checks.update(audit)
        elif spec.integration == "mmdetection":
            if not repo_root:
                raise DependencyUnavailable("MMDetection source root is required")
            model, record = load_mmdetection(experiment_id, repo_root, checkpoint, strategy)
            adaptation = record.to_dict()
            checks = _mmdet_step(model, device, (1, 16, image_size, image_size))
        elif spec.integration == "timm_backbone_transfer":
            model, record = load_timm_transfer(experiment_id, strategy, pretrained=True)
            adaptation = record.to_dict()
            model.to(device).train()
            pixels = torch.rand((1, 16, image_size, image_size), device=device)
            output = model(pixels)
            loss = sum(value.float().mean() for value in _tensor_values(output))
            loss.backward()
            checks = {
                "backbone_forward_finite": _finite_output(output),
                "backbone_backward_finite": bool(torch.isfinite(loss).item()),
                "prediction_contract": False,
            }
            detail = "Backbone transfer is finite but lacks the required object-detection neck/head and predictions"
        elif spec.integration == "df3_isolated_only":
            raise IncompatibleArchitecture("DF3 needs a project RT-DETR parent and an explicitly isolated regression diff")
        else:
            raise DependencyUnavailable(
                f"{spec.name} must run from its official external detector repository using {spec.config}"
            )
        required = (
            checks.get("forward_finite", False),
            checks.get("prediction_contract", False),
            checks.get("training_loss_finite", False),
            checks.get("backward_gradient_finite", False),
        )
        if all(required):
            status = "passed"
            detail = "Finite 16-band detector forward, prediction, loss, and backward checks passed"
        elif not detail:
            detail = "One or more detector smoke checks failed"
    except IncompatibleArchitecture as exc:
        status = "incompatible"
        detail = str(exc)
    except (DependencyUnavailable, FileNotFoundError) as exc:
        status = "implementation_failure"
        detail = str(exc)
    except Exception as exc:
        status = "implementation_failure"
        detail = f"{type(exc).__name__}: {exc}"
        checks["traceback"] = traceback.format_exc()
    elapsed = time.perf_counter() - start
    peak = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    return SmokeReport(
        experiment_id,
        status,
        detail,
        checks,
        elapsed,
        peak,
        device,
        _versions(spec.required_modules),
        adaptation,
        spec.to_dict(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--strategy", choices=("random", "rgb_mean"), default="rgb_mean")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repo-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_smoke(
        args.candidate,
        args.strategy,
        args.device,
        args.repo_root,
        args.checkpoint,
        args.image_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    raise SystemExit(0 if report.status == "passed" else 2)


if __name__ == "__main__":
    main()
