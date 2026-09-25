import importlib
import importlib.util
import sys
from pathlib import Path

import torch

from .adaptation import adapt_input_projection
from .catalog import get_candidate
from .dfine import verify_full_dfine


class DependencyUnavailable(RuntimeError):
    pass


class IncompatibleArchitecture(RuntimeError):
    pass


def missing_modules(names):
    return [name for name in names if importlib.util.find_spec(name) is None]


def _matching_state_dict(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "ema" in state:
        state = state["ema"]
        if isinstance(state, dict) and "module" in state:
            state = state["module"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    return state


def load_dfine(experiment_id, repo_root, checkpoint, strategy="rgb_mean", in_channels=16):
    if experiment_id not in {"DF1", "DF2", "E1b", "E1c"}:
        raise ValueError(f"Not a full D-FINE experiment: {experiment_id}")
    spec = get_candidate(experiment_id)
    root = Path(repo_root).resolve()
    config_path = root / spec.config
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    YAMLConfig = importlib.import_module("src.core").YAMLConfig
    cfg = YAMLConfig(str(config_path))
    model = cfg.model
    criterion = cfg.criterion
    if checkpoint:
        state = _matching_state_dict(checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        checkpoint_load = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    else:
        checkpoint_load = None
    adaptation = adapt_input_projection(model, in_channels, strategy, layer_kind="input_stem")
    integrity = verify_full_dfine(model, criterion)
    if not integrity["passed"]:
        raise IncompatibleArchitecture(f"Official full D-FINE integrity checks failed: {integrity['checks']}")
    return model, criterion, adaptation, {"integrity": integrity, "checkpoint_load": checkpoint_load}


def load_rtdetrv2(checkpoint="PekingU/rtdetr_v2_r18vd", strategy="rgb_mean", in_channels=16):
    missing = missing_modules(("transformers",))
    if missing:
        raise DependencyUnavailable(f"Missing dependencies: {', '.join(missing)}")
    from transformers import RTDetrV2ForObjectDetection

    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint)
    adaptation = adapt_input_projection(model, in_channels, strategy, layer_kind="input_stem")
    if hasattr(model.config, "backbone_config"):
        model.config.backbone_config.num_channels = in_channels
    backbone = model.model.backbone.model
    if hasattr(backbone, "config") and hasattr(backbone.config, "num_channels"):
        backbone.config.num_channels = in_channels
    if hasattr(backbone, "embedder") and hasattr(backbone.embedder, "num_channels"):
        backbone.embedder.num_channels = in_channels
    return model, adaptation


def load_timm_transfer(experiment_id, strategy="rgb_mean", in_channels=16, pretrained=True):
    if experiment_id not in {"E1i", "E1j"}:
        raise ValueError(f"Not a timm transfer experiment: {experiment_id}")
    missing = missing_modules(("timm",))
    if missing:
        raise DependencyUnavailable(f"Missing dependencies: {', '.join(missing)}")
    import timm

    spec = get_candidate(experiment_id)
    model = timm.create_model(spec.checkpoint, pretrained=pretrained)
    adaptation = adapt_input_projection(
        model,
        in_channels,
        strategy,
        layer_kind=spec.input_layer_kind,
    )
    return model, adaptation


def load_mmdetection(experiment_id, repo_root, checkpoint=None, strategy="rgb_mean", in_channels=16):
    spec = get_candidate(experiment_id)
    if spec.integration != "mmdetection":
        raise ValueError(f"Not a current MMDetection integration: {experiment_id}")
    missing = missing_modules(spec.required_modules)
    if missing:
        raise DependencyUnavailable(f"Missing dependencies: {', '.join(missing)}")
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS

    config_path = Path(repo_root).resolve() / spec.config
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.init_cfg = None
    init_default_scope("mmdet")
    model = MODELS.build(cfg.model)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    adaptation = adapt_input_projection(model, in_channels, strategy, layer_kind="input_stem")
    return model, adaptation
