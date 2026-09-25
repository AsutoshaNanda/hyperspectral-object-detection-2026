from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AdaptationRecord:
    module_path: str
    layer_kind: str
    strategy: str
    old_in_channels: int
    new_in_channels: int
    scale_correction: float | None

    def to_dict(self):
        return asdict(self)


def _resolve_parent(root: nn.Module, path: str):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, name: str, child: nn.Module):
    if name.isdigit():
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def find_input_projection(model: nn.Module, module_path: str | None = None):
    if module_path:
        modules = dict(model.named_modules())
        if module_path not in modules:
            raise ValueError(f"Input projection path does not exist: {module_path}")
        module = modules[module_path]
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"Input projection must be Conv2d, got {type(module).__name__}")
        if module.in_channels != 3:
            raise ValueError(f"Input projection must have 3 channels, got {module.in_channels}")
        return module_path, module
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            return name, module
    raise ValueError("No 3-channel Conv2d input projection was found")


def adapt_input_projection(
    model: nn.Module,
    in_channels: int = 16,
    strategy: str = "rgb_mean",
    module_path: str | None = None,
    layer_kind: str = "input_stem",
):
    if in_channels <= 0:
        raise ValueError("in_channels must be positive")
    if strategy not in {"random", "rgb_mean"}:
        raise ValueError(f"Unknown initialization strategy: {strategy}")
    if layer_kind not in {"input_stem", "patch_projection"}:
        raise ValueError(f"Unknown layer kind: {layer_kind}")
    path, old = find_input_projection(model, module_path)
    if old.groups != 1:
        raise ValueError("Grouped 3-channel projections are not supported")
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        old.kernel_size,
        old.stride,
        old.padding,
        old.dilation,
        old.groups,
        old.bias is not None,
        old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    scale = None
    with torch.no_grad():
        if strategy == "rgb_mean":
            scale = 3.0 / in_channels
            expanded = old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            new.weight.copy_(expanded * scale)
        if new.bias is not None and old.bias is not None:
            new.bias.copy_(old.bias)
    parent, child_name = _resolve_parent(model, path)
    _set_child(parent, child_name, new)
    return AdaptationRecord(path, layer_kind, strategy, old.in_channels, in_channels, scale)
