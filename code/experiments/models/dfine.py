from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DF3Assessment:
    compatible: bool
    status: str
    reason: str
    changed_components: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def verify_full_dfine(model, criterion):
    decoder = getattr(model, "decoder", None)
    decoder_type = type(decoder).__name__
    criterion_type = type(criterion).__name__
    losses = set(getattr(criterion, "losses", ()))
    weights = set(getattr(criterion, "weight_dict", {}))
    checks = {
        "model_type_dfine": type(model).__name__ == "DFINE",
        "decoder_type_dfine_transformer": decoder_type == "DFINETransformer",
        "fdr_distribution_head": hasattr(decoder, "integral") and hasattr(decoder, "dec_bbox_head"),
        "criterion_type_dfine": criterion_type == "DFINECriterion",
        "fine_grained_localization_loss": "local" in losses and "loss_fgl" in weights,
        "go_lsd_distillation_loss": "local" in losses and "loss_ddf" in weights,
    }
    return {"passed": all(checks.values()), "checks": checks}


def assess_df3_isolation(parent_family: str, proposed_family: str, changed_components):
    changed = tuple(sorted(set(changed_components)))
    if parent_family != "RT-DETR":
        return DF3Assessment(False, "incompatible", "DF3 requires an RT-DETR baseline parent", changed)
    if proposed_family != parent_family:
        return DF3Assessment(
            False,
            "incompatible",
            "A family change is an architecture swap and cannot be called isolated FDR",
            changed,
        )
    allowed = {"regression_head", "regression_loss"}
    disallowed = sorted(set(changed) - allowed)
    if disallowed:
        return DF3Assessment(
            False,
            "incompatible",
            f"DF3 changes unrelated components: {', '.join(disallowed)}",
            changed,
        )
    if "regression_head" not in changed:
        return DF3Assessment(False, "implementation_failure", "No FDR-style regression head change was supplied", changed)
    return DF3Assessment(True, "feasible", "The proposal retains the RT-DETR family and isolates regression", changed)
