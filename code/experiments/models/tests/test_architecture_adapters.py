import unittest

import torch
from torch import nn

from Experiments.models.adaptation import adapt_input_projection
from Experiments.models.catalog import CANDIDATES, validate_catalog
from Experiments.models.dfine import assess_df3_isolation, verify_full_dfine
from Experiments.models.smoke import _prediction_contract, _generic_detector_step


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.logits = nn.Linear(8, 4)
        self.boxes = nn.Linear(8, 4)

    def forward(self, pixel_values, labels=None):
        features = self.pool(torch.relu(self.stem(pixel_values))).flatten(1)
        logits = self.logits(features).unsqueeze(1)
        boxes = torch.sigmoid(self.boxes(features)).unsqueeze(1)
        result = type("Output", (), {})()
        result.logits = logits
        result.pred_boxes = boxes
        result.loss = logits.square().mean() + boxes.square().mean() if labels is not None else None
        result.to_tuple = lambda: (logits, boxes)
        return result


class FakeDFINEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.integral = nn.Identity()
        self.dec_bbox_head = nn.ModuleList([nn.Linear(1, 1)])


FakeDFINEDecoder.__name__ = "DFINETransformer"


class FakeDFINE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = FakeDFINEDecoder()


FakeDFINE.__name__ = "DFINE"


class FakeCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = ["vfl", "boxes", "local"]
        self.weight_dict = {"loss_fgl": 0.15, "loss_ddf": 1.5}


FakeCriterion.__name__ = "DFINECriterion"


class ArchitectureAdapterTests(unittest.TestCase):
    def test_catalog_covers_every_required_architecture(self):
        expected = {"CR1", "DF1", "DF2", "DF3", *(f"E1{x}" for x in "abcdefghij")}
        self.assertEqual(set(CANDIDATES), expected)
        self.assertEqual(validate_catalog(), [])

    def test_rgb_mean_expansion_matches_scale_corrected_rgb_response(self):
        torch.manual_seed(4)
        model = TinyDetector()
        old_weight = model.stem.weight.detach().clone()
        old_bias = model.stem.bias.detach().clone()
        record = adapt_input_projection(model, 16, "rgb_mean")
        expected = old_weight.mean(1, keepdim=True).repeat(1, 16, 1, 1) * (3 / 16)
        self.assertTrue(torch.equal(model.stem.weight, expected))
        self.assertTrue(torch.equal(model.stem.bias, old_bias))
        self.assertEqual(record.scale_correction, 3 / 16)
        rgb_gray = torch.rand(2, 1, 9, 9).repeat(1, 3, 1, 1)
        hsi_gray = rgb_gray[:, :1].repeat(1, 16, 1, 1)
        source = nn.functional.conv2d(rgb_gray, old_weight, old_bias, padding=1)
        target = model.stem(hsi_gray)
        self.assertTrue(torch.allclose(source, target, atol=1e-6))

    def test_random_expansion_is_finite_and_not_rgb_copy(self):
        torch.manual_seed(7)
        model = TinyDetector()
        old_mean = model.stem.weight.detach().mean(1, keepdim=True).repeat(1, 16, 1, 1) * (3 / 16)
        record = adapt_input_projection(model, 16, "random")
        self.assertTrue(torch.isfinite(model.stem.weight).all())
        self.assertFalse(torch.equal(model.stem.weight, old_mean))
        self.assertIsNone(record.scale_correction)

    def test_patch_projection_is_named_truthfully(self):
        model = TinyDetector()
        record = adapt_input_projection(model, 16, "rgb_mean", "stem", "patch_projection")
        self.assertEqual(record.layer_kind, "patch_projection")
        self.assertEqual(model.stem.in_channels, 16)

    def test_finite_detector_forward_and_one_step_training(self):
        model = TinyDetector()
        adapt_input_projection(model, 16, "rgb_mean")
        checks = _generic_detector_step(model, "cpu", (2, 16, 16, 16))
        self.assertTrue(checks["forward_finite"])
        self.assertTrue(checks["prediction_contract"])
        self.assertTrue(checks["training_loss_finite"])
        self.assertTrue(checks["backward_gradient_finite"])

    def test_invalid_detector_boxes_fail_prediction_contract(self):
        output = {"pred_logits": torch.zeros(1, 2, 3), "pred_boxes": torch.full((1, 2, 4), 2.0)}
        passed, detail = _prediction_contract(output)
        self.assertFalse(passed)
        self.assertIn("outside", detail)

    def test_full_dfine_requires_fdr_and_go_lsd(self):
        result = verify_full_dfine(FakeDFINE(), FakeCriterion())
        self.assertTrue(result["passed"])
        criterion = FakeCriterion()
        criterion.weight_dict.pop("loss_ddf")
        self.assertFalse(verify_full_dfine(FakeDFINE(), criterion)["passed"])

    def test_df3_refuses_full_architecture_swap(self):
        result = assess_df3_isolation(
            "RT-DETR",
            "D-FINE",
            {"backbone", "encoder", "decoder", "regression_head", "regression_loss"},
        )
        self.assertFalse(result.compatible)
        self.assertEqual(result.status, "incompatible")
        self.assertIn("architecture swap", result.reason)

    def test_df3_accepts_only_isolated_regression_change(self):
        result = assess_df3_isolation(
            "RT-DETR",
            "RT-DETR",
            {"regression_head", "regression_loss"},
        )
        self.assertTrue(result.compatible)
        self.assertEqual(result.status, "feasible")


if __name__ == "__main__":
    unittest.main()
