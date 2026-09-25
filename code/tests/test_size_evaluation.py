import copy
import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "Experiments" / "evaluation" / "evaluate_size_ap.py"
SPEC = importlib.util.spec_from_file_location("size_evaluation", MODULE_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def fixture(boxes=None):
    boxes = boxes or [[5, 5, 10, 10], [30, 30, 32, 32], [100, 100, 96, 96]]
    gt = {"images": [{"id": 1, "width": 500, "height": 300}],
          "categories": [{"id": 3, "name": "apple"}, {"id": 8, "name": "plastic"}],
          "annotations": [{"id": i + 10, "image_id": 1, "category_id": 3, "bbox": box,
                           "area": box[2] * box[3], "iscrowd": 0} for i, box in enumerate(boxes)]}
    predictions = [{"image_id": 1, "category_id": 3, "bbox": box[:], "score": 0.9} for box in boxes]
    return gt, predictions


class ConfusionTests(unittest.TestCase):
    def test_misclassification_duplicate_and_missed_box(self):
        gt, predictions = fixture()
        predictions[0]["category_id"] = 8
        predictions = predictions[:2] + [copy.deepcopy(predictions[1])]
        report = EVALUATOR.detection_confusion(gt, predictions, confidence_threshold=0.5, iou_threshold=0.5)
        self.assertEqual(report["matrix"], [[1, 1, 1], [0, 0, 0], [1, 0, 0]])
        apple, plastic = report["per_class"]
        self.assertEqual(apple["false_positives"], 1)
        self.assertEqual(apple["false_negatives"], 2)
        self.assertEqual(plastic["false_positives"], 1)
        self.assertIsNone(plastic["recall"])

    def test_explicit_confidence_boundary_and_no_mutation(self):
        gt, predictions = fixture()
        gt_before, predictions_before = copy.deepcopy(gt), copy.deepcopy(predictions)
        report = EVALUATOR.detection_confusion(gt, predictions, confidence_threshold=0.9, iou_threshold=1)
        self.assertEqual(report["matrix"][0][0], 3)
        self.assertEqual(gt, gt_before)
        self.assertEqual(predictions, predictions_before)

    def test_invalid_inputs_fail_before_reporting_metrics(self):
        for invalid_box in ([0, 0, 0, 10], [0, 0, float("nan"), 10], [0, 0, -1, 1]):
            gt, predictions = fixture()
            predictions[0]["bbox"] = invalid_box
            with self.assertRaises(ValueError):
                EVALUATOR.detection_confusion(gt, predictions, confidence_threshold=0.5, iou_threshold=0.5)
        gt, predictions = fixture()
        predictions[0]["image_id"] = 999
        with self.assertRaises(ValueError):
            EVALUATOR.detection_confusion(gt, predictions, confidence_threshold=0.5, iou_threshold=0.5)

    def test_crowd_confusion_rejected(self):
        gt, predictions = fixture()
        gt["annotations"][0]["iscrowd"] = 1
        with self.assertRaisesRegex(ValueError, "crowd"):
            EVALUATOR.detection_confusion(gt, predictions, confidence_threshold=0.5, iou_threshold=0.5)


@unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools is required for COCO integration tests")
class CocoSizeTests(unittest.TestCase):
    def test_perfect_predictions_and_absent_classes(self):
        gt, predictions = fixture()
        before = copy.deepcopy(gt)
        report = EVALUATOR.evaluate_size_ap(gt, predictions)
        for size in ("all", "small", "medium", "large"):
            self.assertAlmostEqual(report["roadmap_strict"]["size_metrics"][size]["AP50_95"], 1)
            self.assertAlmostEqual(report["roadmap_strict"]["size_metrics"][size]["AR50_95"], 1)
        self.assertIsNone(report["roadmap_strict"]["class_by_size"][1]["sizes"]["all"]["AP50_95"])
        self.assertEqual(sum(row["count"] for row in report["box_area_histogram"]), 3)
        self.assertEqual([row["area"] for row in report["raw_box_areas"]], [100, 1024, 9216])
        self.assertEqual(gt, before)
        json.dumps(report, allow_nan=False)
        self.assertIs(type(report["roadmap_strict"]["class_by_size"][0]["category_id"]), int)

    def test_exact_boundaries_have_non_overlapping_roadmap_sizes(self):
        gt, predictions = fixture([[0, 0, 32, 32]])
        report = EVALUATOR.evaluate_size_ap(gt, predictions)
        self.assertIsNone(report["roadmap_strict"]["size_metrics"]["small"]["AP50_95"])
        self.assertAlmostEqual(report["coco_standard"]["size_metrics"]["small"]["AP50_95"], 1)
        self.assertAlmostEqual(report["roadmap_strict"]["size_metrics"]["medium"]["AP50_95"], 1)
        gt, predictions = fixture([[0, 0, 96, 96]])
        report = EVALUATOR.evaluate_size_ap(gt, predictions)
        self.assertIsNone(report["roadmap_strict"]["size_metrics"]["medium"]["AP50_95"])
        self.assertAlmostEqual(report["coco_standard"]["size_metrics"]["medium"]["AP50_95"], 1)
        self.assertAlmostEqual(report["roadmap_strict"]["size_metrics"]["large"]["AP50_95"], 1)
        self.assertAlmostEqual(report["coco_standard"]["size_metrics"]["large"]["AP50_95"], 1)

    def test_empty_predictions_are_zero_and_absent_sizes_null(self):
        gt, _ = fixture([[0, 0, 10, 10]])
        report = EVALUATOR.evaluate_size_ap(gt, [])
        self.assertEqual(report["roadmap_strict"]["size_metrics"]["small"]["AP50_95"], 0)
        self.assertEqual(report["roadmap_strict"]["size_metrics"]["small"]["AR50_95"], 0)
        self.assertIsNone(report["roadmap_strict"]["size_metrics"]["large"]["AP50_95"])

    def test_original_bbox_area_overrides_stale_area_without_resizing(self):
        gt, predictions = fixture([[0, 0, 96, 96]])
        gt["annotations"][0]["area"] = 20
        report = EVALUATOR.evaluate_size_ap(gt, predictions)
        self.assertEqual(report["supplied_gt_areas_replaced"], 1)
        self.assertEqual(report["raw_box_areas"][0]["area"], 9216)
        self.assertIsNone(report["roadmap_strict"]["size_metrics"]["small"]["AP50_95"])

    def test_max_detections_is_explicit_and_effective(self):
        gt, predictions = fixture([[0, 0, 10, 10], [30, 30, 10, 10]])
        report = EVALUATOR.evaluate_size_ap(gt, predictions, max_dets=1)
        self.assertEqual(report["max_detections_per_image_per_class"], 1)
        self.assertAlmostEqual(report["roadmap_strict"]["size_metrics"]["all"]["AR50_95"], 0.5)

    def test_background_image_false_positive_counts(self):
        gt, predictions = fixture([[0, 0, 10, 10]])
        gt["images"].append({"id": 2, "width": 500, "height": 300})
        predictions.insert(0, {"image_id": 2, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.99})
        report = EVALUATOR.evaluate_size_ap(gt, predictions)
        self.assertAlmostEqual(report["roadmap_strict"]["size_metrics"]["all"]["AP50_95"], 0.5)


if __name__ == "__main__":
    unittest.main()
