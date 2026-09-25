# Original-coordinate detection evaluation

`evaluate_size_ap.py` consumes a COCO ground-truth dictionary and COCO detection dictionaries with original image pixel XYWH boxes. Image IDs and category IDs must match. Pass every evaluation image, including images with no objects or predictions. The caller must undo all image resizing and padding first. The evaluator never rescales boxes and cannot infer whether the caller supplied normalized coordinates.

Box area is original width × height. Supplied annotation areas are replaced in an internal copy because segmentation area or stale resized area would violate this roadmap's box-area definition. The report records how many supplied areas differ. Official source files are never changed.

Two tables are produced using the installed official `pycocotools.COCOeval` bbox evaluator:

- `roadmap_strict`: small `<1024`, medium `>=1024 and <9216`, large `>=9216`.
- `coco_standard`: the official evaluator's inclusive size intervals. Exactly 1024 belongs to both small and medium; exactly 9216 belongs to both medium and large. This table preserves that behavior, with original bbox-derived GT areas.

The strict table uses the immediately preceding floating-point value for exclusive upper bounds. Matching, ignored out-of-size predictions, crowd handling, IoU thresholds 0.50:0.05:0.95, 101-point interpolated AP and class averaging otherwise remain COCO behavior. The standard table retains COCO's 1e10 upper area cap. The strict table has no finite upper cap.

Each table reports AP50-95, AP50, AP75, AR50-95, recall50 and recall75 by size and by class × size. AR50-95 is recall averaged across the ten IoU thresholds at the stated maximum detection count. Unsupported classes or sizes return JSON null, never an invented zero. Actual missed objects return zero. Raw per-box areas and a histogram accompany the tables.

The default COCO maximum is 100 detections per image per category. It is configurable and recorded explicitly. No extra confidence cutoff is applied for AP. Saved predictions may already have been filtered; preserve their inference configuration beside the report.

`detection_confusion` is an independent diagnostic. Both confidence and IoU thresholds are required. It greedily matches descending-confidence predictions to the highest-IoU unmatched GT regardless of class, so real/plastic mistakes are visible. Rows are true classes; columns are predicted classes; the final row and column represent background. Incorrect classes count as a false negative for the true class and a false positive for the predicted class. This matrix is not the COCO AP matching procedure. Crowd annotations and non-crowd ignored annotations are rejected for this diagnostic to prevent an ambiguous count.

Example:

```sh
python Experiments/evaluation/evaluate_size_ap.py \
  --ground-truth original_gt.json \
  --predictions original_predictions.json \
  --output size_diagnostics.json \
  --max-dets 100 \
  --confusion-confidence 0.25 \
  --confusion-iou 0.5
```

The example confusion thresholds only illustrate the API. Freeze thresholds from validation before holdout; they are not a selected competition policy.

Source checked 2026-09-19: [official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py), specifically `evaluateImg`, `accumulate`, and `Params.setDetParams`. The report records the installed pycocotools version. Synthetic unit tests establish implementation behavior only. They do not establish competition AP or complete the D1 checkpoint evaluations.
