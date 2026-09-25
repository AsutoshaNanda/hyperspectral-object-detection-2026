"""Score detections with the SAME mAP code Ultralytics uses (so the number is comparable to Combo's 0.672).

Usage (run with /venv/main python):
  python eval_ultra.py <gt_coco.json> <preds_coco_results.json>
Preds: COCO results list [{image_id, category_id, bbox[x,y,w,h], score}], same image ids as gt.
Matching replicates ultralytics DetectionValidator.match_predictions (10 IoU thresholds 0.50:0.95).
"""
import json
import sys
from collections import defaultdict

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

iouv = torch.linspace(0.5, 0.95, 10)


def match(pred_cls, true_cls, iou):
    correct = np.zeros((pred_cls.shape[0], iouv.shape[0]), dtype=bool)
    correct_class = true_cls[:, None] == pred_cls
    iou = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def xywh2xyxy(b):
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    return np.c_[b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]]


def score(gt_path, pred_path, max_det=300):
    gt = json.load(open(gt_path))
    preds = json.load(open(pred_path))
    gts, pds = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        gts[a["image_id"]].append((a["category_id"], *a["bbox"]))
    for p in preds:
        pds[p["image_id"]].append((p["score"], p["category_id"], *p["bbox"]))
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    for img in gt["images"]:
        g = np.array(gts[img["id"]], dtype=np.float32).reshape(-1, 5)
        p = sorted(pds[img["id"]], key=lambda r: -r[0])[:max_det]
        p = np.array(p, dtype=np.float32).reshape(-1, 6)
        stats["target_cls"].append(g[:, 0])
        if len(p) == 0:
            continue
        stats["conf"].append(p[:, 0])
        stats["pred_cls"].append(p[:, 1])
        if len(g):
            iou = box_iou(torch.from_numpy(xywh2xyxy(g[:, 1:])), torch.from_numpy(xywh2xyxy(p[:, 2:])))
            stats["tp"].append(match(torch.from_numpy(p[:, 1]), torch.from_numpy(g[:, 0]), iou))
        else:
            stats["tp"].append(np.zeros((len(p), 10), dtype=bool))
    tp = np.concatenate(stats["tp"])
    conf = np.concatenate(stats["conf"])
    pred_cls = np.concatenate(stats["pred_cls"])
    target_cls = np.concatenate(stats["target_cls"])
    res = ap_per_class(tp, conf, pred_cls, target_cls)
    ap = res[5] if len(res) > 5 else res[-1]  # ap array (nc, 10)
    return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()), "map75": float(ap[:, 5].mean()),
            "images": len(gt["images"]), "classes_with_gt": int(len(ap))}


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1], sys.argv[2]), indent=2))
