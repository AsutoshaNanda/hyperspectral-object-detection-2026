"""Build the 16-band Cascade R-CNN config + 16-channel COCO-pretrained checkpoint (run in /venv/mmdet).

Base: MMDetection cascade-rcnn_r50_fpn_1x_coco. Changes: 16-band input (backbone in_channels=16,
first conv initialised from the COCO RGB kernel mean x 3/16 -- same "rgb_mean" trick as Combo),
18 classes, per-band mean/std from TRAIN pixels, long side resized to 1024 (Combo's imgsz),
horizontal flip, 24 epochs (2x schedule), score_thr 0.001 / max 300 boxes (same as Combo eval).
"""
import json
import urllib.request
from pathlib import Path

import mmdet
import torch
from mmengine.config import Config

W = Path("/workspace/cascade")
D = W / "data"
stats = json.loads((D / "pixel_stats.json").read_text())
classes = tuple(stats["classes"])

base = Path(mmdet.__file__).parent / ".mim/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(str(base))

# ---- 16-channel COCO-pretrained checkpoint ----
url = ("https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/"
       "cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth")
raw = W / "cascade_rcnn_r50_fpn_1x_coco.pth"
if not raw.exists():
    urllib.request.urlretrieve(url, raw)
ck = torch.load(raw, map_location="cpu")
sd = ck.get("state_dict", ck)
w = sd["backbone.conv1.weight"]
sd["backbone.conv1.weight"] = w.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1) * (3.0 / 16.0)
for k in [k for k in sd if ".fc_cls." in k]:
    del sd[k]  # 81-way COCO classifier -> re-initialised for 18 classes
conv = W / "cascade_r50_coco_16band.pth"
torch.save({"state_dict": sd, "meta": {"source": url, "note": "conv1 rgb_mean x3/16; fc_cls removed"}}, conv)

cfg.custom_imports = dict(imports=["cascade_hsi"], allow_failed_imports=False)
cfg.model.data_preprocessor.mean = None  # normalisation done in LoadNpyImage (16 bands)
cfg.model.data_preprocessor.std = None
cfg.model.data_preprocessor.bgr_to_rgb = False
cfg.model.backbone.in_channels = 16
cfg.model.backbone.init_cfg = None
for head in cfg.model.roi_head.bbox_head:
    head.num_classes = len(classes)
cfg.model.test_cfg.rcnn.score_thr = 0.001
cfg.model.test_cfg.rcnn.max_per_img = 300

train_pipeline = [
    dict(type="LoadNpyImage", mean=stats["mean"], std=stats["std"]),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadNpyImage", mean=stats["mean"], std=stats["std"]),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]


def ds(split, pipeline, test_mode):
    return dict(type="CocoDataset", data_root=str(D), metainfo=dict(classes=classes),
                ann_file=f"{split}.json", data_prefix=dict(img=f"{split}/"),
                filter_cfg=dict(filter_empty_gt=not test_mode, min_size=0),
                pipeline=pipeline, test_mode=test_mode)


cfg.train_dataloader = dict(batch_size=4, num_workers=8, persistent_workers=True,
                            sampler=dict(type="DefaultSampler", shuffle=True),
                            batch_sampler=dict(type="AspectRatioBatchSampler"),
                            dataset=ds("train", train_pipeline, False))
cfg.val_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True, drop_last=False,
                          sampler=dict(type="DefaultSampler", shuffle=False),
                          dataset=ds("val", test_pipeline, True))
cfg.test_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True, drop_last=False,
                           sampler=dict(type="DefaultSampler", shuffle=False),
                           dataset=ds("test", test_pipeline, True))
cfg.val_evaluator = dict(type="CocoMetric", ann_file=str(D / "val.json"), metric="bbox", classwise=False)
cfg.test_evaluator = dict(type="CocoMetric", ann_file=str(D / "test.json"), metric="bbox", classwise=True,
                          outfile_prefix=str(W / "holdout_preds"))

cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=2)
cfg.param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1),
]
cfg.optim_wrapper = dict(type="AmpOptimWrapper", loss_scale="dynamic",
                         optimizer=dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001))
cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)
cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=2, max_keep_ckpts=2,
                                    save_best="coco/bbox_mAP", rule="greater")
cfg.default_hooks.logger = dict(type="LoggerHook", interval=100)
cfg.load_from = str(conv)
cfg.work_dir = str(W / "work")
cfg.randomness = dict(seed=42)
cfg.dump(str(W / "cascade_hsi16.py"))
print("config + 16-band checkpoint written; classes", len(classes))
